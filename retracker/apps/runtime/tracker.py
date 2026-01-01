"""Tracking engine wrapper."""

import torch
import numpy as np
from typing import Tuple, Union, Optional, Dict

from ..config import ModelConfig
from retracker.inference.engine import ReTrackerEngine, StreamingEngine, MultiGPUStreamingEngine
from retracker.utils.profiler import global_profiler, enable_profiling, disable_profiling, print_profile_summary, export_profile_csv
from retracker.utils.rich_utils import CONSOLE


class Tracker:
    """Unified tracking interface."""

    def __init__(self, config: ModelConfig):
        """
        Initialize tracker.

        Args:
            config: ModelConfig instance
        """
        from retracker.utils.device import resolve_device

        self.config = config
        # Resolve "cuda" -> "cpu" on CPU-only machines so inference can still run.
        # (Checkpoints are loaded onto CPU first, then moved to the resolved device.)
        self.config.device = resolve_device(self.config.device)
        if self.config.device == "cpu" and self.config.devices:
            # Multi-GPU streaming requires CUDA devices.
            self.config.devices = None
        self.engine = self._build_engine()
        self._dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }

        # Streaming engine (lazy initialized)
        # Can be StreamingEngine or MultiGPUStreamingEngine
        self._streaming_engine: Optional[Union[StreamingEngine, MultiGPUStreamingEngine]] = None

        # Enable dense matching if configured
        if config.enable_dense_matching:
            self.set_dense_matching(
                enable=True,
                level=config.dense_level
            )

        # Apply task-specific visibility thresholds and default task mode
        self.set_visibility_thresholds(
            tracking=config.tracking_visibility_threshold,
            matching=config.matching_visibility_threshold,
        )
        self.set_task_mode('tracking')

    def _build_engine(self) -> ReTrackerEngine:
        """Initialize tracking engine."""
        # Create engine with compile but without warmup (warmup needs GPU)
        engine = ReTrackerEngine(
            ckpt_path=self.config.ckpt_path,
            interp_shape=self.config.interp_shape,
            enable_highres_inference=self.config.enable_highres_inference,
            coarse_resolution=self.config.coarse_resolution,
            fast_start=self.config.fast_start,
            compile=self.config.compile,
            compile_mode=self.config.compile_mode,
            warmup=False,  # Don't warmup yet - need to move to device first
            query_batch_size=self.config.query_batch_size,
        )
        engine.eval()
        engine.to(self.config.device)

        # Now run warmup after model is on device
        if self.config.compile and self.config.compile_warmup:
            engine.warmup(device=self.config.device)

        return engine

    def set_highres_inference(
        self,
        enable: bool = True,
        coarse_resolution: Tuple[int, int] = (512, 512)
    ) -> None:
        """
        Enable or disable high-resolution inference mode.

        In high-resolution mode:
        - Coarse/global stage uses coarse_resolution (default 512x512)
        - Refinement stage uses original input resolution

        Args:
            enable: Whether to enable high-resolution inference
            coarse_resolution: Resolution (H, W) for coarse stage
        """
        if hasattr(self.engine, 'model') and hasattr(self.engine.model, 'set_highres_inference'):
            self.engine.model.set_highres_inference(enable, coarse_resolution)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Model does not support set_highres_inference")

    def set_dense_matching(self, enable: bool = True, level: int = 2) -> None:
        """
        Enable or disable dense matching output.

        When enabled, each query point outputs W*W (7*7=49) predictions
        representing the dense flow field around the query point.

        Args:
            enable: Whether to enable dense matching
            level: Refinement level for dense matching (0=coarsest, 2=finest)
        """
        if hasattr(self.engine, 'set_dense_matching'):
            self.engine.set_dense_matching(enable, level)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Engine does not support set_dense_matching")

    def set_task_mode(self, task_mode: str) -> None:
        """Set runtime task mode: 'tracking' or 'matching'."""
        if hasattr(self.engine, 'model') and hasattr(self.engine.model, 'set_task_mode'):
            self.engine.model.set_task_mode(task_mode)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Model does not support set_task_mode")

    def set_visibility_thresholds(
        self,
        tracking: Optional[float] = None,
        matching: Optional[float] = None
    ) -> None:
        """Set task-specific visibility thresholds."""
        if hasattr(self.engine, 'model') and hasattr(self.engine.model, 'set_visibility_thresholds'):
            self.engine.model.set_visibility_thresholds(tracking=tracking, matching=matching)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Model does not support set_visibility_thresholds")

    def _compute_patch_offsets(self, patch_size: int = 7, downsample_factor: float = 2.0) -> torch.Tensor:
        """
        Compute the pixel offsets for the 7x7 patch around a query point.

        The patch is at 2x downsampling of the finest feature level, meaning
        each patch point is spaced by `downsample_factor` pixels in image space.

        Args:
            patch_size: Size of the patch (default 7 for 7x7)
            downsample_factor: Spacing between patch points in image space

        Returns:
            offsets: [49, 2] tensor with (dx, dy) offsets for each patch point
        """
        # Generate offsets: for 7x7 patch centered at origin
        # Range: [-(patch_size//2) * factor, ..., (patch_size//2) * factor]
        # For 7x7 with factor=2: [-6, -4, -2, 0, 2, 4, 6]
        half = patch_size // 2
        offset_1d = torch.arange(-half, half + 1).float() * downsample_factor

        # Create 2D grid of offsets (row-major order like the model output)
        dy, dx = torch.meshgrid(offset_1d, offset_1d, indexing='ij')
        offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # [49, 2]

        return offsets

    def expand_queries_for_dense(
        self,
        queries: torch.Tensor,
        downsample_factor: float = 2.0
    ) -> torch.Tensor:
        """
        Expand N query points to N*49 points by adding 7x7 patch around each query.

        This is useful when dense_tracks is enabled - the expanded queries represent
        the 49 patch points whose correspondences are in dense_tracks.

        Args:
            queries: [B, N, 3] tensor with (t, x, y) format
            downsample_factor: Spacing between patch points (default 2.0 for 2x downsampling)

        Returns:
            expanded_queries: [B, N*49, 3] tensor where each original query is expanded
                             to 49 queries representing the 7x7 patch
        """
        B, N, D = queries.shape
        device = queries.device
        dtype = queries.dtype

        # Get patch offsets [49, 2]
        offsets = self._compute_patch_offsets(
            patch_size=self.engine.dense_patch_size if hasattr(self.engine, 'dense_patch_size') else 7,
            downsample_factor=downsample_factor
        ).to(device=device, dtype=dtype)

        patch_size_sq = offsets.shape[0]  # 49

        # Expand queries: [B, N, 3] -> [B, N, 49, 3]
        queries_expanded = queries.unsqueeze(2).repeat(1, 1, patch_size_sq, 1)

        # Add offsets to x (dim 1) and y (dim 2) coordinates
        queries_expanded[..., 1] += offsets[:, 0].view(1, 1, patch_size_sq)  # x offset
        queries_expanded[..., 2] += offsets[:, 1].view(1, 1, patch_size_sq)  # y offset

        # Reshape to [B, N*49, 3]
        expanded = queries_expanded.reshape(B, N * patch_size_sq, D)

        return expanded

    def reshape_dense_tracks(
        self,
        dense_tracks: torch.Tensor,
        dense_vis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape dense_tracks from [B, T, N, 49, 2] to [B, T, N*49, 2].

        This converts the dense output to the same format as regular trajectories,
        making it 49x larger.

        Args:
            dense_tracks: [B, T, N, 49, 2] dense patch predictions
            dense_vis: [B, T, N, 49] dense patch visibility

        Returns:
            trajectories: [B, T, N*49, 2] flattened trajectories
            visibility: [B, T, N*49] flattened visibility
        """
        B, T, N, patch_sq, _ = dense_tracks.shape

        # Reshape: [B, T, N, 49, 2] -> [B, T, N*49, 2]
        trajectories = dense_tracks.reshape(B, T, N * patch_sq, 2)
        visibility = dense_vis.reshape(B, T, N * patch_sq)

        return trajectories, visibility

    def set_max_queries(self, max_queries: int = None) -> None:
        """
        Set the maximum number of query points for inference.

        During training, query points are limited to 500 to save memory.
        For inference with more points, use this method to increase the limit.

        Args:
            max_queries: Maximum number of query points.
                        If None, removes the limit entirely.
        """
        if hasattr(self.engine, 'model') and hasattr(self.engine.model, 'set_max_queries'):
            self.engine.model.set_max_queries(max_queries)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Model does not support set_max_queries")

    def set_query_batch_size(self, batch_size: int) -> None:
        """
        Set the query batch size for OOM prevention.

        When processing more queries than this per frame, they will be split
        into batches. Lower values use less memory but may be slower.

        Args:
            batch_size: Maximum queries to process at once (default: 256)
        """
        if hasattr(self.engine, 'query_batch_size'):
            self.engine.query_batch_size = batch_size
            CONSOLE.print(f"[dim][INFO] Query batch size set to {batch_size}[/dim]")
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Engine does not support query_batch_size")

    def track(
        self,
        video: torch.Tensor,
        queries: torch.Tensor,
        use_aug: bool = False,
        expand_dense: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
        """
        Perform tracking on video.

        Args:
            video: Video tensor of shape (B, T, C, H, W)
            queries: Query points of shape (B, N, 3) in (t, x, y) format
            use_aug: Whether to use test-time augmentation
            expand_dense: If True and dense matching is enabled, expand output to N*49 points
                         Returns a dict with expanded trajectories, visibility, and query coordinates

        Returns:
            When dense matching disabled (default):
                trajectories: Predicted trajectories (B, T, N, 2)
                visibility: Visibility flags (B, T, N)

            When dense matching enabled and expand_dense=False:
                trajectories: Predicted trajectories (B, T, N, 2)
                visibility: Visibility flags (B, T, N)
                dense_tracks: Dense patch predictions (B, T, N, W*W, 2)
                dense_vis: Dense patch visibility (B, T, N, W*W)

            When dense matching enabled and expand_dense=True:
                dict with:
                    'trajectories': (B, T, N*49, 2) - all 49 patch points per query
                    'visibility': (B, T, N*49) - visibility for all patch points
                    'expanded_queries': (B, N*49, 3) - the 49 query coordinates per original query
                    'original_trajectories': (B, T, N, 2) - center point trajectories
                    'original_visibility': (B, T, N) - center point visibility
        """
        dtype = self._dtype_map[self.config.dtype]

        device_type = "cuda" if str(self.config.device).startswith("cuda") else "cpu"
        # AMP is only meaningful for CUDA here. On CPU-only machines, enable=False keeps behavior
        # consistent (and avoids torch warnings about missing CUDA).
        amp_enabled = self.config.use_amp and device_type == "cuda"

        with torch.amp.autocast(device_type, dtype=dtype, enabled=amp_enabled):
            result = self.engine.video_forward(
                video,
                queries,
                use_aug=use_aug
            )

        # Handle both dense and non-dense returns
        if len(result) == 4:
            # Dense matching enabled: (traj, vis, dense_tracks, dense_vis)
            trajectories, visibility, dense_tracks, dense_vis = result

            if expand_dense:
                # Expand to N*49 points
                expanded_traj, expanded_vis = self.reshape_dense_tracks(dense_tracks, dense_vis)

                # Compute expanded query coordinates (the 7x7 patch points on the query frame)
                # Use 2.0 as default downsample factor for finest level
                expanded_queries = self.expand_queries_for_dense(queries, downsample_factor=2.0)

                return {
                    'trajectories': expanded_traj,  # [B, T, N*49, 2]
                    'visibility': expanded_vis,      # [B, T, N*49]
                    'expanded_queries': expanded_queries,  # [B, N*49, 3]
                    'original_trajectories': trajectories,  # [B, T, N, 2]
                    'original_visibility': visibility,       # [B, T, N]
                    'n_original_queries': queries.shape[1],
                    'patch_size': 7,
                    'expansion_factor': 49,
                }
            else:
                return result
        else:
            # Standard: (traj, vis)
            trajectories, visibility = result
            return trajectories, visibility

    def get_visibility_scores(self) -> torch.Tensor:
        """
        Get raw visibility scores from the last track() call.

        Returns:
            visibility_scores: [B, T, N] float tensor with raw visibility scores
                              (certainty * (1 - occlusion))
        """
        return self.engine.last_visibility_scores

    def track_dense(
        self,
        video: torch.Tensor,
        queries: torch.Tensor,
        use_aug: bool = False,
        downsample_factor: float = 2.0
    ) -> Dict:
        """
        Track with dense matching enabled, returning 49x expanded output.

        This is a convenience method that:
        1. Temporarily enables dense matching
        2. Runs tracking
        3. Returns expanded trajectories (N*49 points per N queries)

        Args:
            video: Video tensor of shape (B, T, C, H, W)
            queries: Query points of shape (B, N, 3) in (t, x, y) format
            use_aug: Whether to use test-time augmentation
            downsample_factor: Spacing between patch points (default 2.0)

        Returns:
            dict with:
                'trajectories': (B, T, N*49, 2) - all 49 patch points per query
                'visibility': (B, T, N*49) - visibility for all patch points
                'expanded_queries': (B, N*49, 3) - the 49 query coordinates per original query
                'original_trajectories': (B, T, N, 2) - center point trajectories
                'original_visibility': (B, T, N) - center point visibility
                'n_original_queries': N
                'patch_size': 7
                'expansion_factor': 49

        Example:
            >>> # Track 100 points, get 4900 correspondences
            >>> result = tracker.track_dense(video, queries)
            >>> print(f"Input: {queries.shape[1]} queries")
            >>> print(f"Output: {result['trajectories'].shape[2]} tracks")
            # Input: 100 queries
            # Output: 4900 tracks
        """
        # Ensure dense matching is enabled
        was_enabled = self.dense_matching_enabled
        if not was_enabled:
            self.set_dense_matching(enable=True)

        try:
            result = self.track(video, queries, use_aug=use_aug, expand_dense=True)

            # Update downsample factor if different from default
            if downsample_factor != 2.0 and isinstance(result, dict):
                # Recompute expanded queries with correct downsample factor
                result['expanded_queries'] = self.expand_queries_for_dense(
                    queries, downsample_factor=downsample_factor
                )
                result['downsample_factor'] = downsample_factor

            return result
        finally:
            # Restore previous state
            if not was_enabled:
                self.set_dense_matching(enable=False)

    @property
    def dense_matching_enabled(self) -> bool:
        """Check if dense matching is currently enabled."""
        return getattr(self.engine, 'enable_dense_matching', False)

    # ========== Streaming Mode Methods ==========

    def _get_streaming_engine(self) -> Union[StreamingEngine, MultiGPUStreamingEngine]:
        """Get or create streaming engine (lazy initialization)."""
        if self._streaming_engine is None:
            # Check if multi-GPU is configured
            if self.config.devices and len(self.config.devices) >= 2:
                # Use multi-GPU streaming engine
                # Note: Only works with reset mode, not add_queries mode
                self._streaming_engine = MultiGPUStreamingEngine(
                    devices=list(self.config.devices),
                    ckpt_path=self.config.ckpt_path,
                    interp_shape=self.config.interp_shape,
                    enable_highres_inference=self.config.enable_highres_inference,
                    coarse_resolution=self.config.coarse_resolution,
                    query_batch_size=self.config.query_batch_size,
                    fast_start=self.config.fast_start,
                )
                CONSOLE.print(
                    f"[dim][MultiGPU] Using {len(self.config.devices)} GPUs: {self.config.devices}[/dim]"
                )
            else:
                # Use single GPU streaming engine
                self._streaming_engine = StreamingEngine(
                    retracker_model=self.engine.model,
                    interp_shape=self.config.interp_shape,
                    query_batch_size=self.config.query_batch_size,
                )
                self._streaming_engine.to(self.config.device)
        return self._streaming_engine

    def streaming_init(
        self,
        video_shape: Tuple[int, int, int],
        initial_queries: torch.Tensor = None
    ):
        """
        Initialize streaming mode for per-frame tracking with memory.

        Args:
            video_shape: (C, H, W) shape of video frames
            initial_queries: Optional [N, 3] tensor with (t, x, y) format
        """
        streaming = self._get_streaming_engine()
        streaming.initialize(video_shape, initial_queries)

    def streaming_process_frame(
        self,
        frame: torch.Tensor,
        use_aug: bool = False
    ) -> dict:
        """
        Process a single frame in streaming mode (uses memory from previous frames).

        Args:
            frame: [C, H, W] tensor of current frame
            use_aug: Whether to use augmentation

        Returns:
            dict with 'tracks' [N, 2] and 'visibility' [N] for current frame
        """
        if self._streaming_engine is None or not self._streaming_engine.is_initialized:
            raise RuntimeError("Streaming mode not initialized. Call streaming_init() first.")

        return self._streaming_engine.process_frame(frame, use_aug)

    def streaming_add_queries(
        self,
        new_queries: torch.Tensor,
        frame_idx: int = None
    ) -> torch.Tensor:
        """
        Add new query points in streaming mode.

        Args:
            new_queries: [N_new, 2] tensor with (x, y) coordinates
            frame_idx: Frame index for new queries (default: current frame)

        Returns:
            query_ids: Tensor of assigned query IDs
        """
        if self._streaming_engine is None or not self._streaming_engine.is_initialized:
            raise RuntimeError("Streaming mode not initialized. Call streaming_init() first.")

        return self._streaming_engine.add_queries(new_queries, frame_idx)

    def streaming_get_tracks(self, query_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get tracking history in streaming mode.

        Args:
            query_ids: Optional tensor of query IDs to filter

        Returns:
            tracks: [T, N, 2] trajectory history
            visibility: [T, N] visibility history
        """
        if self._streaming_engine is None or not self._streaming_engine.is_initialized:
            raise RuntimeError("Streaming mode not initialized.")

        return self._streaming_engine.get_tracks(query_ids)

    def streaming_reset(self):
        """Reset streaming mode state."""
        if self._streaming_engine is not None:
            self._streaming_engine.reset()

    @property
    def streaming_current_frame(self) -> int:
        """Get current frame index in streaming mode."""
        if self._streaming_engine is None:
            return 0
        return self._streaming_engine.current_frame_idx

    @property
    def streaming_num_queries(self) -> int:
        """Get number of active queries in streaming mode."""
        if self._streaming_engine is None or self._streaming_engine.queries is None:
            return 0
        return self._streaming_engine.queries.shape[1]

    # ========== Profiling Methods ==========

    def enable_profiling(self) -> None:
        """Enable profiling for inference bottleneck analysis.

        When enabled, the profiler will record timing for key operations:
        - forward: Overall forward pass
        - coarse_stage: Global matching stage
        - fine_stage: Refinement stage
        - extract_features: Backbone feature extraction
        - matches_refinement: TROMA refinement iterations

        Usage:
            tracker.enable_profiling()
            # Run tracking...
            tracker.print_profile_summary()
        """
        enable_profiling()

    def disable_profiling(self) -> None:
        """Disable profiling."""
        disable_profiling()

    def print_profile_summary(self) -> None:
        """Print profiling summary after inference."""
        print_profile_summary()

    def export_profile_csv(self, filepath: str) -> None:
        """Export profiling results to CSV file."""
        export_profile_csv(filepath)

    def reset_profiler(self) -> None:
        """Reset profiler data for a new profiling session."""
        global_profiler.reset()

    # ========== torch.compile Methods ==========

    def compile_model(self, mode: str = "reduce-overhead") -> None:
        """Compile the model using torch.compile for faster inference.

        This can provide 20-50% speedup for transformer components.

        Args:
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        """
        if hasattr(self.engine, 'compile_model'):
            self.engine.compile_model(mode=mode)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Engine does not support compile_model")

    def __repr__(self) -> str:
        """String representation."""
        engine_display = getattr(self.engine, "display_name", self.engine.__class__.__name__)
        return (
            f"Tracker(\n"
            f"  engine={engine_display},\n"
            f"  ckpt={self.config.ckpt_path},\n"
            f"  device={self.config.device},\n"
            f"  dtype={self.config.dtype},\n"
            f"  dense_matching={self.dense_matching_enabled}\n"
            f")"
        )

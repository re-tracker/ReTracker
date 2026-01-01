"""Offline inference engine implementation (ReTrackerEngine).

The stable import path is still `retracker.inference.engine`.
"""

from typing import Optional, Tuple, Union
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from retracker.models import ReTracker
from retracker.models import get_cfg_defaults
from retracker.data.video_sequences import build_pairs, build_triplets
from retracker.utils.checkpoint import safe_torch_load
from retracker.utils.rich_utils import CONSOLE, get_progress, track
from retracker.utils.profiler import global_profiler, enable_profiling, disable_profiling, print_profile_summary, export_profile_csv


class ReTrackerEngine(torch.nn.Module):
    """Offline inference engine for ReTracker."""

    display_name: str = "retracker"

    def __init__(
        self,
        retracker_model: Union[ReTracker, None]=None,
        ckpt_path: Union[str, None]=None,
        locotrack_ckpt_path: Union[str, None]=None,
        interp_shape: Tuple=(256, 256),
        enable_highres_inference: bool = False,
        coarse_resolution: Tuple[int, int] = (512, 512),
        fast_start: bool = False,
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        warmup: bool = False,
        query_batch_size: int = 256,
        # Backward-compatible alias (old project name). Prefer `retracker_model`.
        kttr_model: Union[ReTracker, None]=None,
    ) -> None:
        """Initialize ReTrackerEngine.

        Args:
            retracker_model: Pre-built ReTracker model (optional)
            ckpt_path: Path to checkpoint file
            locotrack_ckpt_path: Path to LocoTrack checkpoint (optional)
            interp_shape: Interpolation shape for inference
            enable_highres_inference: If True, enable high-resolution inference in the underlying model.
                                     This keeps the coarse/global stage at `coarse_resolution` while
                                     running refinement at the (possibly larger) input resolution.
            coarse_resolution: Resolution (H, W) used by the model coarse/global stage when
                               high-resolution inference is enabled.
            compile: If True, compile model with torch.compile during initialization
            compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
            warmup: If True, run warmup inference after compilation to trigger JIT
            query_batch_size: Maximum number of query points to process at once (for OOM prevention)
        """
        super().__init__()
        self.interp_shape = interp_shape
        self._compiled = False
        self.query_batch_size = query_batch_size
        # Store raw visibility scores (float) from last forward pass
        self.last_visibility_scores = None

        # Accept the historical keyword `kttr_model` used by older scripts/tests.
        if retracker_model is None and kttr_model is not None:
            retracker_model = kttr_model

        if retracker_model is None:
            if ckpt_path is None:
                raise ValueError(
                    "ckpt_path is required when constructing ReTrackerEngine without a pre-built model.\n"
                    "Provide a checkpoint via:\n"
                    "  - `python -m retracker apps tracking --ckpt_path path/to/model.ckpt ...`\n"
                    "  - or YAML: `model.ckpt_path: path/to/model.ckpt`\n"
                    "See docs/MODEL_ZOO.md for available checkpoints."
                )
            cfg = get_cfg_defaults()
            if fast_start:
                # Speed up cold-start by skipping the optional fine CNN branch of
                # hybrid DINOv3 backbones (e.g. ConvNeXt-Tiny).
                #
                # This trades tracking quality for startup time and memory use.
                try:
                    cfg.dino.use_cnn_for_fine_features = False
                    cfg.dino.use_fusion = False
                except Exception:
                    # Best-effort: keep the default config if it cannot be patched.
                    pass
            retracker_model = ReTracker(cfg)
            self.load_state_dict_from_pl_checkpoint(retracker_model, ckpt_path)
            CONSOLE.print('load ReTracker checkpoint successfully')
            if locotrack_ckpt_path is not None:
                # Always load checkpoints onto CPU first so CPU-only environments can
                # still run inference (the model is moved to the requested device later).
                locotrack_ckpt = safe_torch_load(locotrack_ckpt_path, map_location="cpu")
                locotrack_dict = locotrack_ckpt["state_dict"] if "state_dict" in locotrack_ckpt else locotrack_ckpt
                locotrack_dict = {k.replace('model.',''):v for k,v in locotrack_dict.items()}
                retracker_model.locotrack[0].load_state_dict(locotrack_dict)
        else:
            CONSOLE.print('ReTrackerEngine use ReTracker model passed from PL')

        self.model = retracker_model

        self.model.eval()
        self.model.mem_manager.sample_method = 'foremost'
        CONSOLE.print(f"[INFO] Memory manager max size: {self.model.mem_manager.MAX_MEMORY_SIZE}")
        self.bidirectional: bool = False

        # Remove query point limit for inference (default training limit is 500)
        self.model.set_max_queries(None)

        # Optional: keep coarse/global stage at `coarse_resolution` while refining at original resolution.
        # Default is off for backward compatibility.
        if enable_highres_inference:
            self.set_highres_inference(enable=True, coarse_resolution=coarse_resolution)

        # Dense matching settings
        # When enabled, outputs W*W points per query (patch-level dense matching)
        # Default: False (only output single point per query)
        self.enable_dense_matching: bool = False
        self.dense_patch_size: int = 7  # W = corr_radius * 2 + 1 = 7
        self.dense_level: int = 2  # Use last level (finest) for dense matching

        # Compile model if requested
        if compile:
            self.compile_model(mode=compile_mode)
            if warmup:
                self.warmup()

    def set_highres_inference(
        self, enable: bool = True, coarse_resolution: Tuple[int, int] = (512, 512)
    ) -> None:
        """Enable or disable high-resolution inference mode on the underlying model."""
        if hasattr(self.model, "set_highres_inference"):
            self.model.set_highres_inference(enable=enable, coarse_resolution=coarse_resolution)
        else:
            CONSOLE.print("[yellow][WARNING][/yellow] Model does not support set_highres_inference")

    def compile_model(self, mode: str = "reduce-overhead", fullgraph: bool = False) -> None:
        """Compile the model using torch.compile for faster inference.

        This can provide significant speedups (20-50%) for transformer-based
        components like TROMA blocks.

        Args:
            mode: Compilation mode. Options:
                - "default": Good balance of compile time and speedup
                - "reduce-overhead": Best for inference (recommended)
                - "max-autotune": Maximum optimization, longer compile time
            fullgraph: If True, requires the entire model to be compilable.
                      If False (default), allows graph breaks for unsupported ops.

        Note:
            - First inference will be slower due to compilation
            - Subsequent inferences will be significantly faster
            - Dynamic shapes may cause recompilation

        Example:
            engine = ReTrackerEngine(ckpt_path="...")
            engine.compile_model(mode="reduce-overhead")
            # First call triggers compilation (slow)
            result = engine(video, queries)
            # Subsequent calls are much faster
        """
        if self._compiled:
            CONSOLE.print("[torch.compile] Model already compiled, skipping")
            return

        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        # Increase cache size to avoid recompilation due to dynamic structures
        torch._dynamo.config.cache_size_limit = 64  # Default is 8

        CONSOLE.print(f"[torch.compile] Compiling model with mode='{mode}', fullgraph={fullgraph}")
        CONSOLE.print("[torch.compile] First inference will be slower due to compilation...")

        try:
            # Compile the TROMA blocks (main bottleneck)
            refinement = self.model.matches_refinement
            for level_name in ['16x', '8x', '2x']:
                troma_block = getattr(refinement, f'troma_flow_{level_name}', None)
                if troma_block is not None:
                    compiled_block = torch.compile(troma_block, mode=mode, fullgraph=fullgraph)
                    setattr(refinement, f'troma_flow_{level_name}', compiled_block)
                    CONSOLE.print(f"  - Compiled troma_flow_{level_name}")

            # NOTE: Backbone (DINOv3) compilation disabled - adds ~7s warmup with minimal benefit
            # The backbone is already efficient and doesn't benefit much from torch.compile
            # Uncomment below if you want to experiment with backbone compilation:
            # if hasattr(self.model, 'backbone') and self.model.use_dinov3:
            #     self.model.backbone = torch.compile(self.model.backbone, mode=mode, fullgraph=fullgraph)
            #     CONSOLE.print("  - Compiled backbone (DINOv3)")

            self._compiled = True
            CONSOLE.print("[torch.compile] Compilation setup complete")

        except Exception as e:
            CONSOLE.print(f"[torch.compile] Warning: Compilation failed: {e}")
            CONSOLE.print("[torch.compile] Falling back to eager mode")

    @torch.no_grad()
    def warmup(self, device: str = "cuda", num_queries: int = 400) -> None:
        """Run warmup inference to trigger JIT compilation.

        This method runs dummy forward passes to trigger the actual
        compilation of torch.compile'd modules. After warmup, subsequent
        inferences will be much faster.

        Args:
            device: Device to run warmup on
            num_queries: Number of query points to use (should match typical inference)
        """
        if not self._compiled:
            CONSOLE.print("[Warmup] Model not compiled, skipping warmup")
            return

        CONSOLE.print(f"[Warmup] Running warmup to trigger JIT compilation (N={num_queries})...")

        # Create dummy data matching expected input shapes
        # Use shapes that match real inference to avoid recompilation later
        H, W = self.interp_shape
        image0 = torch.randn(1, 3, H, W, device=device)  # B=1, C=3, H, W
        image1 = torch.randn(1, 3, H, W, device=device)

        # Generate grid queries matching typical inference
        # queries shape: (B, N, 2) with (x, y) only - NO time dimension!
        grid_size = int(num_queries ** 0.5)
        xs = torch.linspace(10, W - 10, grid_size, device=device)
        ys = torch.linspace(10, H - 10, grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)
        dummy_queries = coords.unsqueeze(0)  # (1, N, 2)

        # Run warmup iterations with progress bar
        num_warmup_steps = 3
        total_steps = num_warmup_steps * 3  # 3 TROMA blocks per iteration

        progress = get_progress("{task.description}")
        with progress:
            task = progress.add_task("[Warmup] Compiling TROMA blocks", total=total_steps)
            for step in range(num_warmup_steps):
                try:
                    # Build data dict with all required keys
                    data = {
                        'image0': image0,
                        'image1': image1,
                        'images': torch.stack([image0, image1], dim=1),  # (B, T, C, H, W)
                        'queries': dummy_queries,  # (B, N, 2) - only (x, y), no time
                    }

                    # Run forward pass - this triggers compilation of each TROMA block
                    self.model.mem_manager.reset_all_memory()
                    _ = self.model.forward(data, mode='test')

                    # Each forward triggers compilation of 3 TROMA blocks.
                    progress.update(task, advance=3)

                except Exception as e:
                    CONSOLE.print(f"[Warmup] Warning: Warmup step {step} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        # Clear memory after warmup
        self.model.mem_manager.reset_all_memory()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

        CONSOLE.print("[Warmup] Compilation complete! Subsequent inferences will be fast.")

    def set_dense_matching(self, enable: bool = True, level: int = 2) -> None:
        """
        Enable or disable dense matching output.

        When enabled, each query point will output W*W predictions instead of 1,
        representing the dense flow field around the query point.

        Args:
            enable: Whether to enable dense matching
            level: Which refinement level to use (0=coarsest, 2=finest). Default=2
        """
        self.enable_dense_matching = enable
        self.dense_level = level
        if enable:
            CONSOLE.print(f"[DenseMatching] Enabled, using level {level}, patch size {self.dense_patch_size}x{self.dense_patch_size}")
        else:
            CONSOLE.print("[DenseMatching] Disabled")

    def set_task_mode(self, task_mode: str) -> None:
        """Set runtime task mode: 'tracking' or 'matching'."""
        if hasattr(self.model, 'set_task_mode'):
            self.model.set_task_mode(task_mode)
        else:
            CONSOLE.print("[Warning] Model does not support set_task_mode")

    def set_visibility_thresholds(
        self,
        tracking: Optional[float] = None,
        matching: Optional[float] = None
    ) -> None:
        """Set task-specific visibility thresholds."""
        if hasattr(self.model, 'set_visibility_thresholds'):
            self.model.set_visibility_thresholds(tracking=tracking, matching=matching)
        else:
            CONSOLE.print("[Warning] Model does not support set_visibility_thresholds")

    def enable_profiling(self) -> None:
        """Enable profiling for inference bottleneck analysis.

        When enabled, the profiler will record timing for key operations:
        - forward: Overall forward pass
        - coarse_stage: Global matching stage (DINO backbone, affinity, GPS)
        - fine_stage: Refinement stage
        - extract_features: Backbone feature extraction
        - matches_refinement: TROMA refinement iterations

        Usage:
            engine.enable_profiling()
            # Run inference...
            engine.print_profile_summary()
            engine.export_profile_csv("profile.csv")
        """
        enable_profiling()

    def disable_profiling(self) -> None:
        """Disable profiling."""
        disable_profiling()

    def print_profile_summary(self) -> None:
        """Print profiling summary after inference.

        Shows timing breakdown for each operation including:
        - Total time and percentage
        - Mean time per call
        - Min/max times
        - Hierarchical breakdown
        """
        print_profile_summary()

    def export_profile_csv(self, filepath: str) -> None:
        """Export profiling results to CSV file.

        Args:
            filepath: Path to output CSV file

        The CSV contains columns:
        - section: Name of the profiled section
        - depth: Hierarchy depth (0 = root)
        - count: Number of calls
        - total_ms, mean_ms, min_ms, max_ms, std_ms: Timing statistics
        """
        export_profile_csv(filepath)

    def reset_profiler(self) -> None:
        """Reset profiler data for a new profiling session."""
        global_profiler.reset()

    def _generate_patch_offsets(self, device: torch.device) -> Tensor:
        """
        Generate relative offsets for a W*W patch centered at (0, 0).

        Returns:
            Tensor of shape (W*W, 2) with (dx, dy) offsets
        """
        W = self.dense_patch_size
        half_W = W // 2
        # Generate grid from -half_W to +half_W
        offsets_1d = torch.arange(-half_W, half_W + 1, device=device, dtype=torch.float32)
        # Create meshgrid (dy, dx order)
        grid_y, grid_x = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        # Flatten to (W*W, 2) in (x, y) format
        offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        return offsets  # (49, 2)

    def _expand_queries_to_patches(self, queries: Tensor) -> Tuple[Tensor, int]:
        """
        Expand N query points to N*W*W patch points.

        Args:
            queries: (B, N, 3) with (t, x, y) format

        Returns:
            expanded_queries: (B, N*W*W, 3)
            original_n: Original number of queries
        """
        B, N, _ = queries.shape
        device = queries.device
        W = self.dense_patch_size
        WW = W * W

        # Get patch offsets (W*W, 2)
        offsets = self._generate_patch_offsets(device)

        # Expand queries: (B, N, 1, 3) + (1, 1, W*W, 2) for xy part
        queries_expanded = queries.unsqueeze(2).repeat(1, 1, WW, 1)  # (B, N, W*W, 3)

        # Add offsets to x, y coordinates (indices 1, 2)
        queries_expanded[..., 1:3] = queries_expanded[..., 1:3] + offsets.unsqueeze(0).unsqueeze(0)

        # Reshape to (B, N*W*W, 3)
        queries_expanded = queries_expanded.reshape(B, N * WW, 3)

        return queries_expanded, N

    def _extract_dense_flow_from_result(self, res: dict, original_n: int) -> Tuple[Tensor, Tensor]:
        """
        Extract dense flow predictions from updated_pos_nlvl_flow.

        The model outputs updated_pos_nlvl_flow with shape:
            (N, nlvl=3, T, W*W=49, 2)
        We extract the finest level (level 2) and reshape.

        NOTE: Returned coordinates are in interp_shape space. Caller must handle
        scaling to original image space.

        Args:
            res: Model output dict containing 'updated_pos_nlvl_flow'
            original_n: Original number of query points (before padding)

        Returns:
            dense_tracks: (B=1, T, N, W*W, 2) - dense predictions in interp_shape space
            dense_vis: (B=1, T, N, W*W) - visibility for each dense point
        """
        # Get the dense flow output
        # Shape: (N_padded, nlvl=3, T, 49, 2)
        dense_flow = res.get('updated_pos_nlvl_flow', None)
        dense_occ = res.get('updated_occ_nlvl_flow', None)

        if dense_flow is None:
            CONSOLE.print("[Warning] updated_pos_nlvl_flow not found in model output")
            return None, None

        # Extract the specified level (default: level 2 = finest)
        # dense_flow: (N, nlvl, T, 49, 2) -> (N, T, 49, 2)
        level = self.dense_level
        dense_flow_level = dense_flow[:, level, :, :, :]  # (N, T, 49, 2)

        if dense_occ is not None:
            dense_occ_level = dense_occ[:, level, :, :, :]  # (N, T, 49, 1)
            # Convert to visibility: sigmoid and threshold
            dense_vis = (torch.sigmoid(dense_occ_level.squeeze(-1)) > 0.5)  # (N, T, 49)
        else:
            dense_vis = torch.ones(dense_flow_level.shape[:-1], dtype=torch.bool, device=dense_flow_level.device)

        # Take only original_n points (before padding)
        dense_flow_level = dense_flow_level[:original_n]  # (N, T, 49, 2)
        dense_vis = dense_vis[:original_n]  # (N, T, 49)

        # Rearrange to (B=1, T, N, 49, 2)
        T = dense_flow_level.shape[1]
        dense_tracks = dense_flow_level.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T, N, 49, 2)
        dense_vis = dense_vis.permute(1, 0, 2).unsqueeze(0)  # (1, T, N, 49)

        # Return in interp_shape space - caller handles scaling
        return dense_tracks, dense_vis
       
    def _preprocess_frame_chunk(self, video_chunk: Tensor, device: torch.device):
        """
        Helper to process a chunk of video frames on demand.
        Ensures consistent FP32 processing.
        """
        # video_chunk: [B, T_chunk, C, H, W] on CPU
        B, T_chunk, C, H, W = video_chunk.shape
        
        # 1. 显存优化关键：按需移动到 GPU
        video_chunk = video_chunk.to(device)
        
        # 2. 精度保证：强制转换为 float32 进行计算
        # NOTE: use in-place division to avoid a second full-sized GPU allocation.
        video_chunk = video_chunk.float()
        video_chunk.div_(255.0)
        
        # 3. 插值
        video_chunk = video_chunk.reshape(B * T_chunk, C, H, W)
        video_chunk = F.interpolate(video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=False)
        video_chunk = video_chunk.reshape(B, T_chunk, 3, self.interp_shape[0], self.interp_shape[1])
        
        return video_chunk

    @torch.no_grad()
    def forward(self, video: Tensor, queries: Tensor, pair_batch_size: int = 16):
        """ online mode
        """
        # 显存优化：输入留在 CPU
        video = video.cpu() 
        B, T, C, H, W = video.shape
        B, N, D = queries.shape
        assert D == 3, f"queries should have 3 channels (t, x, y), got {D}"
        
        device = next(self.model.parameters()).device
        is_cuda = device.type == "cuda"
        self.model.to(device)

        # Queries 比较小，可以放在 GPU 上
        queries = queries.clone().to(device)

        # ==================================================
        # [ADD] 强行凑够 200 个点
        # ==================================================
        queries, valid_mask = self._pad_queries_with_grid(queries, H, W, target_n=300)
        B, N, D = queries.shape # 更新 N
        # ==================================================


        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        occlusion_pred = []
        tracks = []
        tracks_idx = []
        
        queries_idx = torch.arange(N, device=device)
        selected_images_ids = torch.unique(queries[0,:,0]) 
        
        for idx, q_frame_idx in track(
            enumerate(selected_images_ids),
            description="Processing Query Frames",
            total=int(selected_images_ids.numel()),
        ):
            q_frame_idx_int = int(q_frame_idx.item())
            
            mask = queries[0,:,0] == q_frame_idx
            _queries = queries[0, mask]
            _queries_idx = queries_idx[mask]
            
            fwd_occlusion_pred, fwd_tracks = [], []
            bwd_occlusion_pred, bwd_tracks = [], []

            # --- 1. Forward-flow ---
            forward_pairs_ids = build_pairs(q_frame_idx, T, True)
            forward_pairs_ids = build_triplets(forward_pairs_ids, is_forward=True) 
            
            self.model.mem_manager.reset_all_memory()
            
            # 批量处理 Pairs
            for i in range(0, len(forward_pairs_ids), pair_batch_size):
                batch_ids = forward_pairs_ids[i : i + pair_batch_size]
                if len(batch_ids) == 0: continue

                # 按需加载 GPU -> Resize -> FP32
                raw_frames = video[0, batch_ids] 
                processed_batch = self._preprocess_frame_chunk(raw_frames.unsqueeze(0), device)
                _image_pairs = processed_batch[0] 
                
                for idy, _image_pair in enumerate(_image_pairs):
                    # === 修复：移除 autocast，保持 FP32 ===
                    _fwd_occlusion_pred, _fwd_tracks = \
                        self.pairs_forward(_image_pair[None], _queries[None])
                    
                    fwd_occlusion_pred.append(_fwd_occlusion_pred[0].cpu())
                    fwd_tracks.append(_fwd_tracks[0].cpu())
                
                del processed_batch, raw_frames, _image_pairs
            
            # --- 2. Backward-flow ---
            backward_pairs_ids = build_pairs(q_frame_idx, T, False)
            backward_pairs_ids = build_triplets(backward_pairs_ids, is_forward=True)
            
            self.model.mem_manager.reset_all_memory()
            
            for i in range(0, len(backward_pairs_ids), pair_batch_size):
                batch_ids = backward_pairs_ids[i : i + pair_batch_size]
                if len(batch_ids) == 0: continue

                raw_frames = video[0, batch_ids]
                processed_batch = self._preprocess_frame_chunk(raw_frames.unsqueeze(0), device)
                _image_pairs = processed_batch[0]

                for idy, _image_pair in enumerate(_image_pairs):
                    if self.bidirectional:
                        # === 修复：移除 autocast，保持 FP32 ===
                        _bwd_occlusion_pred, _bwd_tracks = \
                            self.pairs_forward(_image_pair[None],  _queries[None])
                        bwd_occlusion_pred.append(_bwd_occlusion_pred[0].cpu())
                        bwd_tracks.append(_bwd_tracks[0].cpu())
                    else:
                        _N = _queries.shape[0]
                        bwd_occlusion_pred.append(torch.zeros((_N, 1), device='cpu'))
                        bwd_tracks.append(torch.zeros((_N, 2), device='cpu'))
                
                del processed_batch, raw_frames, _image_pairs

            # --- 3. Concatenate ---
            bwd_occlusion_pred.reverse() 
            bwd_tracks.reverse() 

            if (len(bwd_tracks) + len(fwd_tracks)) > 0:
                occlusion_pred.append(torch.stack(bwd_occlusion_pred + fwd_occlusion_pred))
                tracks.append(torch.stack(bwd_tracks + fwd_tracks)) 
                tracks_idx.append(_queries_idx.cpu())
            
            if is_cuda:
                torch.cuda.empty_cache()

        if len(tracks) == 0:
            return torch.empty(0), torch.empty(0)

        tracks = torch.cat(tracks, dim=1) 
        occlusion_pred = torch.cat(occlusion_pred, dim=1)

        tracks_idx = torch.cat(tracks_idx, dim=0)
        new_perm = torch.argsort(tracks_idx)
        tracks, occlusion_pred = map(lambda x: x[:, new_perm], [tracks, occlusion_pred])

        tracks = tracks.reshape(1, -1, N, 2)
        occlusion_pred = occlusion_pred.reshape(1, -1, N)

        traj_e = tracks
        vis_e = occlusion_pred

        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        
        return traj_e, vis_e

    def pairs_forward(self, image_pair: Tensor, queries: Tensor):
        data = {
            'image0': image_pair[:,0], 
            'image_m': image_pair[:,1], 
            'image1': image_pair[:,2], 
            'queries': queries[:, :, 1:] 
            }
        B, N = queries.shape[:2]
        self.model(data) 
        traj_e = data['mkpts1_f'].reshape(B, N, 2)
        pred_visibles = data['pred_visibles'].reshape(B, N, 1)
        return pred_visibles, traj_e
 

    @torch.no_grad()
    def video_forward(self, video: Tensor, queries: Tensor, use_aug: bool=False):
        """
        Inference forward pass with memory optimization. BF16 Mode.

        When dense matching is enabled (self.enable_dense_matching=True), returns additional
        dense flow predictions for W*W patch around each query point.

        Returns:
            When dense matching disabled (default):
                traj_e: [B, T, N, 2] - center point trajectories
                vis_e: [B, T, N] - visibility predictions

            When dense matching enabled:
                traj_e: [B, T, N, 2] - center point trajectories
                vis_e: [B, T, N] - visibility predictions
                dense_tracks: [B, T, N, W*W, 2] - dense patch predictions
                dense_vis: [B, T, N, W*W] - dense patch visibility
        """

        video = video.cpu()
        B, T, C, H, W = video.shape
        # 1. 获取原始 N (这是我们需要保存结果的真实点数)
        B_q, N_orig, D = queries.shape
        assert D == 3, f"queries should have 3 channels (t, x, y), got {D}"

        # Handle multi-batch by processing each batch independently
        if B > 1 or B_q > 1:
            return self._video_forward_multibatch(video, queries, use_aug)

        device = next(self.model.parameters()).device
        is_cuda = device.type == "cuda"
        queries = queries.clone().to(device)

        # 目标填充点数
        TARGET_N = 10

        # ==================================================
        # [STEP 1] 坐标预处理: 归一化到 interp_shape
        # ==================================================
        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        # ==================================================
        # [STEP 2] Init Buffers: 基于原始 N_orig 初始化结果容器
        # ==================================================
        tracks = torch.zeros((B, T, N_orig, 2), dtype=torch.float32, device='cpu')
        occlusion_pred = torch.zeros((B, T, N_orig), dtype=torch.bool, device='cpu')
        visibility_scores_buffer = torch.zeros((B, T, N_orig), dtype=torch.float32, device='cpu')  # Raw float scores

        # Dense matching buffers (only if enabled)
        WW = self.dense_patch_size * self.dense_patch_size  # 7*7 = 49
        if self.enable_dense_matching:
            dense_tracks = torch.zeros((B, T, N_orig, WW, 2), dtype=torch.float32, device='cpu')
            dense_vis = torch.zeros((B, T, N_orig, WW), dtype=torch.bool, device='cpu')

        queries_idx = torch.arange(N_orig, device=device)
        selected_images_ids = torch.unique(queries[0,:,0])

        mode_str = "BF16+Dense" if self.enable_dense_matching else "BF16"
        desc_base = f"Query Frames [Mode: {mode_str} | Batch: {TARGET_N}]"
        progress = get_progress("{task.description}")
        task = progress.add_task(desc_base, total=int(selected_images_ids.numel()))
        progress.start()

        for idx, q_frame_idx in enumerate(selected_images_ids):
            q_frame_idx = int(q_frame_idx.item())

            # 1. 筛选当前帧出发的真实点
            mask = queries[0,:,0] == q_frame_idx
            _queries_real = queries[0, mask]      # [N_subset, 3]
            _queries_idx = queries_idx[mask]      # 记录这些点在全局结果中的索引

            N_subset = _queries_real.shape[0]

            # ==================================================
            # [Query Batching] Split queries to prevent OOM
            # ==================================================
            if N_subset > self.query_batch_size:
                # Process in batches to prevent OOM
                num_batches = (N_subset + self.query_batch_size - 1) // self.query_batch_size

                # Preprocess video once (shared across all query batches)
                raw_segment = video[:, q_frame_idx:, ...].clone()
                images_gpu = self._preprocess_frame_chunk(raw_segment, device)
                del raw_segment

                batch_traj_list = []
                batch_vis_list = []
                batch_vis_scores_list = []  # For raw visibility scores
                batch_dense_tracks_list = [] if self.enable_dense_matching else None
                batch_dense_vis_list = [] if self.enable_dense_matching else None

                if is_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.query_batch_size
                    batch_end = min(batch_start + self.query_batch_size, N_subset)

                    _queries_batch = _queries_real[batch_start:batch_end]  # [batch_size, 3]

                    # Reset memory manager for each query batch
                    self.model.mem_manager.reset_all_memory()

                    # Pad queries if needed
                    _queries_padded, _pad_mask = self._pad_queries_with_grid(
                        _queries_batch.unsqueeze(0),
                        H=self.interp_shape[0],
                        W=self.interp_shape[1],
                        target_n=TARGET_N
                    )

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
                        res = self.model.video_forward(
                            {'images': images_gpu, 'queries': _queries_padded[..., 1:]},
                            use_aug=use_aug,
                            return_dense_flow=self.enable_dense_matching
                        )

                    _traj_e = res['mkpts1_f'].detach().float().cpu()
                    _vis_e = res['pred_visibles'].squeeze(-1).detach().cpu()

                    # Extract visibility scores if available
                    _vis_scores_batch = None
                    if 'visibility_scores' in res:
                        _vis_scores_batch = res['visibility_scores'].squeeze(-1).detach().float().cpu()

                    # Filter padding points
                    real_indices_mask = _pad_mask[0].cpu()
                    _traj_real = _traj_e[real_indices_mask, :, :]
                    _vis_real = _vis_e[real_indices_mask, :]

                    batch_traj_list.append(_traj_real)
                    batch_vis_list.append(_vis_real)

                    # Also filter visibility scores
                    if _vis_scores_batch is not None:
                        _vis_scores_real = _vis_scores_batch[real_indices_mask, :]
                        batch_vis_scores_list.append(_vis_scores_real)

                    # Handle dense matching
                    if self.enable_dense_matching:
                        N_real_batch = real_indices_mask.sum().item()
                        _dense_tracks, _dense_vis = self._extract_dense_flow_from_result(res, N_real_batch)
                        if _dense_tracks is not None:
                            batch_dense_tracks_list.append(_dense_tracks[0].float().cpu())
                            batch_dense_vis_list.append(_dense_vis[0].cpu())

                    del res, _traj_e, _vis_e, _queries_padded, _pad_mask

                # Concatenate batch results
                _traj_real = torch.cat(batch_traj_list, dim=0)  # [N_subset, T_seg, 2]
                _vis_real = torch.cat(batch_vis_list, dim=0)    # [N_subset, T_seg]

                # Concatenate visibility scores if available
                _vis_scores_real = None
                if len(batch_vis_scores_list) > 0:
                    _vis_scores_real = torch.cat(batch_vis_scores_list, dim=0)

                if is_cuda:
                    torch.cuda.synchronize()
                end_time = time.time()
                valid_len = _traj_real.shape[1]
                if valid_len > 0:
                    current_fps = valid_len / (end_time - start_time)
                    progress.update(
                        task,
                        description=(
                            f"{desc_base} | FPS: {current_fps:.2f} | Pts: {N_subset} ({num_batches} batches)"
                        ),
                    )

                # Fill into global buffer
                tracks[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _traj_real.transpose(0,1)
                occlusion_pred[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _vis_real.transpose(0,1)

                # Fill visibility scores
                if _vis_scores_real is not None:
                    visibility_scores_buffer[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _vis_scores_real.transpose(0,1)

                # Handle dense matching concatenation
                if self.enable_dense_matching and batch_dense_tracks_list:
                    _dense_tracks_all = torch.cat(batch_dense_tracks_list, dim=1)  # (T_seg, N_subset, 49, 2)
                    _dense_vis_all = torch.cat(batch_dense_vis_list, dim=1)
                    for local_i, global_i in enumerate(_queries_idx.cpu().tolist()):
                        if local_i < _dense_tracks_all.shape[1]:
                            dense_tracks[0, q_frame_idx:q_frame_idx+valid_len, global_i, :, :] = _dense_tracks_all[:, local_i, :, :]
                            dense_vis[0, q_frame_idx:q_frame_idx+valid_len, global_i, :] = _dense_vis_all[:, local_i, :]

                del images_gpu, batch_traj_list, batch_vis_list
                if is_cuda:
                    torch.cuda.empty_cache()
                progress.advance(task)
                continue  # Skip the non-batched path below

            # ==================================================
            # [STEP 3] Loop Padding: 在这里凑点 (original path for small N)
            # ==================================================
            _queries_padded, _pad_mask = self._pad_queries_with_grid(
                _queries_real.unsqueeze(0),
                H=self.interp_shape[0],
                W=self.interp_shape[1],
                target_n=TARGET_N
            )

            # --- [计时开始] ---
            if is_cuda:
                torch.cuda.synchronize()
            start_time = time.time()

            # 按需处理视频
            raw_segment = video[:, q_frame_idx:, ...].clone()
            images_gpu = self._preprocess_frame_chunk(raw_segment, device)
            del raw_segment

            # ==================================================
            # [BF16 核心修改] 开启 Autocast
            # ==================================================
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
                res = self.model.video_forward(
                    {'images': images_gpu, 'queries': _queries_padded[..., 1:]},
                    use_aug=use_aug,
                    return_dense_flow=self.enable_dense_matching
                )

            # 输出转换：detach -> float32 -> cpu
            # res['mkpts1_f'] 的形状通常是 [N_padded, T_seg, 2]
            _traj_e = res['mkpts1_f'].detach().float().cpu()
            _vis_e = res['pred_visibles'].squeeze(-1).detach().cpu() # [N_padded, T_seg]

            # Extract raw visibility scores if available
            _vis_scores = None
            if 'visibility_scores' in res:
                _vis_scores = res['visibility_scores'].squeeze(-1).detach().float().cpu()  # [N_padded, T_seg]

            # --- [计时结束 & FPS计算] ---
            if is_cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            valid_len = _traj_e.shape[1]
            if valid_len > 0:
                current_fps = valid_len / (end_time - start_time)
                progress.update(
                    task,
                    description=f"{desc_base} | FPS: {current_fps:.2f} | Valid Pts: {len(_queries_real)}",
                )

            # ==================================================
            # [STEP 4] Filtering: 过滤 padding 点
            # ==================================================
            # _pad_mask[0] shape: [TARGET_N]
            real_indices_mask = _pad_mask[0].cpu()
            N_real = real_indices_mask.sum().item()

            # [FIXED] 修正索引维度：在第0维 (N维度) 进行筛选
            _traj_real = _traj_e[real_indices_mask, :, :]   # [N_subset, T_seg, 2]
            _vis_real = _vis_e[real_indices_mask, :]        # [N_subset, T_seg]

            # Also filter visibility scores if available
            _vis_scores_real = None
            if _vis_scores is not None:
                _vis_scores_real = _vis_scores[real_indices_mask, :]  # [N_subset, T_seg]

            # 填入全局 buffer
            # 这里的 transpose(0, 1) 会把 [N, T] 变成 [T, N]，正好匹配 tracks[0, T_slice, N_idx]
            tracks[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _traj_real.transpose(0,1)
            occlusion_pred[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _vis_real.transpose(0,1)

            # Store raw visibility scores
            if _vis_scores_real is not None:
                visibility_scores_buffer[0, q_frame_idx : q_frame_idx + valid_len, _queries_idx.cpu()] = _vis_scores_real.transpose(0,1)

            # ==================================================
            # [Dense Matching] Extract patch-level dense predictions
            # ==================================================
            if self.enable_dense_matching:
                # Extract dense flow from updated_pos_nlvl_flow
                # Expected shape after fix: (N_padded, nlvl=3, T_seg, 49, 2)
                dense_flow_raw = res.get('updated_pos_nlvl_flow', None)
                if dense_flow_raw is not None:
                    CONSOLE.print(f"[Debug] updated_pos_nlvl_flow shape: {dense_flow_raw.shape}")
                    # Sample from level 2 (finest), first frame, first few patch points
                    sample = dense_flow_raw[0, self.dense_level, 0, :3, :]
                    CONSOLE.print(f"[Debug] Sample (query0, level{self.dense_level}, frame0, patches0-2): {sample}")
                else:
                    CONSOLE.print("[Debug] updated_pos_nlvl_flow is None!")

                _dense_tracks, _dense_vis = self._extract_dense_flow_from_result(res, N_real)

                if _dense_tracks is not None:
                    CONSOLE.print(f"[Debug] _dense_tracks shape: {_dense_tracks.shape}")
                    # _dense_tracks: (1, T_seg, N_real, 49, 2) in interp_shape space
                    # _dense_vis: (1, T_seg, N_real, 49)
                    _dense_tracks_cpu = _dense_tracks[0].float().cpu()  # (T_seg, N_real, 49, 2)
                    _dense_vis_cpu = _dense_vis[0].cpu()  # (T_seg, N_real, 49)

                    # Fill into global dense buffer (scaling happens later)
                    # dense_tracks shape: (B, T, N_orig, WW, 2)
                    for local_i, global_i in enumerate(_queries_idx.cpu().tolist()):
                        if local_i < _dense_tracks_cpu.shape[1]:
                            dense_tracks[0, q_frame_idx:q_frame_idx+valid_len, global_i, :, :] = _dense_tracks_cpu[:, local_i, :, :]
                            dense_vis[0, q_frame_idx:q_frame_idx+valid_len, global_i, :] = _dense_vis_cpu[:, local_i, :]

            del images_gpu, res, _traj_e, _vis_e, _queries_padded, _pad_mask
            if is_cuda:
                torch.cuda.empty_cache()
            progress.advance(task)

        progress.stop()

        traj_e = tracks
        vis_e = occlusion_pred

        # Store raw visibility scores for external access
        self.last_visibility_scores = visibility_scores_buffer

        # ==================================================
        # [STEP 5] 坐标反归一化
        # ==================================================
        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        if self.enable_dense_matching:
            # Scale dense tracks from interp_shape to original image space
            dense_tracks[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            dense_tracks[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)
            return traj_e, vis_e, dense_tracks, dense_vis

        return traj_e, vis_e

    @torch.no_grad()
    def _video_forward_multibatch(self, video: Tensor, queries: Tensor, use_aug: bool = False):
        """
        Multi-batch video forward pass. Processes each batch independently and concatenates results.

        This method handles B > 1 by:
        1. Processing each batch separately (to avoid memory manager conflicts)
        2. Concatenating results along the batch dimension

        Args:
            video: [B, T, C, H, W] video tensor
            queries: [B, N, 3] queries tensor with (t, x, y) format
            use_aug: Whether to use augmentation

        Returns:
            traj_e: [B, T, N, 2] trajectory predictions
            vis_e: [B, T, N] visibility predictions
            (optional) dense_tracks, dense_vis if dense matching enabled
        """
        B, T, C, H, W = video.shape
        B_q, N, D = queries.shape

        assert B == B_q, f"Video batch size ({B}) must match queries batch size ({B_q})"

        all_traj = []
        all_vis = []
        all_dense_tracks = [] if self.enable_dense_matching else None
        all_dense_vis = [] if self.enable_dense_matching else None

        for b_idx in range(B):
            # Extract single batch
            video_b = video[b_idx:b_idx+1]  # [1, T, C, H, W]
            queries_b = queries[b_idx:b_idx+1]  # [1, N, 3]

            # Reset memory for each batch to ensure independence
            self.model.mem_manager.reset_all_memory()

            # Process single batch (recursive call will handle B=1 case)
            result = self.video_forward(video_b, queries_b, use_aug)

            if self.enable_dense_matching:
                traj_b, vis_b, dense_tracks_b, dense_vis_b = result
                all_traj.append(traj_b)
                all_vis.append(vis_b)
                all_dense_tracks.append(dense_tracks_b)
                all_dense_vis.append(dense_vis_b)
            else:
                traj_b, vis_b = result
                all_traj.append(traj_b)
                all_vis.append(vis_b)

        # Concatenate results along batch dimension
        traj_e = torch.cat(all_traj, dim=0)  # [B, T, N, 2]
        vis_e = torch.cat(all_vis, dim=0)    # [B, T, N]

        if self.enable_dense_matching:
            dense_tracks = torch.cat(all_dense_tracks, dim=0)
            dense_vis = torch.cat(all_dense_vis, dim=0)
            return traj_e, vis_e, dense_tracks, dense_vis

        return traj_e, vis_e


    def _pad_queries_with_grid(self, queries: Tensor, H: int, W: int, target_n: int = 200) -> Tuple[Tensor, Tensor]:
        """
        如果 queries 数量不足 target_n，则使用随机网格点填充。
        
        Args:
            queries (Tensor): [B, N, 3] (t, x, y)
            H (int): 视频高度
            W (int): 视频宽度
            target_n (int): 目标点数
            
        Returns:
            padded_queries (Tensor): [B, target_n, 3]
            valid_mask (Tensor): [B, target_n] 布尔掩码，True表示原始真实点，False表示填充点
        """
        B, N, D = queries.shape
        device = queries.device
        
        # 如果点数已经足够，直接截断或返回 (根据需求，这里选择返回原样或截断)
        if N >= target_n:
            # 如果你希望严格等于200，可以使用 queries[:, :target_n, :]
            # 这里为了安全起见，如果不缺就不补，构建全True掩码
            valid_mask = torch.ones((B, N), dtype=torch.bool, device=device)
            return queries, valid_mask
            
        num_to_pad = target_n - N
        
        # 1. 确定填充点的时间帧索引 (t)
        # 策略：通常将填充点放在第一帧，或者复制第一个查询点的时间
        # 这里我们取每个 batch 中第一个点的 t
        ref_t = queries[:, 0:1, 0] # [B, 1]
        
        # 2. 生成均匀网格 (Grid Generation)
        # 为了保证有足够的点可供采样，我们生成一个足够密集的网格
        # 例如生成 20x20 = 400 个格点，肯定够补
        grid_size = int(np.ceil(np.sqrt(target_n * 2))) 
        xs = torch.linspace(0, W - 1, steps=grid_size, device=device)
        ys = torch.linspace(0, H - 1, steps=grid_size, device=device)
        
        # 生成网格坐标
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2) # [1, G*G, 2]
        
        # 3. 随机采样 (Random Sampling without replacement)
        # 我们从网格中随机选 num_to_pad 个点
        # 为了让每个 Batch 采样的点不一样，我们循环处理或者使用高维索引技巧
        # 这里为了简单高效，假设所有 Batch 使用相同的随机模式，或者简单重复
        
        # 生成随机索引
        perm = torch.randperm(grid_points.shape[1], device=device)[:num_to_pad]
        sampled_grid = grid_points[:, perm, :] # [1, num_to_pad, 2] (x, y)
        
        # 扩展到 Batch 维度
        sampled_grid = sampled_grid.expand(B, -1, -1) # [B, num_to_pad, 2]
        
        # 4. 拼接时间 t
        ref_t_expanded = ref_t.unsqueeze(1).expand(-1, num_to_pad, 1) # [B, num_to_pad, 1]
        padding_queries = torch.cat([ref_t_expanded, sampled_grid], dim=-1) # [B, num_to_pad, 3]
        
        # 5. 合并结果
        padded_queries = torch.cat([queries, padding_queries], dim=1) # [B, target_n, 3]
        
        # 6. 生成掩码 (方便后续区分哪些是真实点，哪些是凑数的)
        mask_real = torch.ones((B, N), dtype=torch.bool, device=device)
        mask_fake = torch.zeros((B, num_to_pad), dtype=torch.bool, device=device)
        valid_mask = torch.cat([mask_real, mask_fake], dim=1)
        
        return padded_queries, valid_mask

    def load_state_dict_from_pl_checkpoint(self, model, pl_checkpoint_path: str) -> None:
        if pl_checkpoint_path is None:
            raise ValueError("pl_checkpoint_path must not be None")
        # Always load Lightning checkpoints onto CPU first so CPU-only machines can
        # load checkpoints trained on GPU. The caller is responsible for moving the
        # model to the desired device afterwards.
        ckpt = safe_torch_load(pl_checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]
        state_dict = {k:v for k,v in state_dict.items() if 'loss.' not in k}
        state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

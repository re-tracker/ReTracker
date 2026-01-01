"""
Multi-view tracker implementation.

This module provides a flexible multi-view tracking system that:
1. Detects keypoints on all views (not just reference view)
2. Matches points bidirectionally across views
3. Tracks points temporally in each view
4. Outputs synchronized visualization

The design is modular and extensible for future enhancements.
"""

import os
# Set environment variable before importing cv2 to handle headless environments
if os.environ.get('DISPLAY') is None:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import imageio

from rich.console import Console

from .config import MultiViewConfig, ViewConfig

CONSOLE = Console()


@dataclass
class PointTrack:
    """Represents a tracked point across views and time."""
    point_id: int
    color: Tuple[int, int, int]  # RGB color for visualization

    # Per-view tracking data: view_id -> list of (frame_idx, x, y, visible)
    view_tracks: Dict[str, List[Tuple[int, float, float, bool]]] = None

    def __post_init__(self):
        if self.view_tracks is None:
            self.view_tracks = {}


class ViewTracker:
    """
    Manages tracking for a single view.

    This class handles:
    - Frame loading from image directory
    - Streaming-based temporal tracking
    - Track history management

    Tracker is created lazily to save GPU memory when processing views sequentially.
    """

    def __init__(
        self,
        view_config: ViewConfig,
        config: MultiViewConfig
    ):
        self.view_config = view_config
        self.config = config

        # Tracker is created lazily to save memory
        self.tracker = None
        self.device = config.model.device

        # Load frame list
        self.frame_paths = self._load_frame_paths()
        self.num_frames = len(self.frame_paths)

        # Tracking state
        self.is_initialized = False
        self.queries = None  # [1, N, 3] tensor
        self.current_frame_idx = 0

        # Track history for visualization
        self.tracks_history: List[np.ndarray] = []  # List of [N, 2] arrays
        self.visibility_history: List[np.ndarray] = []  # List of [N] arrays

        CONSOLE.print(f"[cyan]View {view_config.view_id}: {self.num_frames} frames")

    def _build_tracker(self):
        """Build tracker instance for this view."""
        from retracker.apps.runtime.tracker import Tracker
        from retracker.apps.config.base_config import ModelConfig as BaseModelConfig

        model_config = BaseModelConfig(
            ckpt_path=self.config.model.ckpt_path,
            device=self.config.model.device,
            interp_shape=self.config.model.interp_shape,
            compile=self.config.model.compile,
            use_amp=self.config.model.use_amp,
            dtype=self.config.model.dtype,
            # IMPORTANT: Enable highres inference for correct matching
            enable_highres_inference=self.config.model.enable_highres_inference,
            coarse_resolution=self.config.model.coarse_resolution,
            # Multi-GPU settings
            devices=self.config.model.devices,
            query_batch_size=self.config.model.query_batch_size,
            tracking_visibility_threshold=self.config.model.tracking_visibility_threshold,
            matching_visibility_threshold=self.config.model.matching_visibility_threshold,
        )
        tracker = Tracker(model_config)
        tracker.set_task_mode('tracking')
        return tracker

    def build_tracker(self):
        """Build tracker if not already built."""
        if self.tracker is None:
            CONSOLE.print(f"[cyan]View {self.view_config.view_id}: Building tracker...")
            self.tracker = self._build_tracker()

    def release_tracker(self):
        """Release tracker to free GPU memory (only if we own it)."""
        if self.tracker is not None and getattr(self, '_owns_tracker', True):
            CONSOLE.print(f"[cyan]View {self.view_config.view_id}: Releasing tracker...")
            # Clear model from GPU
            del self.tracker
            self.tracker = None
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Just clear reference if using shared tracker
            self.tracker = None
        self._owns_tracker = True

    def _load_frame_paths(self) -> List[Path]:
        """Load sorted list of frame paths."""
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        paths = []
        for ext in extensions:
            paths.extend(self.view_config.images_dir.glob(f'*{ext}'))
            paths.extend(self.view_config.images_dir.glob(f'*{ext.upper()}'))
        return sorted(paths)

    def load_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load a frame by index and resize to interp_shape."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return None
        frame = cv2.imread(str(self.frame_paths[frame_idx]))
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # IMPORTANT: Resize to interp_shape (H, W) for correct coordinate handling
            target_h, target_w = self.config.model.interp_shape
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def get_frame_shape(self) -> Tuple[int, int, int]:
        """Get frame shape (H, W, C) from first frame."""
        frame = self.load_frame(0)
        return frame.shape if frame is not None else (0, 0, 0)

    def initialize_tracking(self, points: np.ndarray, shared_tracker=None):
        """
        Initialize tracking with given points.

        Args:
            points: [N, 2] array of (x, y) coordinates on first frame
            shared_tracker: Optional shared Tracker instance to reuse
        """
        N = len(points)
        if N == 0:
            CONSOLE.print(f"[yellow]View {self.view_config.view_id}: No points to track")
            return

        # Use shared tracker if provided, otherwise build own tracker
        if shared_tracker is not None:
            self.tracker = shared_tracker
            self._owns_tracker = False
        else:
            self.build_tracker()
            self._owns_tracker = True

        # Create queries tensor [1, N, 3] with (t=0, x, y)
        self.queries = torch.zeros((1, N, 3), dtype=torch.float32, device=self.device)
        self.queries[0, :, 0] = 0  # t = 0 (first frame)
        self.queries[0, :, 1] = torch.from_numpy(points[:, 0]).float()  # x
        self.queries[0, :, 2] = torch.from_numpy(points[:, 1]).float()  # y

        # Initialize streaming mode
        H, W, _ = self.get_frame_shape()
        self.tracker.streaming_reset()
        self.tracker.streaming_init(
            video_shape=(3, H, W),
            initial_queries=self.queries[0]
        )

        self.is_initialized = True
        self.current_frame_idx = 0
        self.tracks_history.clear()
        self.visibility_history.clear()

        CONSOLE.print(f"[green]View {self.view_config.view_id}: Initialized tracking with {N} points")

    def process_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame and return tracking results.

        Args:
            frame_idx: Frame index to process

        Returns:
            tracks: [N, 2] array of point positions
            visibility: [N] array of visibility flags
        """
        if not self.is_initialized:
            raise RuntimeError("Tracking not initialized. Call initialize_tracking first.")

        frame = self.load_frame(frame_idx)
        if frame is None:
            raise RuntimeError(f"Failed to load frame {frame_idx}")

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).to(self.device)

        # Process frame
        result = self.tracker.streaming_process_frame(frame_tensor)
        tracks = result['tracks'].cpu().numpy()  # [N, 2]
        visibility = result['visibility'].cpu().numpy()  # [N]

        # Store history
        self.tracks_history.append(tracks.copy())
        self.visibility_history.append(visibility.copy())

        self.current_frame_idx = frame_idx
        return tracks, visibility

    def reset(self):
        """Reset tracking state."""
        if self.tracker is not None:
            self.tracker.streaming_reset()
        self.is_initialized = False
        self.queries = None
        self.current_frame_idx = 0
        self.tracks_history.clear()
        self.visibility_history.clear()


class CrossViewMatcher:
    """
    Handles cross-view point matching using the tracking model.

    This class matches points from a reference view to other views
    using the model's two-frame matching capability.
    """

    def __init__(self, config: MultiViewConfig):
        self.config = config
        self.tracker = self._build_tracker()
        self.device = config.model.device

    def _build_tracker(self):
        """Build tracker for matching."""
        from retracker.apps.runtime.tracker import Tracker
        from retracker.apps.config.base_config import ModelConfig as BaseModelConfig

        model_config = BaseModelConfig(
            ckpt_path=self.config.model.ckpt_path,
            device=self.config.model.device,
            interp_shape=self.config.model.interp_shape,
            compile=self.config.model.compile,
            use_amp=self.config.model.use_amp,
            dtype=self.config.model.dtype,
            # IMPORTANT: Enable highres inference for correct matching
            enable_highres_inference=self.config.model.enable_highres_inference,
            coarse_resolution=self.config.model.coarse_resolution,
            tracking_visibility_threshold=self.config.model.tracking_visibility_threshold,
            matching_visibility_threshold=self.config.model.matching_visibility_threshold,
        )
        tracker = Tracker(model_config)
        tracker.set_task_mode('matching')
        return tracker

    def release(self):
        """Release tracker to free GPU memory."""
        if self.tracker is not None:
            CONSOLE.print("[cyan]CrossViewMatcher: Releasing tracker...")
            del self.tracker
            self.tracker = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def match_views(
        self,
        ref_frame: np.ndarray,
        target_frame: np.ndarray,
        ref_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match points from reference frame to target frame.

        Uses the tracking model to find corresponding points.

        Args:
            ref_frame: Reference frame [H, W, 3] RGB
            target_frame: Target frame [H, W, 3] RGB
            ref_points: [N, 2] array of (x, y) points on reference frame

        Returns:
            matched_points: [N, 2] array of matched positions on target frame
            confidence: [N] array of match confidence scores
        """
        if len(ref_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])

        N = len(ref_points)
        H, W = ref_frame.shape[:2]

        # Create video tensor [1, 2, C, H, W] with ref and target frames
        # Keep on CPU - the engine will move to GPU as needed
        video = np.stack([ref_frame, target_frame], axis=0)  # [2, H, W, C]
        video = torch.from_numpy(video).float().permute(0, 3, 1, 2)  # [2, C, H, W]
        video = video.unsqueeze(0)  # [1, 2, C, H, W] - keep on CPU

        # Create queries [1, N, 3] with (t=0, x, y)
        # Queries can be on CPU, engine will handle device transfer
        queries = torch.zeros((1, N, 3), dtype=torch.float32)
        queries[0, :, 0] = 0  # All points start at frame 0
        queries[0, :, 1] = torch.from_numpy(ref_points[:, 0]).float()
        queries[0, :, 2] = torch.from_numpy(ref_points[:, 1]).float()

        # Run tracking to get positions at frame 1
        with torch.no_grad():
            trajectories, visibility = self.tracker.track(video, queries)
            # trajectories: [1, 2, N, 2]
            # visibility: [1, 2, N] (boolean)

            # Get raw visibility scores (float)
            visibility_scores = self.tracker.get_visibility_scores()

        # Get positions at target frame (frame 1)
        matched_points = trajectories[0, 1].cpu().numpy()  # [N, 2]

        # Use raw visibility scores if available, otherwise fall back to boolean
        if visibility_scores is not None:
            confidence = visibility_scores[0, 1].cpu().numpy()  # [N] raw float scores
        else:
            confidence = visibility[0, 1].cpu().numpy().astype(float)  # [N] boolean as float

        return matched_points, confidence


class MultiViewTracker:
    """
    Multi-view tracker that tracks corresponding points across multiple camera views.

    Features:
    - Bidirectional SIFT keypoint detection on all views
    - Cross-view matching using the tracking model
    - Temporal tracking in each view (with independent streaming state)
    - Synchronized multi-view visualization

    Usage:
        config = MultiViewConfig(
            data_root=Path("data/multiview_tracker/0172_05/images"),
            view_ids=["19", "25", "28"],
            reference_view="25",
            num_points=100,
        )
        tracker = MultiViewTracker(config)
        tracker.run()
    """

    def __init__(self, config: MultiViewConfig):
        self.config = config
        config.validate()

        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.debug_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.print(f"[cyan]Output directory: {config.output_dir}")

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=config.sift.n_features,
            nOctaveLayers=config.sift.n_octave_layers,
            contrastThreshold=config.sift.contrast_threshold,
            edgeThreshold=config.sift.edge_threshold,
            sigma=config.sift.sigma,
        )

        # Initialize cross-view matcher (separate instance for matching)
        self.matcher = CrossViewMatcher(config)

        # Initialize per-view trackers (each has its own tracker instance)
        self.view_trackers: Dict[str, ViewTracker] = {}
        self._initialize_views()

        # Point tracks (shared across views)
        self.point_tracks: List[PointTrack] = []

        # Video writer
        self.video_writer = None

    def _initialize_views(self):
        """Initialize view trackers for all views."""
        view_configs = self.config.get_view_configs()
        for vc in view_configs:
            CONSOLE.print(f"[cyan]Initializing view tracker for view {vc.view_id}...")
            self.view_trackers[vc.view_id] = ViewTracker(vc, self.config)

    def _get_mask_dir(self) -> Path:
        """Get mask directory corresponding to data_root."""
        # data_root: .../images -> mask_dir: .../fmasks
        return self.config.data_root.parent / "fmasks"

    def _load_mask(self, image_path: Path, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Load mask corresponding to image path.

        Args:
            image_path: Path to image file
            target_shape: (H, W) to resize mask to

        Returns:
            mask: [H, W] binary mask or None if not found
        """
        mask_dir = self._get_mask_dir()
        # images/19/000000.webp -> fmasks/19/000000.png
        view_id = image_path.parent.name
        mask_path = mask_dir / view_id / (image_path.stem + ".png")

        if not mask_path.exists():
            CONSOLE.print(f"[yellow]Mask not found: {mask_path}")
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Resize to target shape
        target_h, target_w = target_shape
        if mask.shape[0] != target_h or mask.shape[1] != target_w:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return mask

    def _filter_points_by_mask(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filter points to keep only those within the mask (mask > 0)."""
        if mask is None or len(points) == 0:
            return points

        H, W = mask.shape
        valid_mask = np.zeros(len(points), dtype=bool)

        for i, pt in enumerate(points):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                valid_mask[i] = mask[y, x] > 0

        return points[valid_mask]

    def _grid_sample_in_mask(self, mask: np.ndarray, num_points: int, existing_points: np.ndarray = None) -> np.ndarray:
        """
        Uniformly sample grid points within the mask region.

        The algorithm:
        1. Find the bounding box of the mask
        2. Estimate grid step based on mask area and target num_points
        3. Generate uniform grid within bounding box
        4. Filter to keep only points inside the mask
        5. Iteratively adjust grid step if needed to get closer to target count

        Args:
            mask: [H, W] binary mask
            num_points: target number of points to sample
            existing_points: [N, 2] existing points to avoid duplicates

        Returns:
            [M, 2] array of sampled points (x, y), uniformly distributed in mask
        """
        if mask is None:
            return np.array([]).reshape(0, 2)

        H, W = mask.shape

        # Find mask bounding box for efficient sampling
        valid_y, valid_x = np.where(mask > 0)
        if len(valid_x) == 0:
            return np.array([]).reshape(0, 2)

        x_min, x_max = valid_x.min(), valid_x.max()
        y_min, y_max = valid_y.min(), valid_y.max()
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        # Estimate mask area and density
        mask_area = len(valid_x)
        mask_density = mask_area / (bbox_w * bbox_h)  # How much of bbox is filled

        # Calculate initial grid step
        # For uniform grid, num_points ≈ mask_area / (step^2)
        # So step ≈ sqrt(mask_area / num_points)
        target_points = num_points * 1.5  # Overshoot slightly to account for filtering
        grid_step = max(1, int(np.sqrt(mask_area / target_points)))

        # Generate grid points within bounding box
        def generate_grid(step):
            points = []
            # Start from half-step offset for centering
            y_start = y_min + step // 2
            x_start = x_min + step // 2
            for y in range(y_start, y_max + 1, step):
                for x in range(x_start, x_max + 1, step):
                    if 0 <= y < H and 0 <= x < W and mask[y, x] > 0:
                        points.append([x, y])
            return np.array(points, dtype=np.float32) if points else np.array([]).reshape(0, 2)

        # Try to get approximately num_points by adjusting step
        grid_points = generate_grid(grid_step)

        # If too few points, decrease step; if too many, increase step
        max_iterations = 5
        for _ in range(max_iterations):
            n_points = len(grid_points)
            if n_points == 0:
                grid_step = max(1, grid_step - 1)
                grid_points = generate_grid(grid_step)
                continue

            # Check if we're close enough to target
            ratio = n_points / num_points
            if 0.8 <= ratio <= 1.5:
                break

            if ratio < 0.8:
                # Too few points, decrease step
                new_step = max(1, int(grid_step * np.sqrt(n_points / num_points)))
                if new_step >= grid_step:
                    new_step = grid_step - 1
                grid_step = max(1, new_step)
            else:
                # Too many points, increase step
                new_step = int(grid_step * np.sqrt(n_points / num_points))
                if new_step <= grid_step:
                    new_step = grid_step + 1
                grid_step = new_step

            grid_points = generate_grid(grid_step)

        # Remove points too close to existing points
        if existing_points is not None and len(existing_points) > 0 and len(grid_points) > 0:
            min_dist = grid_step * 0.8  # Minimum distance based on grid step
            min_dist = max(min_dist, 5)  # At least 5 pixels
            filtered = []
            for gp in grid_points:
                dists = np.linalg.norm(existing_points - gp, axis=1)
                if dists.min() > min_dist:
                    filtered.append(gp)
            grid_points = np.array(filtered, dtype=np.float32) if filtered else np.array([]).reshape(0, 2)

        return grid_points

    def _detect_sift_points(self, frame: np.ndarray, max_points: int = None, mask: np.ndarray = None) -> np.ndarray:
        """
        Detect keypoints on a frame. By default uses uniform grid sampling within mask.
        If config.sift.use_sift=True, uses SIFT keypoints instead.

        Args:
            frame: [H, W, 3] RGB frame
            max_points: Maximum number of points to return (default: config.num_points)
            mask: [H, W] binary mask - points must be where mask > 0

        Returns:
            points: [N, 2] array of (x, y) coordinates
        """
        if max_points is None:
            max_points = self.config.num_points

        # Check if we should use SIFT or just grid points
        use_sift = self.config.sift.use_sift

        if not use_sift:
            # Use only uniform grid sampling within mask
            if mask is not None:
                grid_points = self._grid_sample_in_mask(mask, max_points)
                CONSOLE.print(f"    Grid: {len(grid_points)} points uniformly sampled in mask")
                return grid_points.astype(np.float32) if len(grid_points) > 0 else np.array([]).reshape(0, 2)
            else:
                # No mask - generate uniform grid on entire image
                H, W = frame.shape[:2]
                grid_step = max(1, int(np.sqrt(H * W / max_points)))
                grid_points = []
                for y in range(grid_step // 2, H, grid_step):
                    for x in range(grid_step // 2, W, grid_step):
                        grid_points.append([x, y])
                grid_points = np.array(grid_points[:max_points], dtype=np.float32) if grid_points else np.array([]).reshape(0, 2)
                CONSOLE.print(f"    Grid: {len(grid_points)} points uniformly sampled (no mask)")
                return grid_points

        # SIFT mode: detect SIFT keypoints
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        keypoints = self.sift.detect(gray, None)

        # Sort by response (strength)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

        # Extract all SIFT points
        all_sift_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        if len(all_sift_points) == 0:
            all_sift_points = np.array([]).reshape(0, 2)

        n_all_sift = len(all_sift_points)

        # Filter by mask
        if mask is not None and len(all_sift_points) > 0:
            sift_points = self._filter_points_by_mask(all_sift_points, mask)
        else:
            sift_points = all_sift_points

        n_sift_in_mask = len(sift_points)

        # Take top max_points
        sift_points = sift_points[:max_points]

        CONSOLE.print(f"    SIFT: {n_all_sift} total, {n_sift_in_mask} in mask, using {len(sift_points)}")

        # If not enough points, supplement with grid sampling
        n_grid_added = 0
        if len(sift_points) < max_points and mask is not None:
            need_more = max_points - len(sift_points)
            grid_points = self._grid_sample_in_mask(mask, need_more * 2, existing_points=sift_points)

            if len(grid_points) > 0:
                grid_points = grid_points[:need_more]
                n_grid_added = len(grid_points)
                sift_points = np.vstack([sift_points, grid_points]) if len(sift_points) > 0 else grid_points
                CONSOLE.print(f"    Added {n_grid_added} grid points, total: {len(sift_points)}")

        return sift_points.astype(np.float32) if len(sift_points) > 0 else np.array([]).reshape(0, 2)

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate N distinct colors for visualization."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)  # Spread across hue spectrum
            color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
            colors.append(tuple(map(int, color[0, 0])))
        return colors

    def _bidirectional_matching(self) -> Dict[str, np.ndarray]:
        """
        Perform bidirectional matching across all views.

        1. Detect SIFT on all views (filtered by mask)
        2. Match SIFT from each non-reference view to reference view
        3. Match SIFT from reference view to all other views
        4. Combine all matched points, ensuring each point has at least one valid match

        Returns:
            Dict mapping view_id -> [N, 2] query points for each view
        """
        ref_view_id = self.config.reference_view
        other_view_ids = [vid for vid in self.config.view_ids if vid != ref_view_id]

        # Load first frames and masks
        frames = {}
        masks = {}
        CONSOLE.print("\n[cyan]Loading frames and masks...")
        for view_id, view_tracker in self.view_trackers.items():
            frames[view_id] = view_tracker.load_frame(0)
            # Load corresponding mask
            frame_path = view_tracker.frame_paths[0]
            target_shape = self.config.model.interp_shape  # (H, W)
            masks[view_id] = self._load_mask(frame_path, target_shape)
            if masks[view_id] is not None:
                mask_coverage = (masks[view_id] > 0).sum() / masks[view_id].size * 100
                CONSOLE.print(f"  View {view_id}: Mask loaded, coverage: {mask_coverage:.1f}%")
            else:
                CONSOLE.print(f"  [red]View {view_id}: No mask found! Grid sampling disabled.")

        ref_frame = frames[ref_view_id]

        # Step 1: Detect SIFT on all views (with mask filtering)
        # Each view gets num_points (not divided), matching debug_matching.py behavior
        CONSOLE.print("\n[cyan]Step 1: Detecting SIFT keypoints on all views (with mask)...")
        sift_points = {}

        for view_id in self.config.view_ids:
            sift_points[view_id] = self._detect_sift_points(
                frames[view_id],
                max_points=self.config.num_points,
                mask=masks[view_id]
            )
            CONSOLE.print(f"  View {view_id}: {len(sift_points[view_id])} points (SIFT + grid in mask)")

        # Step 2: Match SIFT from other views to reference view
        CONSOLE.print("\n[cyan]Step 2: Matching SIFT from other views to reference view...")
        other_to_ref_matches = {}  # view_id -> (matched_points_on_ref, confidence, original_points)

        for view_id in other_view_ids:
            other_frame = frames[view_id]
            other_points = sift_points[view_id]

            if len(other_points) > 0:
                matched_on_ref, confidence = self.matcher.match_views(
                    other_frame, ref_frame, other_points
                )
                other_to_ref_matches[view_id] = (matched_on_ref, confidence, other_points)
                high_conf = (confidence > self.config.matching.confidence_threshold).sum()
                CONSOLE.print(f"  View {view_id} -> View {ref_view_id}: {high_conf}/{len(other_points)} high confidence matches")

        # Step 3: Match SIFT from reference view to all other views
        CONSOLE.print("\n[cyan]Step 3: Matching reference SIFT to other views...")
        ref_to_other_matches = {}  # view_id -> (matched_points, confidence)

        ref_sift = sift_points[ref_view_id]
        for view_id in other_view_ids:
            other_frame = frames[view_id]

            if len(ref_sift) > 0:
                matched_points, confidence = self.matcher.match_views(
                    ref_frame, other_frame, ref_sift
                )
                ref_to_other_matches[view_id] = (matched_points, confidence)
                high_conf = (confidence > self.config.matching.confidence_threshold).sum()
                CONSOLE.print(f"  View {ref_view_id} -> View {view_id}: {high_conf}/{len(ref_sift)} high confidence matches")

        # Step 4: Build combined query points for reference view
        CONSOLE.print("\n[cyan]Step 4: Building combined query points...")
        threshold = self.config.matching.confidence_threshold

        # Filter reference SIFT points: only keep those with valid matches (conf > threshold) in at least one other view
        ref_sift = sift_points[ref_view_id]
        ref_sift_valid_mask = np.zeros(len(ref_sift), dtype=bool)
        for view_id in other_view_ids:
            if view_id in ref_to_other_matches:
                _, confidence = ref_to_other_matches[view_id]
                ref_sift_valid_mask |= (confidence > threshold)

        ref_sift_valid = ref_sift[ref_sift_valid_mask]
        CONSOLE.print(f"  Reference SIFT: {len(ref_sift_valid)}/{len(ref_sift)} have valid matches (conf > {threshold})")

        # Start with valid reference SIFT points
        ref_queries = list(ref_sift_valid)
        ref_point_sources = ['ref_sift'] * len(ref_sift_valid)

        # Add matched points from other views (only high confidence)
        for view_id in other_view_ids:
            if view_id in other_to_ref_matches:
                matched_on_ref, confidence, original_points = other_to_ref_matches[view_id]
                # Only add high confidence matches
                valid_mask = confidence > threshold
                valid_points = matched_on_ref[valid_mask]
                for pt in valid_points:
                    ref_queries.append(pt)
                    ref_point_sources.append(f'from_{view_id}')

        ref_queries = np.array(ref_queries, dtype=np.float32) if len(ref_queries) > 0 else np.array([]).reshape(0, 2)
        CONSOLE.print(f"  Reference view {ref_view_id}: {len(ref_queries)} total query points")
        CONSOLE.print(f"    - From own SIFT (valid): {len(ref_sift_valid)}")
        for view_id in other_view_ids:
            count = ref_point_sources.count(f'from_{view_id}')
            CONSOLE.print(f"    - From view {view_id} matches: {count}")

        # Step 5: Match all reference queries to other views
        CONSOLE.print("\n[cyan]Step 5: Matching all queries to other views...")
        query_points = {ref_view_id: ref_queries}
        query_confidence = {ref_view_id: np.ones(len(ref_queries))}

        for view_id in other_view_ids:
            other_frame = frames[view_id]
            matched_points, confidence = self.matcher.match_views(
                ref_frame, other_frame, ref_queries
            )
            query_points[view_id] = matched_points
            query_confidence[view_id] = confidence
            high_conf = (confidence > self.config.matching.confidence_threshold).sum()
            CONSOLE.print(f"  View {ref_view_id} -> View {view_id}: {high_conf}/{len(ref_queries)} matches")

        # Step 6: Filter points based on visibility/confidence
        CONSOLE.print("\n[cyan]Step 6: Filtering points by visibility...")
        threshold = self.config.matching.confidence_threshold
        require_all = self.config.matching.require_visible_in_all_views
        min_visible_views = self.config.matching.min_visible_views

        if require_all:
            # Point must be visible (high confidence) in ALL views
            CONSOLE.print(f"  Mode: require_visible_in_all_views=True (threshold={threshold})")
            valid_mask = np.ones(len(ref_queries), dtype=bool)
            for view_id in other_view_ids:
                view_visible = query_confidence[view_id] > threshold
                valid_mask &= view_visible
                n_visible = view_visible.sum()
                CONSOLE.print(f"    View {view_id}: {n_visible}/{len(ref_queries)} visible")
        else:
            # Point must be visible in at least min_visible_views (including reference)
            # Count how many views each point is visible in
            CONSOLE.print(f"  Mode: min_visible_views={min_visible_views} (threshold={threshold})")
            visibility_count = np.ones(len(ref_queries), dtype=int)  # Reference view counts as 1
            for view_id in other_view_ids:
                view_visible = query_confidence[view_id] > threshold
                visibility_count += view_visible.astype(int)
                n_visible = view_visible.sum()
                CONSOLE.print(f"    View {view_id}: {n_visible}/{len(ref_queries)} visible")

            valid_mask = visibility_count >= min_visible_views
            CONSOLE.print(f"  Visibility distribution:")
            for n_views in range(1, len(self.config.view_ids) + 1):
                count = (visibility_count == n_views).sum()
                if count > 0:
                    CONSOLE.print(f"    {n_views} views: {count} points")

        n_valid = valid_mask.sum()
        filter_mode = 'all views' if require_all else f'at least {min_visible_views} view(s)'
        CONSOLE.print(f"  [green]Final: Keeping {n_valid}/{len(ref_queries)} points visible in {filter_mode}")

        # Apply filter
        filtered_query_points = {}
        filtered_confidence = {}
        for view_id in self.config.view_ids:
            filtered_query_points[view_id] = query_points[view_id][valid_mask]
            filtered_confidence[view_id] = query_confidence[view_id][valid_mask]

        # Step 7: Limit total points to num_points to prevent OOM
        n_filtered = len(filtered_query_points[ref_view_id])
        max_points = self.config.num_points
        if n_filtered > max_points:
            CONSOLE.print(f"\n[yellow]Step 7: Limiting points from {n_filtered} to {max_points} to prevent OOM...")
            # Sort by confidence (average across other views) and keep top max_points
            avg_confidence = np.zeros(n_filtered)
            for view_id in other_view_ids:
                avg_confidence += filtered_confidence[view_id]
            avg_confidence /= len(other_view_ids)

            # Get indices of top max_points by confidence
            top_indices = np.argsort(avg_confidence)[::-1][:max_points]
            top_indices = np.sort(top_indices)  # Keep original order

            for view_id in self.config.view_ids:
                filtered_query_points[view_id] = filtered_query_points[view_id][top_indices]
                filtered_confidence[view_id] = filtered_confidence[view_id][top_indices]

            CONSOLE.print(f"  [green]Kept top {max_points} points by confidence")

        # Save debug visualization
        self._save_bidirectional_matching_debug(
            frames, sift_points, filtered_query_points, filtered_confidence, valid_mask
        )

        return filtered_query_points, filtered_confidence

    def _save_bidirectional_matching_debug(
        self,
        frames: Dict[str, np.ndarray],
        sift_points: Dict[str, np.ndarray],
        query_points: Dict[str, np.ndarray],
        query_confidence: Dict[str, np.ndarray],
        valid_mask: np.ndarray
    ):
        """Save debug visualization of bidirectional matching."""
        ref_view_id = self.config.reference_view
        ref_frame = frames[ref_view_id]
        n_points = len(query_points[ref_view_id])
        colors = self._generate_colors(n_points)
        threshold = self.config.matching.confidence_threshold
        require_all = self.config.matching.require_visible_in_all_views

        CONSOLE.print("\n[cyan]Saving bidirectional matching debug visualizations...")

        # For each view pair, create side-by-side visualization
        for view_id in self.config.view_ids:
            if view_id == ref_view_id:
                continue

            target_frame = frames[view_id]
            ref_pts = query_points[ref_view_id]
            tgt_pts = query_points[view_id]
            tgt_conf = query_confidence[view_id]

            H, W = ref_frame.shape[:2]
            combined = np.zeros((H, W * 2, 3), dtype=np.uint8)
            combined[:, :W] = ref_frame
            combined[:, W:] = target_frame

            # Only draw points with confidence > threshold
            valid_indices = np.where(tgt_conf > threshold)[0]
            n_valid = len(valid_indices)

            # Generate colors for valid points only
            valid_colors = self._generate_colors(n_valid)

            for color_idx, i in enumerate(valid_indices):
                ref_pt = ref_pts[i]
                tgt_pt = tgt_pts[i]
                conf = tgt_conf[i]
                color = valid_colors[color_idx]

                ref_x, ref_y = int(ref_pt[0]), int(ref_pt[1])
                tgt_x, tgt_y = int(tgt_pt[0]) + W, int(tgt_pt[1])

                # Draw both points filled
                cv2.circle(combined, (ref_x, ref_y), 4, color, -1)
                cv2.circle(combined, (tgt_x, tgt_y), 4, color, -1)
                cv2.line(combined, (ref_x, ref_y), (tgt_x, tgt_y), color, 1)

                # Draw confidence text
                conf_text = f"{conf:.2f}"
                cv2.putText(combined, conf_text, (ref_x + 5, ref_y + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Labels
            mode_str = "ALL views" if require_all else "ANY view"
            cv2.putText(combined, f"View {ref_view_id} (Reference)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, f"View {view_id}", (W + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, f"Valid matches: {n_valid}/{n_points} (threshold={threshold})", (10, H - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f"Mode: require visible in {mode_str}", (10, H - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            output_path = self.config.debug_dir / f"matching_{ref_view_id}_to_{view_id}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            CONSOLE.print(f"[green]Saved: {output_path}")

    def _local_feature_matching_impl(
        self,
        frames: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        sift_points: Dict[str, np.ndarray],
        view_ids: List[str],
        threshold: float,
        is_ring: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Implement local feature matching where each view matches only with adjacent neighbors.

        Uses Union-Find to build tracks from pairwise matches without propagation.

        Args:
            frames: Dict of view_id -> frame image
            masks: Dict of view_id -> mask
            sift_points: Dict of view_id -> detected points
            view_ids: Ordered list of view IDs
            threshold: Confidence threshold for valid matches
            is_ring: Whether to close the loop (match last with first)

        Returns:
            query_points: Dict mapping view_id -> [N, 2] points
            query_confidence: Dict mapping view_id -> [N] confidence
        """
        n_views = len(view_ids)

        # Step 2: Build edges between adjacent views
        CONSOLE.print("\n[cyan]Step 2: Bidirectional matching between adjacent views...")

        # Create node IDs: (view_id, point_idx) -> global_id
        node_to_id = {}
        id_to_node = {}
        global_id = 0
        for view_id in view_ids:
            for pt_idx in range(len(sift_points[view_id])):
                node = (view_id, pt_idx)
                node_to_id[node] = global_id
                id_to_node[global_id] = node
                global_id += 1

        total_nodes = global_id

        # Union-Find data structure
        parent = list(range(total_nodes))
        rank = [0] * total_nodes

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # For each adjacent pair, perform bidirectional matching
        adjacent_pairs = []
        for i in range(n_views - 1):
            adjacent_pairs.append((i, i + 1))
        if is_ring and n_views >= 3:
            adjacent_pairs.append((n_views - 1, 0))  # Close the ring

        for i, j in adjacent_pairs:
            src_view = view_ids[i]
            dst_view = view_ids[j]
            CONSOLE.print(f"\n  Matching {src_view} <-> {dst_view}...")

            src_pts = sift_points[src_view]
            dst_pts = sift_points[dst_view]

            # Match src -> dst
            if len(src_pts) > 0 and len(dst_pts) > 0:
                matched_dst, conf_src_to_dst = self.matcher.match_views(
                    frames[src_view], frames[dst_view], src_pts
                )

                # Match dst -> src
                matched_src, conf_dst_to_src = self.matcher.match_views(
                    frames[dst_view], frames[src_view], dst_pts
                )

                n_matches = 0
                # For each src point, find if it matches a dst point (mutual nearest neighbor check)
                for src_idx, (matched_pt, conf) in enumerate(zip(matched_dst, conf_src_to_dst)):
                    if conf > threshold:
                        # Find closest dst point to the matched position
                        dists = np.linalg.norm(dst_pts - matched_pt, axis=1)
                        closest_dst_idx = np.argmin(dists)
                        closest_dist = dists[closest_dst_idx]

                        # Check mutual match: does dst_point[closest_dst_idx] also match back to src_point[src_idx]?
                        if closest_dist < 10:  # Within 10 pixels
                            back_conf = conf_dst_to_src[closest_dst_idx]
                            back_matched = matched_src[closest_dst_idx]
                            back_dist = np.linalg.norm(src_pts[src_idx] - back_matched)

                            if back_conf > threshold and back_dist < 10:
                                # Mutual match confirmed - union the nodes
                                src_node = node_to_id[(src_view, src_idx)]
                                dst_node = node_to_id[(dst_view, closest_dst_idx)]
                                union(src_node, dst_node)
                                n_matches += 1

                CONSOLE.print(f"    {n_matches} mutual matches found")

        # Step 3: Build tracks from connected components
        CONSOLE.print("\n[cyan]Step 3: Building tracks from connected components...")

        # Group nodes by their root
        component_to_nodes = {}
        for node_id in range(total_nodes):
            root = find(node_id)
            if root not in component_to_nodes:
                component_to_nodes[root] = []
            component_to_nodes[root].append(id_to_node[node_id])

        # Convert components to tracks
        tracks = []
        for root, nodes in component_to_nodes.items():
            # A track contains multiple (view_id, point_idx) pairs
            track = {
                'points': {},
                'confidence': {}
            }
            for view_id, pt_idx in nodes:
                pt = sift_points[view_id][pt_idx]
                track['points'][view_id] = pt.copy()
                track['confidence'][view_id] = 1.0  # Original detection has confidence 1.0
            tracks.append(track)

        CONSOLE.print(f"  Built {len(tracks)} tracks from {total_nodes} points")

        # Step 4: Filter tracks by visibility
        CONSOLE.print(f"\n[cyan]Step 4: Filtering tracks by visibility...")
        min_visible_views = self.config.matching.min_visible_views
        require_all = self.config.matching.require_visible_in_all_views

        valid_tracks = []
        visibility_dist = {}

        for track in tracks:
            n_visible = len(track['points'])
            visibility_dist[n_visible] = visibility_dist.get(n_visible, 0) + 1

            if require_all:
                if n_visible == n_views:
                    valid_tracks.append(track)
            else:
                if n_visible >= min_visible_views:
                    valid_tracks.append(track)

        CONSOLE.print(f"  Visibility distribution:")
        for n_vis in sorted(visibility_dist.keys()):
            CONSOLE.print(f"    {n_vis} views: {visibility_dist[n_vis]} tracks")

        filter_mode = 'all views' if require_all else f'at least {min_visible_views} view(s)'
        CONSOLE.print(f"  [green]Kept {len(valid_tracks)}/{len(tracks)} tracks visible in {filter_mode}")

        # Step 5: Limit to num_points
        max_points = self.config.num_points
        if len(valid_tracks) > max_points:
            CONSOLE.print(f"\n[yellow]Step 5: Limiting from {len(valid_tracks)} to {max_points} tracks...")
            # Sort by visibility count (prefer tracks visible in more views)
            def track_score(t):
                return len(t['points'])
            valid_tracks = sorted(valid_tracks, key=track_score, reverse=True)[:max_points]
            CONSOLE.print(f"  [green]Kept top {max_points} tracks")

        # Step 6: Convert tracks to query_points format
        CONSOLE.print(f"\n[cyan]Step 6: Building query points...")
        n_tracks = len(valid_tracks)

        if n_tracks == 0:
            CONSOLE.print("[red]No valid tracks found!")
            query_points = {vid: np.array([]).reshape(0, 2) for vid in view_ids}
            query_confidence = {vid: np.array([]) for vid in view_ids}
            return query_points, query_confidence

        # Initialize with zeros (invalid positions marked by confidence=0)
        query_points = {}
        query_confidence = {}
        for view_id in view_ids:
            query_points[view_id] = np.zeros((n_tracks, 2), dtype=np.float32)
            query_confidence[view_id] = np.zeros(n_tracks, dtype=np.float32)

        for track_idx, track in enumerate(valid_tracks):
            for view_id in view_ids:
                if view_id in track['points']:
                    query_points[view_id][track_idx] = track['points'][view_id]
                    query_confidence[view_id][track_idx] = track['confidence'][view_id]

        # Print statistics
        for view_id in view_ids:
            n_valid = (query_confidence[view_id] > 0).sum()
            CONSOLE.print(f"  View {view_id}: {n_valid}/{n_tracks} valid points")

        # Save debug visualization
        self._save_chain_matching_debug(frames, query_points, query_confidence)

        return query_points, query_confidence

    def _chain_matching(self) -> Dict[str, np.ndarray]:
        """
        Perform chain/ring matching across all views (better for many views).

        Two modes based on config.matching.local_feature_matching:

        local_feature_matching=False (original):
        1. Detects SIFT on all views
        2. Matches sequentially between adjacent views in the list
        3. (Ring mode) Also matches last view to first view to close the loop
        4. Propagates tracks across the entire chain/ring
        5. Each track may only be visible in a subset of views

        local_feature_matching=True (new):
        1. Each view detects its own SIFT/grid points
        2. Each view matches bidirectionally with ONLY its adjacent neighbors
        3. Build tracks using connected components (Union-Find)
        4. NO propagation - a track only spans views with real direct matches

        Returns:
            Dict mapping view_id -> [N, 2] query points for each view
        """
        view_ids = self.config.view_ids
        n_views = len(view_ids)
        threshold = self.config.matching.confidence_threshold
        is_ring = self.config.matching.matching_strategy == 'ring'
        local_feature = self.config.matching.local_feature_matching

        strategy_name = "Ring" if is_ring else "Chain"
        feature_mode = "Local Features" if local_feature else "Propagation"
        CONSOLE.print(f"\n[bold cyan]{strategy_name} Matching Strategy ({n_views} views, {feature_mode})")
        if is_ring:
            CONSOLE.print(f"  View order: {' -> '.join(view_ids)} -> {view_ids[0]} (ring)")
        else:
            CONSOLE.print(f"  View order: {' -> '.join(view_ids)}")
        if local_feature:
            CONSOLE.print(f"  [cyan]Mode: Each view detects own features, matches only with adjacent neighbors")

        # Load first frames and masks
        frames = {}
        masks = {}
        CONSOLE.print("\n[cyan]Loading frames and masks...")
        for view_id, view_tracker in self.view_trackers.items():
            frames[view_id] = view_tracker.load_frame(0)
            frame_path = view_tracker.frame_paths[0]
            target_shape = self.config.model.interp_shape
            masks[view_id] = self._load_mask(frame_path, target_shape)
            if masks[view_id] is not None:
                mask_coverage = (masks[view_id] > 0).sum() / masks[view_id].size * 100
                CONSOLE.print(f"  View {view_id}: Mask loaded, coverage: {mask_coverage:.1f}%")

        # Step 1: Detect SIFT on all views
        CONSOLE.print("\n[cyan]Step 1: Detecting SIFT keypoints on all views...")
        sift_points = {}
        for view_id in view_ids:
            sift_points[view_id] = self._detect_sift_points(
                frames[view_id],
                max_points=self.config.num_points,
                mask=masks[view_id]
            )
            CONSOLE.print(f"  View {view_id}: {len(sift_points[view_id])} points")

        # Branch based on local_feature_matching mode
        if local_feature:
            return self._local_feature_matching_impl(
                frames, masks, sift_points, view_ids, threshold, is_ring
            )

        # Step 2: Build tracks by chaining matches (original propagation mode)
        # Each track is a dict: {view_id: (x, y, confidence)}
        # We'll use Union-Find to merge tracks that connect
        CONSOLE.print("\n[cyan]Step 2: Chain matching between adjacent views...")

        # Initialize tracks from first view's SIFT points
        first_view = view_ids[0]
        tracks = []  # List of dicts: {view_id: np.array([x, y]), 'confidence': {view_id: conf}}
        for pt in sift_points[first_view]:
            tracks.append({
                'points': {first_view: pt.copy()},
                'confidence': {first_view: 1.0}
            })

        # Match each adjacent pair and extend tracks
        for i in range(n_views - 1):
            src_view = view_ids[i]
            dst_view = view_ids[i + 1]
            CONSOLE.print(f"\n  Matching {src_view} -> {dst_view}...")

            # Get points from current view that have track entries
            src_points = []
            src_track_indices = []
            for track_idx, track in enumerate(tracks):
                if src_view in track['points']:
                    src_points.append(track['points'][src_view])
                    src_track_indices.append(track_idx)

            if len(src_points) == 0:
                CONSOLE.print(f"    [yellow]No points in {src_view}, skipping")
                continue

            src_points = np.array(src_points)

            # Match to next view
            matched_points, confidence = self.matcher.match_views(
                frames[src_view], frames[dst_view], src_points
            )

            # Update tracks with matched points
            # Only keep matches with confidence > threshold, stop propagation for low confidence
            n_high_conf = 0
            for j, (track_idx, pt, conf) in enumerate(zip(src_track_indices, matched_points, confidence)):
                if conf > threshold:
                    tracks[track_idx]['points'][dst_view] = pt.copy()
                    tracks[track_idx]['confidence'][dst_view] = conf
                    n_high_conf += 1
                # else: don't add to track, stop propagation for this point

            CONSOLE.print(f"    {n_high_conf}/{len(src_points)} points matched (conf > {threshold})")

            # Also add SIFT points from dst_view that weren't matched
            # Match dst SIFT back to src to find new unique points
            dst_sift = sift_points[dst_view]
            if len(dst_sift) > 0:
                # Match dst -> src
                matched_back, conf_back = self.matcher.match_views(
                    frames[dst_view], frames[src_view], dst_sift
                )

                # Find dst SIFT points that don't match any existing track
                existing_src_points = src_points if len(src_points) > 0 else np.array([]).reshape(0, 2)
                new_tracks_count = 0

                for k, (dst_pt, src_pt, conf) in enumerate(zip(dst_sift, matched_back, conf_back)):
                    if conf > threshold:
                        # Check if this matches an existing track (within 5 pixels)
                        if len(existing_src_points) > 0:
                            dists = np.linalg.norm(existing_src_points - src_pt, axis=1)
                            if dists.min() < 5:  # Already in a track
                                continue

                        # New point - create new track starting from dst_view
                        tracks.append({
                            'points': {dst_view: dst_pt.copy()},
                            'confidence': {dst_view: 1.0}
                        })
                        new_tracks_count += 1

                if new_tracks_count > 0:
                    CONSOLE.print(f"    Added {new_tracks_count} new tracks from {dst_view} SIFT")

        # Ring matching: close the loop by matching last view <-> first view
        if is_ring and n_views >= 3:
            last_view = view_ids[-1]
            first_view = view_ids[0]
            CONSOLE.print(f"\n  [cyan]Ring closure: Matching {last_view} <-> {first_view}...")

            # Match last -> first (extend existing tracks)
            last_points = []
            last_track_indices = []
            for track_idx, track in enumerate(tracks):
                if last_view in track['points'] and first_view not in track['points']:
                    last_points.append(track['points'][last_view])
                    last_track_indices.append(track_idx)

            if len(last_points) > 0:
                last_points = np.array(last_points)
                matched_to_first, confidence = self.matcher.match_views(
                    frames[last_view], frames[first_view], last_points
                )

                n_ring_matched = 0
                for j, (track_idx, pt, conf) in enumerate(zip(last_track_indices, matched_to_first, confidence)):
                    if conf > threshold:
                        tracks[track_idx]['points'][first_view] = pt.copy()
                        tracks[track_idx]['confidence'][first_view] = conf
                        n_ring_matched += 1

                CONSOLE.print(f"    {last_view} -> {first_view}: {n_ring_matched}/{len(last_points)} tracks extended")

            # Match first -> last (extend existing tracks that have first but not last)
            first_points = []
            first_track_indices = []
            for track_idx, track in enumerate(tracks):
                if first_view in track['points'] and last_view not in track['points']:
                    first_points.append(track['points'][first_view])
                    first_track_indices.append(track_idx)

            if len(first_points) > 0:
                first_points = np.array(first_points)
                matched_to_last, confidence = self.matcher.match_views(
                    frames[first_view], frames[last_view], first_points
                )

                n_ring_matched = 0
                for j, (track_idx, pt, conf) in enumerate(zip(first_track_indices, matched_to_last, confidence)):
                    if conf > threshold:
                        tracks[track_idx]['points'][last_view] = pt.copy()
                        tracks[track_idx]['confidence'][last_view] = conf
                        n_ring_matched += 1

                CONSOLE.print(f"    {first_view} -> {last_view}: {n_ring_matched}/{len(first_points)} tracks extended")

        # Step 3: Filter tracks by visibility
        CONSOLE.print(f"\n[cyan]Step 3: Filtering tracks by visibility...")
        min_visible_views = self.config.matching.min_visible_views
        require_all = self.config.matching.require_visible_in_all_views

        valid_tracks = []
        visibility_dist = {}

        for track in tracks:
            n_visible = len(track['points'])
            visibility_dist[n_visible] = visibility_dist.get(n_visible, 0) + 1

            if require_all:
                if n_visible == n_views:
                    valid_tracks.append(track)
            else:
                if n_visible >= min_visible_views:
                    valid_tracks.append(track)

        CONSOLE.print(f"  Visibility distribution:")
        for n_vis in sorted(visibility_dist.keys()):
            CONSOLE.print(f"    {n_vis} views: {visibility_dist[n_vis]} tracks")

        filter_mode = 'all views' if require_all else f'at least {min_visible_views} view(s)'
        CONSOLE.print(f"  [green]Kept {len(valid_tracks)}/{len(tracks)} tracks visible in {filter_mode}")

        # Step 4: Limit to num_points
        max_points = self.config.num_points
        if len(valid_tracks) > max_points:
            CONSOLE.print(f"\n[yellow]Step 4: Limiting from {len(valid_tracks)} to {max_points} tracks...")
            # Sort by average confidence and visibility count
            def track_score(t):
                avg_conf = np.mean(list(t['confidence'].values()))
                n_views = len(t['points'])
                return n_views * 10 + avg_conf  # Prefer more views

            valid_tracks = sorted(valid_tracks, key=track_score, reverse=True)[:max_points]
            CONSOLE.print(f"  [green]Kept top {max_points} tracks")

        # Step 5: Convert tracks to query_points format
        # For views without a match, set position to (0,0) and confidence to 0
        # The triangulation step will skip these based on low confidence
        CONSOLE.print(f"\n[cyan]Step 5: Building query points...")
        n_tracks = len(valid_tracks)

        # Initialize with zeros (invalid positions marked by confidence=0)
        query_points = {}
        query_confidence = {}
        for view_id in view_ids:
            query_points[view_id] = np.zeros((n_tracks, 2), dtype=np.float32)
            query_confidence[view_id] = np.zeros(n_tracks, dtype=np.float32)

        for track_idx, track in enumerate(valid_tracks):
            for view_id in view_ids:
                if view_id in track['points']:
                    query_points[view_id][track_idx] = track['points'][view_id]
                    query_confidence[view_id][track_idx] = track['confidence'][view_id]
                # else: leave as zeros (position=0,0, confidence=0) to mark as invalid

        # Print statistics
        for view_id in view_ids:
            n_valid = (query_confidence[view_id] > threshold).sum()
            CONSOLE.print(f"  View {view_id}: {n_valid}/{n_tracks} valid points")

        # Save debug visualization
        self._save_chain_matching_debug(frames, query_points, query_confidence)

        # Return both points and confidence (confidence needed for triangulation)
        return query_points, query_confidence

    def _save_chain_matching_debug(
        self,
        frames: Dict[str, np.ndarray],
        query_points: Dict[str, np.ndarray],
        query_confidence: Dict[str, np.ndarray]
    ):
        """Save debug visualization for chain matching."""
        view_ids = self.config.view_ids
        n_views = len(view_ids)
        n_points = len(query_points[view_ids[0]])
        threshold = self.config.matching.confidence_threshold

        CONSOLE.print("\n[cyan]Saving chain matching debug visualizations...")

        # Create a combined view of all views with matched points
        sample_frame = frames[view_ids[0]]
        H, W = sample_frame.shape[:2]

        # Arrange views in a grid
        cols = min(4, n_views)
        rows = (n_views + cols - 1) // cols
        combined = np.zeros((H * rows, W * cols, 3), dtype=np.uint8)

        colors = self._generate_colors(n_points)

        for i, view_id in enumerate(view_ids):
            r, c = i // cols, i % cols
            frame = frames[view_id].copy()
            pts = query_points[view_id]
            conf = query_confidence[view_id]

            # Draw points
            for j, (pt, cf) in enumerate(zip(pts, conf)):
                if cf > threshold:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(frame, (x, y), 4, colors[j], -1)

            # Draw label
            n_valid = (conf > threshold).sum()
            cv2.putText(frame, f"View {view_id}: {n_valid}/{n_points}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined[r*H:(r+1)*H, c*W:(c+1)*W] = frame

        output_path = self.config.debug_dir / "chain_matching_all_views.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        CONSOLE.print(f"[green]Saved: {output_path}")

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        visibility: np.ndarray,
        colors: List[Tuple[int, int, int]],
        view_id: str,
        frame_idx: int
    ) -> np.ndarray:
        """Draw tracking visualization on a frame."""
        vis_frame = frame.copy()
        H, W = frame.shape[:2]

        # Draw points
        for i, (pt, vis, color) in enumerate(zip(tracks, visibility, colors)):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                # Draw filled circle for visible, hollow for occluded
                if vis > 0.5:
                    cv2.circle(vis_frame, (x, y), self.config.visualization.point_radius, color, -1)
                else:
                    cv2.circle(vis_frame, (x, y), self.config.visualization.point_radius, color, 1)

        # Draw view label
        label = f"View {view_id} - Frame {frame_idx}"
        cv2.putText(vis_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_frame

    def _create_combined_frame(
        self,
        frames: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple view frames into a single output frame.

        Args:
            frames: Dict mapping view_id -> visualization frame

        Returns:
            Combined frame
        """
        # Get frames in order of view_ids
        frame_list = [frames[vid] for vid in self.config.view_ids]

        if self.config.visualization.layout == 'horizontal':
            return np.concatenate(frame_list, axis=1)
        elif self.config.visualization.layout == 'vertical':
            return np.concatenate(frame_list, axis=0)
        else:  # grid
            # For 3 views, use 2x2 grid with empty last cell
            n_views = len(frame_list)
            cols = int(np.ceil(np.sqrt(n_views)))
            rows = int(np.ceil(n_views / cols))

            H, W = frame_list[0].shape[:2]
            grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

            for i, frame in enumerate(frame_list):
                r, c = i // cols, i % cols
                grid[r*H:(r+1)*H, c*W:(c+1)*W] = frame

            return grid

    def run(self):
        """
        Run multi-view tracking.

        Steps:
        1. Perform cross-view matching (star or chain strategy)
        2. Process each view sequentially (to save GPU memory):
           - Initialize tracker, process all frames, release tracker
        3. Generate combined visualization from stored results
        """
        CONSOLE.print("\n[bold cyan]" + "="*60)
        CONSOLE.print("[bold cyan]Multi-View Tracker (Sequential Processing)")
        CONSOLE.print("[bold cyan]" + "="*60 + "\n")

        # Disable live display if no display available
        if self.config.visualization.show_live and os.environ.get('DISPLAY') is None:
            CONSOLE.print("[yellow]No display available, disabling live visualization")
            self.config.visualization.show_live = False

        # Step 1: Cross-view matching (select strategy based on config)
        CONSOLE.print("[bold cyan]Phase 1: First Frame Matching")
        CONSOLE.print("-" * 40)

        strategy = self.config.matching.matching_strategy
        CONSOLE.print(f"[cyan]Using matching strategy: {strategy}")

        if strategy in ('chain', 'ring'):
            matched_points, initial_confidence = self._chain_matching()
        else:  # 'star' - original bidirectional matching
            matched_points, initial_confidence = self._bidirectional_matching()

        # Get point count from first view (all views should have same count now)
        first_view = self.config.view_ids[0]
        n_points = len(matched_points[first_view])
        if n_points == 0:
            CONSOLE.print("[red]No valid points found. Exiting.")
            return

        # Release matcher's tracker to free GPU memory before Phase 2
        self.matcher.release()

        # Generate colors for visualization
        colors = self._generate_colors(n_points)

        # Determine frame range
        min_frames = min(vt.num_frames for vt in self.view_trackers.values())
        start_frame = self.config.start_frame
        end_frame = self.config.end_frame if self.config.end_frame else min_frames
        end_frame = min(end_frame, min_frames)
        n_frames = end_frame - start_frame

        # Step 2: Process each view using a SHARED tracker
        CONSOLE.print(f"\n[bold cyan]Phase 2: Sequential Temporal Tracking")
        CONSOLE.print("-" * 40)
        CONSOLE.print(f"Processing frames {start_frame} to {end_frame} ({n_frames} frames)")
        CONSOLE.print(f"[cyan]Using shared tracker for all views (initialized once)...")

        all_tracks = {vid: [] for vid in self.config.view_ids}
        all_visibility = {vid: [] for vid in self.config.view_ids}

        # Create a shared tracker instance (only initialized once)
        first_view_tracker = self.view_trackers[self.config.view_ids[0]]
        first_view_tracker.build_tracker()
        shared_tracker = first_view_tracker.tracker
        CONSOLE.print(f"[green]Shared tracker initialized")

        for view_idx, view_id in enumerate(self.config.view_ids):
            view_tracker = self.view_trackers[view_id]
            CONSOLE.print(f"\n[cyan]Processing view {view_id} ({view_idx + 1}/{len(self.config.view_ids)})...")

            # Initialize tracking with shared tracker (just reset state, no re-initialization)
            view_tracker.initialize_tracking(matched_points[view_id], shared_tracker=shared_tracker)

            # Process all frames for this view
            for frame_idx in range(start_frame, end_frame):
                tracks, visibility = view_tracker.process_frame(frame_idx)
                all_tracks[view_id].append(tracks)
                all_visibility[view_id].append(visibility)

                if (frame_idx + 1) % 100 == 0:
                    CONSOLE.print(f"  [dim]View {view_id}: Frame {frame_idx + 1}/{end_frame}")

            CONSOLE.print(f"[green]View {view_id}: Completed {n_frames} frames")

            # Just clear reference, don't release the shared tracker
            view_tracker.release_tracker()

        # Now release the shared tracker
        CONSOLE.print(f"[cyan]Releasing shared tracker...")
        first_view_tracker._owns_tracker = True  # Mark as owner so it gets released
        first_view_tracker.tracker = shared_tracker
        first_view_tracker.release_tracker()

        # Step 3: Generate combined visualization from stored results
        CONSOLE.print(f"\n[bold cyan]Phase 3: Generating Visualization")
        CONSOLE.print("-" * 40)

        # Initialize video writer
        if self.config.visualization.save_video:
            ref_view_id = self.config.reference_view
            sample_frame = self.view_trackers[ref_view_id].load_frame(0)
            H, W = sample_frame.shape[:2]
            n_views = len(self.config.view_ids)

            if self.config.visualization.layout == 'horizontal':
                out_shape = (H, W * n_views)
            elif self.config.visualization.layout == 'vertical':
                out_shape = (H * n_views, W)
            else:
                cols = int(np.ceil(np.sqrt(n_views)))
                rows = int(np.ceil(n_views / cols))
                out_shape = (H * rows, W * cols)

            output_path = self.config.visualization.output_path
            if output_path is None:
                views_str = "_".join([f"v{vid}" for vid in self.config.view_ids])
                output_path = self.config.output_dir / f"tracking_{views_str}_f{start_frame}-{end_frame}.mp4"

            self.video_writer = imageio.get_writer(
                str(output_path),
                fps=self.config.visualization.fps,
                codec='libx264',
                quality=8
            )
            CONSOLE.print(f"[green]Saving video to: {output_path}")

        # Generate frames from stored tracking results
        for rel_frame_idx in range(n_frames):
            frame_idx = start_frame + rel_frame_idx
            vis_frames = {}

            for view_id in self.config.view_ids:
                view_tracker = self.view_trackers[view_id]
                tracks = all_tracks[view_id][rel_frame_idx]
                visibility = all_visibility[view_id][rel_frame_idx]

                # Load frame and draw visualization
                frame = view_tracker.load_frame(frame_idx)
                vis_frames[view_id] = self._draw_frame(
                    frame, tracks, visibility, colors, view_id, frame_idx
                )

            # Combine frames
            combined = self._create_combined_frame(vis_frames)

            # Show live
            if self.config.visualization.show_live:
                cv2.imshow("Multi-View Tracking", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    CONSOLE.print("\n[yellow]User requested quit")
                    break

            # Save to video
            if self.video_writer is not None:
                self.video_writer.append_data(combined)

            if (frame_idx + 1) % 100 == 0:
                CONSOLE.print(f"[cyan]Generated visualization for frame {frame_idx + 1}/{end_frame}")

        # Cleanup
        if self.video_writer is not None:
            self.video_writer.close()
            CONSOLE.print(f"\n[green]Video saved!")

        if self.config.visualization.show_live:
            cv2.destroyAllWindows()

        CONSOLE.print("\n[bold green]Multi-view tracking complete!")

        # Return tracks, visibility, and initial cross-view matching confidence
        # initial_confidence is important for triangulation to know which views have valid matches
        return all_tracks, all_visibility, initial_confidence

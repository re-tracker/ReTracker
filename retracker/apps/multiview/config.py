"""Configuration for multi-view tracking."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
from pathlib import Path


@dataclass
class ViewConfig:
    """Configuration for a single view."""
    view_id: str  # e.g., "19", "25", "28"
    images_dir: Path  # Path to images directory
    is_reference: bool = False  # True if this is the reference view for point selection


@dataclass
class SIFTConfig:
    """SIFT keypoint detection configuration."""
    use_sift: bool = False  # If False, use only uniform grid points (default)
    n_features: int = 500  # Number of keypoints to detect (0 = detect all)
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6


@dataclass
class MatchingConfig:
    """Cross-view matching configuration."""
    method: Literal['model', 'sift_bf'] = 'model'  # 'model' uses ReTracker, 'sift_bf' uses brute-force SIFT matching
    confidence_threshold: float = 0.3  # Minimum confidence for valid matches

    # Matching strategy for multi-view
    # 'star': All views match to a single reference view (original, good for 3 views)
    # 'chain': Sequential matching between adjacent views (better for many views)
    # 'ring': Chain + first-last view matching (best for surrounding cameras)
    matching_strategy: Literal['star', 'chain', 'ring'] = 'ring'

    # Local feature detection: each view detects its own features and matches with neighbors
    # If True: each view detects grid/SIFT points and matches bidirectionally with adjacent views
    # If False: features propagate from first view through the chain (original behavior)
    # Recommended True for many views to avoid accumulated matching errors
    local_feature_matching: bool = True

    # Visibility filtering: control how many views a point must be visible in
    # If require_visible_in_all_views=True: point must be visible in ALL views (strict)
    # If require_visible_in_all_views=False: use min_visible_views parameter
    require_visible_in_all_views: bool = False  # Changed default to False for multi-view
    # Minimum number of views a point must be visible in (only used when require_visible_in_all_views=False)
    # Set to 2 for at least stereo triangulation, 3+ for more robust results
    min_visible_views: int = 3


@dataclass
class TrackingConfig:
    """Temporal tracking configuration."""
    use_streaming: bool = True  # Use streaming mode for efficiency
    query_batch_size: int = 256  # Batch size for query processing


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    show_live: bool = True  # Show live visualization
    save_video: bool = True  # Save output video
    output_path: Optional[Path] = None  # Output video path
    point_radius: int = 4
    line_thickness: int = 1
    show_trajectories: bool = False  # Show trajectory traces
    trajectory_length: int = 30  # Number of frames to show in trajectory
    fps: int = 30  # Output video FPS
    layout: Literal['horizontal', 'vertical', 'grid'] = 'horizontal'  # View layout


@dataclass
class ModelConfig:
    """Model configuration."""
    ckpt_path: Optional[str] = None  # Required: specify via config or command line
    device: str = "cuda"
    # IMPORTANT: interp_shape must match the input image size
    # The engine scales coordinates based on the ratio of input size to interp_shape
    interp_shape: Tuple[int, int] = (512, 512)  # (H, W) - must match input image size
    compile: bool = False
    use_amp: bool = True
    dtype: Literal['fp32', 'fp16', 'bf16'] = 'bf16'

    # High-resolution inference settings
    enable_highres_inference: bool = True  # Enable by default for correct matching
    coarse_resolution: Tuple[int, int] = (512, 512)  # (H, W) for coarse/global stage

    # Multi-GPU settings
    # When devices has multiple GPUs, queries are split across them for parallel processing
    # Example: devices=('cuda:0', 'cuda:1') will use both GPUs
    # If None, uses the single device specified above
    devices: Optional[Tuple[str, ...]] = None

    # Query batching settings (to prevent OOM with many query points)
    query_batch_size: int = 256

    # Task-specific visibility thresholds
    tracking_visibility_threshold: float = 0.1
    matching_visibility_threshold: float = 0.5


@dataclass
class MultiViewConfig:
    """
    Configuration for multi-view tracking.

    Example usage:
        config = MultiViewConfig(
            data_root=Path("data/multiview_tracker/0172_05/images"),
            view_ids=["19", "25", "28"],
            reference_view="25",
            num_points=100,
        )
    """
    # Data configuration
    data_root: Path = None  # Root directory containing view subdirectories
    view_ids: List[str] = field(default_factory=lambda: ["19", "25", "28"])
    reference_view: str = "25"  # Reference view for point selection

    # Output configuration
    output_base: Path = field(default_factory=lambda: Path("outputs/multiview_tracking"))

    # Point selection
    num_points: int = 400  # Number of points to track (also limits max points to prevent OOM)

    # Sub-configurations
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Processing options
    start_frame: int = 0
    end_frame: Optional[int] = None  # None = process all frames
    verbose: bool = True

    @property
    def dataset_name(self) -> str:
        """Extract dataset name from data_root parent (e.g., '0172_05')."""
        if self.data_root is None:
            return "unknown"
        # data_root is typically .../0172_05/images, so parent.name gives dataset name
        return self.data_root.parent.name

    @property
    def output_dir(self) -> Path:
        """Output directory: outputs/multiview_tracking/{dataset_name}/"""
        return self.output_base / self.dataset_name

    @property
    def debug_dir(self) -> Path:
        """Debug output directory."""
        return self.output_dir / "debug"

    def get_view_configs(self) -> List[ViewConfig]:
        """Generate ViewConfig for each view."""
        configs = []
        for view_id in self.view_ids:
            configs.append(ViewConfig(
                view_id=view_id,
                images_dir=self.data_root / view_id,
                is_reference=(view_id == self.reference_view)
            ))
        return configs

    def validate(self):
        """Validate configuration."""
        if self.data_root is None:
            raise ValueError("data_root must be specified")
        if not self.data_root.exists():
            raise ValueError(f"data_root does not exist: {self.data_root}")
        if self.reference_view not in self.view_ids:
            raise ValueError(f"reference_view '{self.reference_view}' must be in view_ids")
        for view_id in self.view_ids:
            view_dir = self.data_root / view_id
            if not view_dir.exists():
                raise ValueError(f"View directory does not exist: {view_dir}")

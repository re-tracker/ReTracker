"""Base configuration classes for runnable apps (tracking/streaming)."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Literal
from pathlib import Path
import yaml
import json


@dataclass
class VideoConfig:
    """Video loading and preprocessing configuration."""
    
    resized_wh: Tuple[int, int] = (512, 384)  # (W, H)
    auto_rotate: bool = True  # Auto-detect and apply rotation
    max_frames: Optional[int] = None  # Limit frames for testing
    backend: Literal['opencv', 'mediapy', 'auto'] = 'auto'
    

@dataclass
class ModelConfig:
    """Model configuration."""

    ckpt_path: Optional[str] = None  # Required: specify via config or command line
    interp_shape: Tuple[int, int] = (512, 512)  # (H, W)
    dtype: Literal['fp32', 'fp16', 'bf16'] = 'bf16'
    # Use "auto" so CPU-only environments can still run (falls back to "cpu").
    device: str = 'auto'
    use_amp: bool = True
    # Speed up cold-start by building the DINOv3 hub backbones without loading
    # hub weights. This is intended for inference when `ckpt_path` points to a
    # full ReTracker checkpoint that already contains backbone weights.
    #
    # Note: this does NOT change the model architecture. It only affects hub
    # weight loading during initialization.
    fast_start: bool = False

    # Multi-GPU settings
    # When devices has multiple GPUs, queries are split across them for parallel processing
    # Example: devices=['cuda:0', 'cuda:1'] will use both GPUs
    # If None or empty, uses the single device specified above
    devices: Optional[Tuple[str, ...]] = None

    # High-resolution inference settings
    # When enabled, coarse stage uses coarse_resolution while refinement uses original resolution
    enable_highres_inference: bool = False
    coarse_resolution: Tuple[int, int] = (512, 512)  # (H, W) for coarse/global stage

    # Dense matching settings
    # When enabled, outputs W*W points per query (patch-level dense predictions)
    enable_dense_matching: bool = False
    dense_level: int = 2  # Refinement level for dense matching (0=coarsest, 2=finest)

    # Query batching settings (to prevent OOM with many query points)
    # When processing more than query_batch_size queries per frame, split into batches
    query_batch_size: int = 256

    # Task-specific visibility thresholds
    tracking_visibility_threshold: float = 0.1
    matching_visibility_threshold: float = 0.5

    # torch.compile settings
    # When enabled, compiles TROMA blocks for faster inference (20-30% speedup after warmup)
    compile: bool = False
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'reduce-overhead'
    compile_warmup: bool = True  # Run warmup after compilation to trigger JIT
    

@dataclass
class QueryConfig:
    """Query points generation configuration."""

    strategy: Literal['grid', 'segmentation', 'detection', 'custom', 'sift'] = 'grid'
    grid_size: int = 20
    initial_frame: int = 0  # Which frame to generate queries
    use_mask_filter: bool = True  # Filter by segmentation mask
    margin_ratio: float = 1.0 / 64  # Grid margin ratio

    # SIFT-specific settings
    sift_n_features: int = 0  # 0 means detect all features
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    
    fps: int = 10
    linewidth: int = 3
    tracks_leave_trace: int = 1
    hide_occ_points: bool = True
    colormap: str = 'jet'
    # Optional separate resize for visualization output only.
    # When set, tracking runs at video.resized_wh, but the saved MP4 is rendered at vis_resized_wh.
    vis_resized_wh: Optional[Tuple[int, int]] = None  # (W, H)
    

@dataclass
class OutputConfig:
    """Output configuration."""
    
    output_dir: str = "./outputs"
    save_video: bool = True
    save_npz: bool = False
    save_images: bool = False
    save_txt: bool = False  # For future evaluation
    

@dataclass
class TrackingConfig:
    """Configuration for the offline tracking app."""
    
    # Input
    video_path: Optional[str] = None
    seg_path: Optional[str] = None
    query_path: Optional[str] = None
    
    # Sub-configs
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(output_dir="./outputs/tracking"))
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrackingConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrackingConfig":
        """Create config from dictionary."""
        # Extract nested configs
        video_data = data.pop('video', {})
        model_data = data.pop('model', {})
        query_data = data.pop('query', {})
        vis_data = data.pop('visualization', {})
        output_data = data.pop('output', {})
        
        return cls(
            **data,
            video=VideoConfig(**video_data),
            model=ModelConfig(**model_data),
            query=QueryConfig(**query_data),
            visualization=VisualizationConfig(**vis_data),
            output=OutputConfig(**output_data),
        )
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)

"""Streaming-specific configuration classes."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional, Tuple, Literal, List
from .base_config import (
    ModelConfig,
    QueryConfig,
    VisualizationConfig,
    OutputConfig
)


@dataclass
class StreamingSourceConfig:
    """Configuration for streaming video source."""

    # Source type
    source_type: Literal['camera', 'video_file', 'rtsp', 'http', 'image_sequence'] = 'camera'

    # Camera settings
    camera_id: int = 0  # Camera device ID (0 for default camera)

    # Video file settings (for simulation)
    video_path: Optional[str] = None
    simulate_realtime: bool = True  # Simulate real-time playback
    target_fps: float = 30.0  # Target FPS for simulation

    # Image sequence settings
    image_dir: Optional[str] = None  # Directory containing images
    image_pattern: str = '*'  # Glob pattern for image files (e.g., '*.jpg', '*.png')
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    sort_by: Literal['name', 'natural', 'mtime'] = 'natural'  # Sort order for images

    # Frame range settings (for video_file and image_sequence source)
    # Format: List of (start_frame, end_frame) tuples
    # Example: [(0, 50), (100, 150), (200, 300)] - track 3 segments
    # Empty list or None = process all frames
    frame_segments: Optional[List[Tuple[int, int]]] = None

    # RTSP settings (for network cameras)
    rtsp_url: Optional[str] = None

    # HTTP stream settings (for IP Webcam apps, MJPEG streams, etc.)
    http_url: Optional[str] = None
    rtsp_use_threading: bool = True  # Use background thread for reading
    rtsp_use_gstreamer: bool = False  # Use GStreamer backend (requires GStreamer installed)
    rtsp_max_retries: int = 5  # Max reconnection attempts
    rtsp_retry_delay: float = 2.0  # Delay between retries (seconds)
    rtsp_timeout: float = 10.0  # Read timeout (seconds)
    rtsp_queue_size: int = 2  # Frame queue size for threaded reading

    # Frame preprocessing
    resized_wh: Tuple[int, int] = (512, 384)  # (W, H)
    auto_rotate: bool = True

    # Center crop settings
    # If enabled, crop the center region of the frame before resizing
    enable_center_crop: bool = False
    center_crop_ratio: float = 1.0  # Crop ratio (0.5 = crop to 50% of original size)

    # Buffer settings
    buffer_size: int = 30  # Number of frames to buffer
    skip_frames: int = 0  # Skip N frames between processing


@dataclass
class StreamingProcessingConfig:
    """Configuration for streaming processing behavior."""

    # Processing mode (currently only global tracking is supported)
    mode: Literal['global'] = 'global'

    # Query update strategy
    query_update_strategy: Literal['static', 'dynamic', 'redetect'] = 'static'
    redetect_interval: int = 30  # Frames between re-detection (if dynamic)

    # Performance settings
    max_points: int = 64  # Maximum number of points to track (default 64 for real-time)
    enable_async: bool = False  # Async processing (future)


@dataclass
class StreamingVisualizationConfig:
    """Visualization configuration for streaming."""

    # Display settings
    show_live: bool = True  # Show live visualization window
    window_name: str = "Streaming Tracker"
    display_fps: bool = True  # Show FPS counter
    display_info: bool = True  # Show tracking info

    # Recording settings
    record_output: bool = False  # Record output to video file
    output_path: Optional[str] = None

    # Plot mode
    # - 'tracks': show trajectories over time (default for video)
    # - 'pairs': show matching lines between first frame and current frame (default for image_sequence)
    plot_mode: Literal['tracks', 'pairs'] = 'tracks'

    # Whether to draw matching lines between corresponding points in 'pairs' mode
    show_matching_lines: bool = True

    # Confidence filtering
    show_low_confidence: bool = True  # Show low confidence points (vis <= 0.5)

    # Motion filtering
    # Skip drawing trajectory line when motion exceeds this threshold (in pixels)
    # Set to 0 or None to disable filtering
    max_motion_threshold: float = 50.0

    # Visualization style
    linewidth: int = 1
    point_radius: int = 3  # Small points for cleaner display
    tracks_leave_trace: int = 10  # Number of frames to leave trace
    colormap: str = 'jet'
    alpha: float = 0.7  # Transparency for overlays


@dataclass
class StreamingConfig:
    """Configuration for streaming/online tracking."""

    # Sub-configs
    source: StreamingSourceConfig = field(default_factory=StreamingSourceConfig)
    processing: StreamingProcessingConfig = field(default_factory=StreamingProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    visualization: StreamingVisualizationConfig = field(default_factory=StreamingVisualizationConfig)
    # Keep streaming outputs separate by default.
    output: OutputConfig = field(default_factory=lambda: OutputConfig(output_dir="./outputs/streaming"))

    # Global settings
    max_duration: Optional[float] = None  # Maximum duration in seconds (None = unlimited)
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "StreamingConfig":
        """Load config from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict) -> "StreamingConfig":
        """Create config from dictionary."""
        source_data = data.pop('source', {})
        processing_data = data.pop('processing', {})
        model_data = data.pop('model', {})
        query_data = data.pop('query', {})
        vis_data = data.pop('visualization', {})
        output_data = data.pop('output', {})

        return cls(
            **data,
            source=StreamingSourceConfig(**source_data),
            processing=StreamingProcessingConfig(**processing_data),
            model=ModelConfig(**model_data),
            query=QueryConfig(**query_data),
            visualization=StreamingVisualizationConfig(**vis_data),
            output=OutputConfig(**output_data),
        )

    def to_dict(self) -> dict:
        """Convert config to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

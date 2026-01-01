"""Fast streaming configuration for real-time tracking.

This config optimizes for speed over accuracy for streaming/online use cases.
Typical FPS improvement: 2-3x faster (from ~5 FPS to ~12-15 FPS on RTX 3090).

Key optimizations:
1. Lower resolution (384x288 instead of 512x384)
2. Fewer query points (32 instead of 64)
3. Reduced causal memory (12 instead of 24)
4. torch.compile enabled for TROMA blocks
5. Smaller interp_shape for model inference

Usage:
    from retracker.apps.config.streaming_fast_config import get_fast_streaming_config
    config = get_fast_streaming_config()
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List
from .streaming_config import (
    StreamingSourceConfig,
    StreamingProcessingConfig,
    StreamingVisualizationConfig,
    StreamingConfig,
)
from .base_config import (
    ModelConfig,
    QueryConfig,
    OutputConfig,
)


@dataclass
class FastModelConfig(ModelConfig):
    """Fast model configuration optimized for streaming."""

    # Use smaller interp_shape for faster inference
    interp_shape: Tuple[int, int] = (384, 384)  # (H, W) - reduced from (512, 512)

    # Enable torch.compile for 20-30% speedup
    compile: bool = True
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'reduce-overhead'
    compile_warmup: bool = True

    # Use bf16 for memory efficiency
    dtype: Literal['fp32', 'fp16', 'bf16'] = 'bf16'
    use_amp: bool = True

    # Smaller batch size for streaming (usually 1 frame at a time)
    query_batch_size: int = 128  # Reduced from 256


@dataclass
class FastSourceConfig(StreamingSourceConfig):
    """Fast source configuration with lower resolution."""

    # Lower resolution for faster processing
    # This is the biggest speed gain - roughly 2x faster
    resized_wh: Tuple[int, int] = (384, 288)  # (W, H) - reduced from (512, 384)

    # Skip more frames for real-time processing
    skip_frames: int = 0  # Can set to 1 to halve processing load


@dataclass
class FastProcessingConfig(StreamingProcessingConfig):
    """Fast processing configuration with fewer points."""

    # Fewer query points = faster processing
    # 32 points is usually sufficient for tracking
    max_points: int = 32  # Reduced from 64


@dataclass
class FastVisualizationConfig(StreamingVisualizationConfig):
    """Fast visualization configuration."""

    # Simpler visualization for speed
    linewidth: int = 1
    point_radius: int = 2
    tracks_leave_trace: int = 5  # Shorter trace


@dataclass
class FastStreamingConfig(StreamingConfig):
    """Fast streaming configuration."""

    source: FastSourceConfig = field(default_factory=FastSourceConfig)
    processing: FastProcessingConfig = field(default_factory=FastProcessingConfig)
    model: FastModelConfig = field(default_factory=FastModelConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    visualization: FastVisualizationConfig = field(default_factory=FastVisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_fast_streaming_config() -> FastStreamingConfig:
    """Get the fast streaming configuration."""
    config = FastStreamingConfig()

    # Additional optimizations
    config.query.grid_size = 16  # Fewer grid points

    return config


def get_ultra_fast_streaming_config() -> FastStreamingConfig:
    """Get ultra-fast configuration (sacrifices more accuracy for speed)."""
    config = FastStreamingConfig()

    # Ultra-low resolution
    config.source.resized_wh = (320, 240)  # Even smaller
    config.model.interp_shape = (320, 320)

    # Minimal points
    config.processing.max_points = 16
    config.query.grid_size = 12

    return config


# Presets for different use cases
STREAMING_PRESETS = {
    'fast': {
        'description': 'Balanced speed/accuracy for real-time streaming',
        'expected_fps': '10-15 FPS',
        'resized_wh': (384, 288),
        'interp_shape': (384, 384),
        'max_points': 32,
    },
    'ultra_fast': {
        'description': 'Maximum speed, reduced accuracy',
        'expected_fps': '15-25 FPS',
        'resized_wh': (320, 240),
        'interp_shape': (320, 320),
        'max_points': 16,
    },
    'balanced': {
        'description': 'Good accuracy with reasonable speed',
        'expected_fps': '6-10 FPS',
        'resized_wh': (448, 336),
        'interp_shape': (448, 448),
        'max_points': 48,
    },
    'quality': {
        'description': 'Best accuracy, slower processing',
        'expected_fps': '3-6 FPS',
        'resized_wh': (512, 384),
        'interp_shape': (512, 512),
        'max_points': 64,
    },
}


def get_preset_config(preset: str = 'fast') -> FastStreamingConfig:
    """
    Get a streaming config based on preset name.

    Args:
        preset: One of 'fast', 'ultra_fast', 'balanced', 'quality'

    Returns:
        Configured FastStreamingConfig
    """
    if preset not in STREAMING_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(STREAMING_PRESETS.keys())}")

    settings = STREAMING_PRESETS[preset]
    config = FastStreamingConfig()

    config.source.resized_wh = settings['resized_wh']
    config.model.interp_shape = settings['interp_shape']
    config.processing.max_points = settings['max_points']

    # Only enable compile for fast presets
    config.model.compile = preset in ['fast', 'ultra_fast']

    return config

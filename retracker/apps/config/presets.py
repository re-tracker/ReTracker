"""Preset configurations for quick start."""

from .base_config import (
    TrackingConfig,
    VideoConfig,
    ModelConfig,
    QueryConfig,
    VisualizationConfig,
    OutputConfig,
)
from retracker.utils.rich_utils import CONSOLE


def get_preset_config(preset_name: str) -> TrackingConfig:
    """
    Get predefined configuration presets.
    
    Args:
        preset_name: Name of the preset ('fast', 'balanced', 'high_quality')
        
    Returns:
        TrackingConfig instance
    """
    presets = {
        'fast': TrackingConfig(
            video=VideoConfig(resized_wh=(256, 192)),
            model=ModelConfig(dtype='bf16'),
            query=QueryConfig(grid_size=10),
            visualization=VisualizationConfig(fps=15),
        ),
        'balanced': TrackingConfig(
            video=VideoConfig(resized_wh=(512, 384)),
            model=ModelConfig(dtype='bf16'),
            query=QueryConfig(grid_size=20),
            visualization=VisualizationConfig(fps=10),
        ),
        'high_quality': TrackingConfig(
            video=VideoConfig(resized_wh=(1024, 768)),
            model=ModelConfig(dtype='fp32'),
            query=QueryConfig(grid_size=30),
            visualization=VisualizationConfig(fps=10, linewidth=4),
        ),
        'debug': TrackingConfig(
            video=VideoConfig(resized_wh=(256, 192), max_frames=50),
            model=ModelConfig(dtype='bf16'),
            query=QueryConfig(grid_size=5),
            visualization=VisualizationConfig(fps=5),
            output=OutputConfig(save_npz=True),
        ),
    }
    
    if preset_name not in presets:
        CONSOLE.print(f"[yellow]Warning: Unknown preset '{preset_name}', using 'balanced'[/yellow]")
        preset_name = 'balanced'
    
    return presets[preset_name]

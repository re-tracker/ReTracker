"""
ReTracker: A unified framework for point tracking and image matching.

Example usage:
    >>> from retracker import ReTracker, ReTrackerEngine
    >>>
    >>> # For inference
    >>> engine = ReTrackerEngine(ckpt_path="path/to/checkpoint.pth")
    >>> tracks, visibility = engine.video_forward(video, queries)
"""

from __future__ import annotations

from typing import Any

from retracker.version import __version__


# Note: Avoid importing heavy dependencies here to keep import time fast
# Users can explicitly import what they need

__all__ = [
    "__version__",
    # Lazy public API
    "ReTracker",
    "ReTrackerEngine",
    "StreamingEngine",
    "Visualizer",
    "ResultsDumper",
    "dump_results",
    "VideoFramesWriter",
]

# Lazy imports for optional components
def __getattr__(name: str) -> Any:
    """Lazy import for heavy modules."""
    if name == "ReTracker":
        from retracker.models import ReTracker
        return ReTracker
    elif name == "ReTrackerEngine":
        from retracker.inference import ReTrackerEngine
        return ReTrackerEngine
    elif name == "StreamingEngine":
        from retracker.inference import StreamingEngine
        return StreamingEngine
    elif name == "Visualizer":
        from retracker.visualization import Visualizer
        return Visualizer
    elif name == "ResultsDumper":
        from retracker.io.results import ResultsDumper
        return ResultsDumper
    elif name == "dump_results":
        from retracker.io.results import dump_results
        return dump_results
    elif name == "VideoFramesWriter":
        from retracker.io.images import VideoFramesWriter
        return VideoFramesWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

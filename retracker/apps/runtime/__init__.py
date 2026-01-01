"""Runtime building blocks for apps (pipelines, trackers, IO).

Keep this package lightweight at import time: avoid importing optional/heavy
dependencies (e.g. cv2, matplotlib) unless the runtime components are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "VideoLoaderFactory",
    "QueryGeneratorFactory",
    "Tracker",
    "TrackingPipeline",
    "StreamingSourceFactory",
    "StreamingTrackingPipeline",
]


def __getattr__(name: str) -> Any:
    if name == "VideoLoaderFactory":
        from .video_loader import VideoLoaderFactory

        return VideoLoaderFactory
    if name == "QueryGeneratorFactory":
        from .query_generator import QueryGeneratorFactory

        return QueryGeneratorFactory
    if name == "Tracker":
        from .tracker import Tracker

        return Tracker
    if name == "TrackingPipeline":
        from .pipeline import TrackingPipeline

        return TrackingPipeline
    if name == "StreamingSourceFactory":
        from .streaming_source import StreamingSourceFactory

        return StreamingSourceFactory
    if name == "StreamingTrackingPipeline":
        from .streaming_pipeline import StreamingTrackingPipeline

        return StreamingTrackingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

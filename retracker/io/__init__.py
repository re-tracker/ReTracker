"""I/O helpers for ReTracker.

Keep I/O code here (serialization, results export) instead of `retracker.utils`
to avoid turning `utils` into a grab-bag of domain-specific helpers.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "ResultsDumper",
    "dump_results",
    "VideoFramesWriter",
    "as_uint8_frames",
    "save_video_frames",
]


def __getattr__(name: str) -> Any:
    # Keep imports lazy to avoid pulling torch/PIL unless IO is used.
    if name in {"ResultsDumper", "dump_results"}:
        from retracker.io.results import ResultsDumper, dump_results

        return ResultsDumper if name == "ResultsDumper" else dump_results
    if name in {"VideoFramesWriter", "as_uint8_frames", "save_video_frames"}:
        from retracker.io.images import VideoFramesWriter, as_uint8_frames, save_video_frames

        if name == "VideoFramesWriter":
            return VideoFramesWriter
        if name == "as_uint8_frames":
            return as_uint8_frames
        return save_video_frames
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

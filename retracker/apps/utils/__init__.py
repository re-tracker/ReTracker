"""Utilities for runnable apps (tracking/streaming/SLAM).

Keep this package lightweight at import time: avoid importing torch/cv2 unless
the corresponding utilities are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_points_on_a_grid",
    "filter_queries_by_mask",
    "detect_video_rotation",
    "prepare_demo_video",
]


def __getattr__(name: str) -> Any:
    if name in {"get_points_on_a_grid", "filter_queries_by_mask"}:
        from .geometry import get_points_on_a_grid, filter_queries_by_mask

        return get_points_on_a_grid if name == "get_points_on_a_grid" else filter_queries_by_mask
    if name in {"detect_video_rotation", "prepare_demo_video"}:
        from .video_utils import detect_video_rotation, prepare_demo_video

        return detect_video_rotation if name == "detect_video_rotation" else prepare_demo_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

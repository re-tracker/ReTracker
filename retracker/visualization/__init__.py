"""Visualization utilities for ReTracker.

Keep this module import-lightweight. Some visualization helpers may rely on
optional dependencies (e.g. matplotlib). Those imports should only happen when
the corresponding symbols are accessed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Visualizer",
]


def __getattr__(name: str) -> Any:
    if name == "Visualizer":
        from retracker.visualization.visualizer import Visualizer

        return Visualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


"""Backward-compatible entrypoint for training geometry helpers.

The canonical implementation lives in `retracker.training.geometry`.
These utilities are used for generating supervision signals and are not required
for inference.
"""

from __future__ import annotations

from typing import Any


def _load_training_geometry():
    try:
        from retracker.training import geometry as _geometry
    except Exception as exc:  # pragma: no cover - depends on optional training deps
        raise ImportError(
            "Training geometry utilities are not available. "
            "This build of ReTracker is likely inference-only; install the training extras "
            "or use a full source checkout that includes `retracker.training`."
        ) from exc
    return _geometry


def __getattr__(name: str) -> Any:  # PEP 562
    return getattr(_load_training_geometry(), name)


def __dir__() -> list[str]:
    try:
        mod = _load_training_geometry()
    except Exception:
        return []
    return sorted(set(globals().keys()) | set(dir(mod)))

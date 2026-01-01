"""Backward-compatible entrypoint for training supervision utilities.

These helpers are used only for training. The canonical implementation lives in
`retracker.training.supervision`.

This indirection lets us keep the public import path stable while allowing an
inference-only distribution to omit the training package entirely.
"""

from __future__ import annotations

from typing import Any


def _load_training_supervision():
    try:
        from retracker.training import supervision as _supervision
    except Exception as exc:  # pragma: no cover - depends on optional training deps
        raise ImportError(
            "Training supervision utilities are not available. "
            "This build of ReTracker is likely inference-only; install the training extras "
            "or use a full source checkout that includes `retracker.training`."
        ) from exc
    return _supervision


def __getattr__(name: str) -> Any:  # PEP 562
    return getattr(_load_training_supervision(), name)


def __dir__() -> list[str]:
    try:
        mod = _load_training_supervision()
    except Exception:
        return []
    return sorted(set(globals().keys()) | set(dir(mod)))

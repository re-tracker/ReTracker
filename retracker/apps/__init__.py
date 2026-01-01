"""ReTracker runnable apps.

This package contains installable, user-facing applications built on top of the
core `retracker/` library (e.g. tracking, streaming).

Keep this module lightweight: avoid importing heavy dependencies at import time.
"""

from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "config",
    "utils",
    "runtime",
    "components",
    "multiview",
]


def __getattr__(name: str) -> Any:
    # Lazy module access (helps keep `import retracker` fast).
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

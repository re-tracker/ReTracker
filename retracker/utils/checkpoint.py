"""Checkpoint loading helpers.

We prefer `torch.load(..., weights_only=True)` where possible:
- safer (avoids unpickling arbitrary objects),
- often faster for Lightning checkpoints when we only need tensors.

We keep a small compatibility fallback for older Torch versions or checkpoints
that require a full load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from retracker.utils.rich_utils import CONSOLE


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load a torch checkpoint with a safe default.

    Args:
        path: Checkpoint path.
        map_location: Where to load tensors (default: CPU).
        weights_only: Prefer weights-only loading when supported.

    Returns:
        The deserialized checkpoint object (usually a dict).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older torch versions: no `weights_only` kwarg.
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if not weights_only:
            raise

        # Some checkpoints may require a full load (e.g. legacy pickled objects).
        CONSOLE.print(
            f"[yellow]safe_torch_load: weights_only load failed for {path!s} "
            f"({type(exc).__name__}: {exc}); retrying full load[/yellow]"
        )
        return torch.load(path, map_location=map_location, weights_only=False)  # Contains non-tensor data


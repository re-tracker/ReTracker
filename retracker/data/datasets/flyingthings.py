"""FlyingThings3D dataset loader (tracking).

This dataset is supported in the original research codebase but is not yet
fully ported/refactored in this repository's packaging-first layout.

If you need it:
1) Prefer using `pointodyssey` (or `dummy`) for now.
2) Port the implementation from the original research codebase.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class FlyingThingsDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        raise NotImplementedError(
            "FlyingThingsDataset is not yet available in this refactor. "
            "Use dataset_name=pointodyssey or dataset_name=dummy for now."
        )


__all__ = ["FlyingThingsDataset"]


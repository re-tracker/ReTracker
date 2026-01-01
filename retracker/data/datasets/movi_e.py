"""MOVi-E / Kubrics-style dataset loader (tracking).

This module is kept as a placeholder to keep configs/builders stable. The full
implementation has not been ported into this refactor yet.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class MOViEDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MOViEDataset is not yet available in this refactor. "
            "Use dataset_name=pointodyssey or dataset_name=dummy for now."
        )


__all__ = ["MOViEDataset"]


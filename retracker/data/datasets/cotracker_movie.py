"""K-EPIC / CoTracker movie-style datasets (placeholder)."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class KubricMovifEpicDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "KubricMovifEpicDataset is not yet available in this refactor. "
            "Use dataset_name=pointodyssey or dataset_name=dummy for now."
        )


__all__ = ["KubricMovifEpicDataset"]


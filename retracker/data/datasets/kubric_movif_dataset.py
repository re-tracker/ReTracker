"""CoTracker3 Kubric dataset wrappers (placeholder).

These datasets are referenced by configs/builders but are not yet ported into
this refactor. Keeping placeholders allows the rest of the training stack to be
imported and used with other datasets.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class KubricMovifDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "KubricMovifDataset is not yet available in this refactor. "
            "Use dataset_name=pointodyssey or dataset_name=dummy for now."
        )


__all__ = ["KubricMovifDataset"]


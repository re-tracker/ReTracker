"""Kubric / Panning MOVi-E dataset wrapper (tracking).

The historical implementation relies on TensorFlow Datasets (TFDS) to read the
`panning_movi_e` dataset. This is intentionally kept as an optional dependency.

Status: placeholder. The full TFDS pipeline has not been ported into this
packaging-first refactor yet.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class KubricData(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import tensorflow as _tf  # noqa: F401
            import tensorflow_datasets as _tfds  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "KubricData requires TensorFlow + tensorflow-datasets. "
                "Install them to use dataset_name=panning_movie."
            ) from exc

        raise NotImplementedError(
            "KubricData (panning_movie) loader is not yet implemented in this refactor."
        )


__all__ = ["KubricData"]


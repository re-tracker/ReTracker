"""Utilities to normalize dataset outputs for the unified training pipeline.

Historically different datasets in this codebase returned slightly different
schemas (tensor vs numpy, `visibles` vs `valids`, etc.). The unified datamodule
wraps datasets and calls :func:`unified_matching_tracking_dataset` to ensure a
consistent, training-friendly output format.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _to_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        # Use as_tensor to avoid an extra copy when possible.
        return torch.as_tensor(value)
    return value


def _squeeze_last_dim_if_one(x: torch.Tensor) -> torch.Tensor:
    if x.dim() >= 1 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


def unified_matching_tracking_dataset(raw_data: Any, config: Any | None = None) -> Dict[str, Any]:
    """Convert a dataset sample into the canonical unified schema.

    Expected (tracking) keys after normalization:
    - images: float tensor, shape [T, C, H, W]
    - trajs: float tensor, shape [T, N, 2] (x, y)
    - visibles: bool tensor, shape [T, N]
    - valids: bool tensor, shape [T, N] (defaults to visibles if missing)
    - occs: bool tensor, shape [T, N] (defaults to ~valids if missing)

    For matching datasets, some of these fields may be absent. In that case,
    downstream code must provide supervision (e.g. via depth-based warping).
    """
    # `config` is currently reserved for future use (per-dataset policies).
    _ = config

    if not isinstance(raw_data, dict):
        raise TypeError(f"Expected dataset sample to be dict-like, got: {type(raw_data)!r}")

    data: Dict[str, Any] = {k: _to_tensor(v) for k, v in raw_data.items()}

    # Unify naming: many datasets use `visibles` only.
    if "valids" not in data and "visibles" in data:
        data["valids"] = data["visibles"]

    # Convert vis/valid/occ tensors to bool and remove trailing singleton dims.
    for key in ("visibles", "valids", "occs"):
        if key not in data:
            continue
        v = data[key]
        if isinstance(v, torch.Tensor):
            v = _squeeze_last_dim_if_one(v)
            if v.dtype != torch.bool:
                v = v > 0
            data[key] = v

    # Default occs from valids if missing.
    if "occs" not in data and "valids" in data and isinstance(data["valids"], torch.Tensor):
        data["occs"] = ~data["valids"]

    # If we only have image0/image1 (matching datasets), build `images`.
    if "images" not in data and "image0" in data and "image1" in data:
        img0, img1 = data["image0"], data["image1"]
        if not isinstance(img0, torch.Tensor) or not isinstance(img1, torch.Tensor):
            raise TypeError("image0/image1 must be torch tensors after normalization")
        data["images"] = torch.stack([img0, img1], dim=1) if img0.dim() == 4 else torch.stack([img0, img1], dim=0)

    # Common fallbacks for metadata keys (used for logging).
    if "scene_name" not in data:
        if "scene_id" in data:
            data["scene_name"] = data["scene_id"]
        elif "images_path" in data:
            data["scene_name"] = data["images_path"]
        else:
            data["scene_name"] = "unknown"
    if "dataset_name" not in data:
        data["dataset_name"] = "unknown"

    return data


__all__ = ["unified_matching_tracking_dataset"]


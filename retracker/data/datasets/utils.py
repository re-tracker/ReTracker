"""Dataset utilities shared by evaluation code.

This module intentionally mirrors a small subset of CoTracker's dataset helpers
so that our evaluation pipeline (`retracker.evaluation.*`) can stay close to the
original TAP-Vid evaluation flow.

Keep this file dependency-light: torch + stdlib only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(eq=False)
class ReTrackerData:
    """Dataclass for storing video point tracking data.

    Note: Individual dataset items typically store tensors without a batch
    dimension (e.g. `video` is [T,C,H,W]). The dataloader `collate_fn` stacks
    them into batched tensors (e.g. [B,T,C,H,W]).
    """

    video: torch.Tensor  # (T, C, H, W) per item; collated to (B, T, C, H, W)
    trajectory: torch.Tensor  # (T, N, 2) per item; collated to (B, T, N, 2)
    visibility: torch.Tensor  # (T, N) per item; collated to (B, T, N)

    # Optional / dataset-specific fields.
    valid: Optional[torch.Tensor] = None  # (T, N)
    segmentation: Optional[torch.Tensor] = None  # (T, 1, H, W) or similar
    # During collation we keep this as a python list[str] for logging.
    seq_name: Optional[str | list[str]] = None
    query_points: Optional[torch.Tensor] = None  # TAP-Vid format: (N, 3) = (t, y, x)
    transforms: Optional[Dict[str, Any]] = None
    aug_video: Optional[torch.Tensor] = None


def collate_fn(batch: list[ReTrackerData]) -> ReTrackerData:
    """Collate function for evaluation dataloaders."""

    if len(batch) == 0:
        raise ValueError("Empty batch")

    video = torch.stack([b.video for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)

    query_points = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)

    segmentation = None
    if batch[0].segmentation is not None:
        segmentation = torch.stack([b.segmentation for b in batch], dim=0)

    valid = None
    if batch[0].valid is not None:
        valid = torch.stack([b.valid for b in batch], dim=0)

    # Keep seq_name as a python list for logging (matches existing evaluator code).
    seq_name = [b.seq_name for b in batch]

    return ReTrackerData(
        video=video,
        trajectory=trajectory,
        visibility=visibility,
        valid=valid,
        segmentation=segmentation,
        seq_name=seq_name,  # type: ignore[arg-type]
        query_points=query_points,
    )


def try_to_cuda(t: Any) -> Any:
    """Try to move `t` to CUDA if it is a torch Tensor."""

    if isinstance(t, torch.Tensor):
        return t.cuda(non_blocking=True)
    return t


def dataclass_to_cuda_(obj: Any) -> Any:
    """Move all torch.Tensor fields of a dataclass to CUDA in-place."""

    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


__all__ = [
    "ReTrackerData",
    "collate_fn",
    "dataclass_to_cuda_",
]

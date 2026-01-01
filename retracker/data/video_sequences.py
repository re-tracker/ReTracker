"""Utilities for building frame index sequences for video tracking.

This module contains small helpers that are shared by:
- inference engines (pairwise tracking)
- model-level video evaluation helpers

They are *not* training-only, so they do not belong under `retracker.training.utils`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def construct_triplets_for_eval(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build (image0, image_m, image1) triplets for video evaluation/inference.

    Expected batch keys (subset):
    - images: Tensor with shape (B, F, C, H, W)
    - queries: Tensor with shape (B, N, 2) (x,y) for test
    """
    triplets: List[Dict[str, Any]] = []
    frame_count = batch["images"].shape[1]

    for idx in range(frame_count - 1):
        triplets.append(
            {
                "image0": batch["images"][:, 0],
                "image_m": batch["images"][:, max(idx, 0)],
                "image1": batch["images"][:, idx + 1],
                "queries": batch["queries"],
            }
        )
    return triplets


def build_pairs(q_frame_idx: Any, sequence_length: int, is_forward: bool = True) -> np.ndarray:
    """Build forward/backward index pairs used by the pairwise tracker.

    Args:
        q_frame_idx: Query frame index. Typically a scalar torch Tensor.
        sequence_length: Total number of frames in the sequence.
        is_forward: If True, build (q, q..T-1). If False, build (q, q-1..0).

    Returns:
        pairs_ids: ndarray of shape (K, 2) with rows [q_frame_idx, other_frame_idx].
    """
    if isinstance(q_frame_idx, torch.Tensor):
        q_frame_idx = q_frame_idx.cpu().numpy()

    begin, stop = (q_frame_idx, sequence_length) if is_forward else (q_frame_idx - 1, -1)
    pairs_ids = (
        np.array(
            np.meshgrid(np.arange(begin, stop, 1 if is_forward else -1), q_frame_idx)
        )
        .reshape(2, -1)
        .T[:, [1, 0]]
    )
    return pairs_ids


def build_triplets(pairs_ids: np.ndarray, is_forward: bool = True) -> np.ndarray:
    """Build frame triplets based on pairs.

    Examples:
        [[1,2], [1,3], ...] -> [[1,1(New),2], [1,2(New),3], ...]
        [[5,4], [5,3], ...] -> [[5,5(New),4], [5,4(New),3], ...]
    """
    mid_ids = pairs_ids[:, 1:] - 1 if is_forward else pairs_ids[:, 1:] + 1  # [K, 1]
    return np.concatenate([pairs_ids[:, :1], mid_ids, pairs_ids[:, 1:]], axis=-1)


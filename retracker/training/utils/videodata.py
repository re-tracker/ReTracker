"""Training-time helpers for turning sequence batches into pairs/triplets.

These helpers are used by the Lightning training stack to convert a batch of
video data into the dict format expected by the model forward.

They intentionally live under `retracker.training.utils` (not `retracker.utils`)
because they are not needed for inference-only usage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def construct_pairs(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct pairwise samples from a batched video sequence.

    Expected batch keys (subset):
    - images: Tensor with shape (B, F, C, H, W)
    - occs, trajs, visibles, valids: tensors with matching frame dimension.

    Returns a list of dicts, each representing (image0, image1) pairs.
    """
    pairs: List[Dict[str, Any]] = []
    frame_count = batch["images"].shape[1]

    # (0,0,1) to (0,6,7)
    for idx in range(1, frame_count):
        pairs.append(
            {
                "image0": batch["images"][:, 0],
                "image1": batch["images"][:, idx],
                "occs": batch["occs"][:, [0, idx]],
                "trajs": batch["trajs"][:, [0, idx]],
                "visibles": batch["visibles"][:, [0, idx]],
                "valids": batch["valids"][:, [0, idx]],
                "frame_idx": idx,
                "frame_num": frame_count,
                "is_first_frame": idx == 1,
                "is_last_frame": idx == frame_count - 1,
                "dataset_name": batch["dataset_name"],
                "scene_name": batch["scene_name"],
            }
        )
    return pairs


def construct_triplets(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct triplet samples from an (unbatched) video sequence.

    Note: This helper preserves the legacy behavior where the frame dimension
    is expected at dim=0 (i.e. (F, C, H, W) rather than (B, F, ...)).
    """
    triplets: List[Dict[str, Any]] = []
    frame_count = batch["images"].shape[0]

    # (0,0,1) to (0,6,7)
    for idx in range(frame_count - 1):
        triplets.append(
            {
                "scale": batch["scale"],
                "image0": batch["images"][0],
                "image_m": batch["images"][max(idx, 0)],
                "image1": batch["images"][idx + 1],
                "occs": batch["occs"][[0, idx, idx + 1]],
                "trajs": batch["trajs"][[0, idx, idx + 1]],
                "visibles": batch["visibles"][[0, idx, idx + 1]],
                "valids": batch["valids"][[0, idx, idx + 1]],
                "frame_idx": idx + 1,
                "frame_num": frame_count,
                "is_first_frame": idx + 1 == 1,
                "is_last_frame": idx + 1 == frame_count - 1,
                "dataset_name": batch["dataset_name"],
                "scene_name": batch["scene_name"],
            }
        )
    return triplets


def construct_triplets_in_one_forward(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct triplets from a batched video sequence (B, F, ...)."""
    triplets: List[Dict[str, Any]] = []
    frame_count = batch["images"].shape[1]

    # (0,0,1) to (0,6,7)
    for idx in range(frame_count - 1):
        triplets.append(
            {
                "scale": batch["scale"],
                "image0": batch["images"][:, 0],
                "image_m": batch["images"][:, max(idx, 0)],
                "image1": batch["images"][:, idx + 1],
                "occs": batch["occs"][:, [0, idx, idx + 1]],
                "trajs": batch["trajs"][:, [0, idx, idx + 1]],
                "visibles": batch["visibles"][:, [0, idx, idx + 1]],
                "valids": batch["valids"][:, [0, idx, idx + 1]],
                # Preserve legacy tensor-valued flags.
                "frame_idx": torch.Tensor([idx + 1]),
                "frame_num": torch.Tensor([frame_count]),
                "is_first_frame": torch.Tensor([idx + 1]) == 1,
                "is_last_frame": torch.Tensor([idx + 1]) == frame_count - 1,
                "dataset_name": batch["dataset_name"],
                "scene_name": batch["scene_name"],
            }
        )
    return triplets


def cache_video_data(cache_list: List[Dict[str, Any]], frame_data: Dict[str, Any]) -> None:
    """Append a single-frame slice into a rolling cache used by some trainers."""
    extracted_data = {
        "images": frame_data["image1"],
        # 0,1,2
        "occs": frame_data["occs"][:, -1],
        "trajs": frame_data["trajs"][:, -1],
        "visibles": frame_data["visibles"][:, -1],
        "valids": frame_data["valids"][:, -1],
    }
    cache_list.append(extracted_data)

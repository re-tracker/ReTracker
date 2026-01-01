"""Synthetic tracking dataset for smoke tests and development.

This dataset is intentionally dependency-light (torch only). It can be used to
validate the training pipeline without downloading any real datasets.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class DummyTrackingDataset(Dataset):
    """A tiny synthetic tracking dataset producing video + point trajectories."""

    def __init__(
        self,
        length: int = 32,
        sequence_length: int = 8,
        num_points: int = 128,
        image_size: Tuple[int, int] = (64, 64),  # (H, W)
        channels: int = 3,
        seed: int = 0,
    ) -> None:
        self.length = int(length)
        self.sequence_length = int(sequence_length)
        self.num_points = int(num_points)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.channels = int(channels)
        self.seed = int(seed)

        if self.length <= 0:
            raise ValueError("length must be > 0")
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")
        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        # Per-sample deterministic randomness for reproducibility.
        gen = torch.Generator()
        gen.manual_seed(self.seed + int(index))

        t = self.sequence_length
        n = self.num_points
        h, w = self.image_size

        images = torch.rand((t, self.channels, h, w), generator=gen, dtype=torch.float32)

        # Random walk trajectories in pixel coordinates (x, y).
        xy0 = torch.stack(
            [
                torch.rand((n,), generator=gen) * (w - 1),
                torch.rand((n,), generator=gen) * (h - 1),
            ],
            dim=-1,
        )  # [N, 2]
        steps = torch.randn((t, n, 2), generator=gen) * 0.5
        trajs = xy0[None].repeat(t, 1, 1) + torch.cumsum(steps, dim=0)
        trajs[..., 0] = trajs[..., 0].clamp_(0, w - 1)
        trajs[..., 1] = trajs[..., 1].clamp_(0, h - 1)

        valids = torch.ones((t, n), dtype=torch.bool)
        visibles = valids.clone()
        occs = ~valids

        # Minimal coarse supervision signals used by the stage4 training loop.
        # We treat each pixel as a class id (y * W + x), with the last class
        # (H*W) reserved for "no match / occluded".
        trajs_rounded = trajs.round().to(torch.long)
        trajs_rounded[..., 0] = trajs_rounded[..., 0].clamp_(0, w - 1)
        trajs_rounded[..., 1] = trajs_rounded[..., 1].clamp_(0, h - 1)
        gt_cls_ids_S = trajs_rounded[..., 1] * w + trajs_rounded[..., 0]  # [T, N]
        gt_cls_ids_vis_S = visibles.clone()
        gt_cls_ids_S = gt_cls_ids_S.clone()
        gt_cls_ids_S[~gt_cls_ids_vis_S] = h * w

        return {
            "images": images,
            "trajs": trajs,
            "visibles": visibles,
            "valids": valids,
            "occs": occs,
            "gt_cls_ids_S": gt_cls_ids_S,
            "gt_cls_ids_vis_S": gt_cls_ids_vis_S,
            "dataset_name": "dummy",
            "scene_name": f"dummy_{index:05d}",
        }


__all__ = ["DummyTrackingDataset"]

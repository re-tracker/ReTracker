"""TAP-Vid dataset loader used by evaluation (`retracker.evaluation.cli`).

This implementation is adapted from the official CoTracker repository:
https://github.com/facebookresearch/co-tracker

We keep the interface compatible with the original `TapVidDataset` so that the
legacy wrapper scripts in `scripts/evaluation/` keep working.
"""

from __future__ import annotations

import glob
import os
import pickle
import random
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch

from retracker.data.datasets.utils import ReTrackerData
from retracker.utils.rich_utils import CONSOLE


DatasetElement = Mapping[str, Mapping[str, np.ndarray | str]]


def _resolve_existing_pickle_path(path: str, *, max_depth: int = 8) -> str:
    """Resolve TAP-Vid pickle paths that may be stored as broken symlinks.

    Some setups store `*.pkl` as a symlink, while the actual payload lives as
    `*.pkl.full` next to the symlink target. We follow a short symlink chain
    and check the `.full` variant at each step.
    """

    p = Path(path)
    for _ in range(int(max_depth)):
        if p.exists():
            return str(p)

        full = Path(str(p) + ".full")
        if full.exists():
            return str(full)

        if not p.is_symlink():
            break

        try:
            target = os.readlink(p)
        except OSError:
            break

        p = Path(os.path.normpath(os.path.join(str(p.parent), target)))

    return str(path)


def _decode_jpeg_bytes(frame: bytes) -> np.ndarray:
    """Decode one JPEG frame stored as bytes into an RGB uint8 array."""

    arr = np.frombuffer(frame, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def resize_video(video: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Resize a video to (H, W) = output_size."""

    out_h, out_w = int(output_size[0]), int(output_size[1])
    frames = [cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA) for f in video]
    return np.stack(frames, axis=0)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Use the first visible point in each track as the query (TAP-Vid "first")."""

    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x], dtype=np.float32))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Sample queries every `query_stride` frames (TAP-Vid "strided")."""

    tracks = []
    occs = []
    queries = []
    trackgroups = []
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], int(query_stride)):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        ).astype(np.float32)
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TapVidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_type: str = "davis",
        resize_to: tuple[int, int] | None = (256, 256),
        queried_first: bool = True,
        fast_eval: bool = False,
    ) -> None:
        local_random = random.Random(42)
        self.fast_eval = bool(fast_eval)
        self.dataset_type = str(dataset_type)
        self.resize_to = resize_to
        self.queried_first = bool(queried_first)

        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset: list[dict] = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    # Trusted source: internal cache file
                    data = pickle.load(f)
                points_dataset.extend(list(data))
            if self.fast_eval and len(points_dataset) > 50:
                points_dataset = local_random.sample(points_dataset, 50)
            self.points_dataset = points_dataset
            self.video_names = list(range(len(self.points_dataset)))

        elif self.dataset_type == "robotap":
            all_paths = glob.glob(os.path.join(data_root, "robotap_split*.pkl"))
            points_dataset: dict | None = None
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    # Trusted source: internal cache file
                    data = pickle.load(f)
                if points_dataset is None:
                    points_dataset = dict(data)
                else:
                    points_dataset.update(data)
            if points_dataset is None:
                raise FileNotFoundError(f"No robotap_split*.pkl found under: {data_root}")
            if self.fast_eval and len(points_dataset) > 50:
                keys = local_random.sample(sorted(points_dataset.keys()), 50)
                points_dataset = {k: points_dataset[k] for k in keys}
            self.points_dataset = points_dataset
            self.video_names = list(self.points_dataset.keys())

        else:
            data_root = _resolve_existing_pickle_path(data_root)
            with open(data_root, "rb") as f:
                # Trusted source: internal cache file
                self.points_dataset = pickle.load(f)
            if self.dataset_type == "davis":
                self.video_names = list(self.points_dataset.keys())
            elif self.dataset_type == "stacking":
                self.video_names = list(range(len(self.points_dataset)))
            elif self.dataset_type == "retrack":
                if isinstance(self.points_dataset, dict):
                    self.video_names = list(self.points_dataset.keys())
                else:
                    self.video_names = list(range(len(self.points_dataset)))
            else:
                raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        CONSOLE.print(f"[dim]found {len(self.points_dataset)} unique videos in {data_root}[/dim]")

    def __len__(self) -> int:
        return len(self.points_dataset)

    def __getitem__(self, index: int) -> ReTrackerData:
        video_name = self.video_names[index]

        video = self.points_dataset[video_name]
        frames = video["video"]

        if self.fast_eval and hasattr(frames, "shape") and frames.shape[0] > 300:
            return self.__getitem__((index + 1) % self.__len__())

        if len(frames) == 0:
            raise ValueError(f"Empty TAP-Vid video at index={index}")

        # TAP-Vid may store frames as JPEG bytes instead of ndarrays.
        if isinstance(frames[0], (bytes, bytearray)):
            frames = np.array([_decode_jpeg_bytes(f) for f in frames], dtype=np.uint8)
        else:
            frames = np.asarray(frames)

        target_points = np.asarray(video["points"], dtype=np.float32)  # [N, T, 2] in [0, 1]

        if self.resize_to is not None:
            frames = resize_video(frames, self.resize_to)
            # 1 should map to (size-1) in pixel space.
            target_points *= np.array([self.resize_to[1] - 1, self.resize_to[0] - 1], dtype=np.float32)
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1], dtype=np.float32)

        target_occ = np.asarray(video["occluded"])

        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)

        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        # [T, N, 2] in pixel coordinates.
        trajs = torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()

        # [T, 3, H, W] in uint8 (0..255).
        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        if rgbs.dtype != torch.uint8:
            rgbs = rgbs.to(torch.uint8)

        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(1, 0)
        query_points = torch.from_numpy(converted["query_points"])[0]  # [N, 3] (t, y, x)

        return ReTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            seq_name=str(video_name),
            query_points=query_points,
        )


__all__ = ["TapVidDataset"]

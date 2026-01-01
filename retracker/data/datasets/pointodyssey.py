"""PointOdyssey dataset loader (tracking).

This implementation follows the "video + point trajectories" schema used by the
ReTracker training pipeline.

Expected on-disk structure (one scene folder):
  <root>/<split>/<scene_name>/
    rgbs/rgb_00000.jpg
    rgbs/rgb_00001.jpg
    ...
    anno.npz  (keys: trajs_2d, valids, visibs, ...)

Notes:
- The dataset provides grayscale images by default (C=1). The model backbones
  in this repo handle C=1 by repeating channels as needed.
- Coordinates are returned in the resized/cropped resolution space.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _Scene:
    name: str
    rgb_dir: Path
    anno_path: Path


class PointOdysseyDataset(Dataset):
    def __init__(
        self,
        dataset_location: str,
        dset: str = "train",
        mode: str = "videos",
        use_augs: bool = False,
        N: int = 400,
        S: int = 24,
        crop_size: Tuple[int, int] = (256, 256),  # (H, W)
        subset: str = "all",
        global_rank: int = 0,
        world_size: int = 1,
        real_batch_size: int = 1,  # kept for API compatibility (unused)
        videos_per_scene: int = 20,
        anno_cache_size: int = 0,
        seed: int = 42,
    ) -> None:
        if mode != "videos":
            raise ValueError("PointOdysseyDataset currently supports mode='videos' only")

        self.dataset_location = str(dataset_location)
        self.dset = str(dset)
        self.mode = str(mode)
        self.use_augs = bool(use_augs)
        self.N = int(N)
        self.S = int(S)
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.subset = str(subset)
        self.global_rank = int(global_rank)
        self.world_size = int(world_size)
        self.real_batch_size = int(real_batch_size)
        self.videos_per_scene = int(videos_per_scene)
        self.anno_cache_size = int(anno_cache_size)
        self.seed = int(seed)

        if self.N <= 0:
            raise ValueError("N must be > 0")
        if self.S <= 1:
            raise ValueError("S must be > 1")
        if self.videos_per_scene <= 0:
            raise ValueError("videos_per_scene must be > 0")

        self._scenes: List[_Scene] = self._discover_scenes()
        if not self._scenes:
            raise FileNotFoundError(
                f"No PointOdyssey scenes found under: {Path(self.dataset_location) / self.dset}"
            )

        # Optional LRU cache for per-scene anno.npz loads. This avoids repeatedly
        # mmap'ing/decompressing the same scene metadata when `videos_per_scene > 1`.
        self._anno_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _discover_scenes(self) -> List[_Scene]:
        root = Path(self.dataset_location) / self.dset
        if not root.exists():
            # Some local copies only provide a "train" split. For convenience, fall back
            # to train when val/test are missing.
            if self.dset.lower() in {"val", "valid", "validation", "test"}:
                train_root = Path(self.dataset_location) / "train"
                if train_root.exists():
                    root = train_root
            if not root.exists():
                raise FileNotFoundError(f"PointOdyssey split directory not found: {root}")

        scenes: List[_Scene] = []
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rgb_dir = scene_dir / "rgbs"
            anno_path = scene_dir / "anno.npz"
            if rgb_dir.is_dir() and anno_path.is_file():
                scenes.append(_Scene(scene_dir.name, rgb_dir, anno_path))

        # Optional quick subset for local smoke tests.
        if self.subset.lower() == "sample":
            scenes = scenes[: max(1, min(2, len(scenes)))]

        # Deterministic split across ranks (dataset is not replicated).
        if self.world_size > 1:
            scenes = scenes[self.global_rank :: self.world_size]

        return scenes

    def __len__(self) -> int:
        return len(self._scenes) * self.videos_per_scene

    def _rng(self, index: int) -> np.random.Generator:
        # Per-sample deterministic RNG; allows reproducible sampling across workers.
        return np.random.default_rng(self.seed + int(index))

    def _sample_frame_indices(self, video_length: int, rng: np.random.Generator) -> np.ndarray:
        # Sample a sequence with a random stride (1..3), similar to classic training code.
        stride = int(rng.integers(1, 4))
        max_start = video_length - (self.S - 1) * stride
        if max_start <= 0:
            # Fall back to a contiguous clip from the start.
            return np.arange(min(self.S, video_length))
        start = int(rng.integers(0, max_start))
        return np.arange(start, start + self.S * stride, stride)[: self.S]

    def _load_images(self, rgb_dir: Path, frame_ids: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load grayscale frames and resize to crop_size. Return (images, scale).
        h_out, w_out = self.crop_size

        first_path = rgb_dir / f"rgb_{int(frame_ids[0]):05d}.jpg"
        first = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise FileNotFoundError(f"Failed to read frame: {first_path}")
        h_in, w_in = first.shape[:2]
        scale = torch.tensor([w_in / w_out, h_in / h_out], dtype=torch.float32)

        images = torch.empty((self.S, 1, h_out, w_out), dtype=torch.float32)

        for i, fid in enumerate(frame_ids):
            img_path = rgb_dir / f"rgb_{int(fid):05d}.jpg"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read frame: {img_path}")
            img = cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_AREA)
            images[i, 0] = torch.from_numpy(img).float() / 255.0

        return images, scale

    def _load_anno(self, scene: _Scene) -> Dict[str, np.ndarray]:
        """Load (and optionally cache) a scene's anno.npz contents."""
        cache_key = scene.name
        if self.anno_cache_size > 0:
            cached = self._anno_cache.get(cache_key)
            if cached is not None:
                # Mark as recently used.
                self._anno_cache.move_to_end(cache_key)
                return cached

        # Use a context manager to ensure the NPZ file handle is closed.
        with np.load(scene.anno_path, allow_pickle=True) as anno:
            payload = {k: anno[k] for k in anno.files}

        if self.anno_cache_size > 0:
            self._anno_cache[cache_key] = payload
            self._anno_cache.move_to_end(cache_key)
            while len(self._anno_cache) > self.anno_cache_size:
                self._anno_cache.popitem(last=False)

        return payload

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Map many indices to the same underlying scene, sampling different clips.
        scene = self._scenes[int(index) // self.videos_per_scene]
        rng = self._rng(index)

        anno = self._load_anno(scene)
        trajs_2d = anno["trajs_2d"]  # [T_total, N_total, 2] (x, y)
        visibs = anno.get("visibs", None)
        valids = anno.get("valids", None)

        if visibs is None:
            # Fall back: visible if valid.
            visibs = valids
        if valids is None:
            raise KeyError(f"Missing required key 'valids' in {scene.anno_path}")

        video_length, n_total = trajs_2d.shape[0], trajs_2d.shape[1]

        # Try a few times to get enough valid points at the first frame.
        for _ in range(10):
            frame_ids = self._sample_frame_indices(video_length, rng)
            first = int(frame_ids[0])
            mask0 = valids[first] & visibs[first]
            cand = np.flatnonzero(mask0)
            if cand.size >= self.N:
                point_ids = rng.choice(cand, size=self.N, replace=False)
                break
        else:
            # Hard failure: return a dummy sample to keep training running.
            h_out, w_out = self.crop_size
            return {
                "images": torch.zeros((self.S, 1, h_out, w_out), dtype=torch.float32),
                "trajs": torch.zeros((self.S, self.N, 2), dtype=torch.float32),
                "visibles": torch.zeros((self.S, self.N), dtype=torch.bool),
                "valids": torch.zeros((self.S, self.N), dtype=torch.bool),
                "occs": torch.ones((self.S, self.N), dtype=torch.bool),
                "dataset_name": "pointodyssey",
                "scene_name": scene.name,
                "images_path": str(scene.rgb_dir),
                "scale": torch.tensor([1.0, 1.0], dtype=torch.float32),
            }

        images, scale = self._load_images(scene.rgb_dir, frame_ids)

        trajs = trajs_2d[frame_ids][:, point_ids].astype(np.float32)  # [S, N, 2]
        vis = visibs[frame_ids][:, point_ids].astype(np.bool_)  # [S, N]
        val = valids[frame_ids][:, point_ids].astype(np.bool_)  # [S, N]

        trajs_t = torch.from_numpy(trajs) / scale  # to resized coords
        vis_t = torch.from_numpy(vis)
        val_t = torch.from_numpy(val) & vis_t

        # Recompute bounds in resized space to be safe.
        h_out, w_out = self.crop_size
        inb = (trajs_t[..., 0] >= 0) & (trajs_t[..., 0] < w_out) & (trajs_t[..., 1] >= 0) & (trajs_t[..., 1] < h_out)
        val_t = val_t & inb

        return {
            "scale": scale,
            "images_path": str(scene.rgb_dir),
            "images": images,
            "trajs": trajs_t,
            "visibles": vis_t,
            "valids": val_t,
            "occs": ~val_t,
            "dataset_name": "pointodyssey",
            "scene_name": scene.name,
        }


__all__ = ["PointOdysseyDataset"]

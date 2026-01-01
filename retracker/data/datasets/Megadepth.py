"""MegaDepth dataset wrapper used for image matching training.

This file is a lightly adapted version of the project's historical MegaDepth
loader. It returns per-pair data (image0/image1, depth, poses, intrinsics, etc.)
used by matching supervision utilities.
"""

from __future__ import annotations

import os.path as osp
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from retracker.data.datasets.io import read_megadepth_depth, read_megadepth_gray
from retracker.utils.rich_utils import CONSOLE


class MegaDepthDataset(Dataset):
    """Manage one MegaDepth scene (.npz with pair_infos)."""

    def __init__(
        self,
        root_dir: str,
        npz_path: str,
        mode: str = "train",
        min_overlap_score: float = 0.4,
        img_resize: Optional[list[int]] = None,
        df: Optional[int] = None,
        img_padding: bool = False,
        depth_padding: bool = False,
        augment_fn=None,
        testNpairs: int = 300,
        fp16: bool = False,
        fix_bias: bool = False,
        read_depth: bool = False,
        sample_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = osp.basename(npz_path).split(".")[0]
        self.sample_ratio = float(sample_ratio)

        if mode == "test" and min_overlap_score > 0:
            CONSOLE.print("[yellow]min_overlap_score!=0 in test mode; forcing to 0[/yellow]")
            min_overlap_score = -3.0

        # MegaDepth indices in this repo may be stored either as a real `.npz`
        # (NpzFile with `.files`) or as a pickled python dict (historical format
        # sometimes saved with `.npz` extension). Support both.
        scene_info_obj = np.load(npz_path, allow_pickle=True)
        if isinstance(scene_info_obj, dict):
            scene_info = scene_info_obj
        else:
            # NpzFile-like object
            scene_info = {k: scene_info_obj[k] for k in scene_info_obj.files}
            try:
                scene_info_obj.close()
            except Exception:
                pass

        if mode == "test":
            pair_infos = scene_info["pair_infos"][:testNpairs].copy()
        else:
            pair_infos = scene_info["pair_infos"].copy()

        # Keep the rest of scene_info for intrinsics/poses/etc.
        self.scene_info = {k: scene_info[k] for k in scene_info.keys() if k != "pair_infos"}

        self.pair_infos = [pi for pi in pair_infos if pi[1] > min_overlap_score]

        if mode == "train":
            if img_resize is None or not img_padding or not depth_padding:
                raise ValueError("Training MegaDepth requires img_resize + img_padding + depth_padding")

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None

        self.dataset_name = self.scene_info.get("dataset_name", "megadepth")
        self.load_origin_rgb = bool(kwargs.get("load_origin_rgb", False))
        self.read_gray = bool(kwargs.get("read_gray", True))

        self.augment_fn = augment_fn if mode == "train" else None
        self.coarse_scale = kwargs.get("coarse_scale", 0.125)
        self.dino_scale = kwargs.get("dino_scale", 1 / 16)

        self.fp16 = bool(fp16)
        self.fix_bias = bool(fix_bias)
        if self.fix_bias:
            self.df = 1
        self.read_depth = bool(read_depth)

    def __len__(self) -> int:
        return len(self.pair_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_info = self.pair_infos[idx]
        if len(pair_info) == 3:
            (idx0, idx1), overlap_score, _central_matches = pair_info
            _ = overlap_score
        elif len(pair_info) == 2:
            (idx0, idx1), overlap_score = pair_info
            _ = overlap_score
        else:
            raise NotImplementedError

        img_name0 = osp.join(self.root_dir, self.dataset_name, self.scene_info["image_paths"][idx0])
        img_name1 = osp.join(self.root_dir, self.dataset_name, self.scene_info["image_paths"][idx1])

        # Some MegaDepth variants don't preserve undistorted images; fall back to raw image folder.
        if not osp.exists(img_name0):
            base = osp.basename(img_name0)
            base_path = osp.join(self.root_dir, self.dataset_name, self.scene_info["depth_paths"][idx0]).split("/depths/")[0]
            img_name0 = osp.join(base_path, "imgs", base)
        if not osp.exists(img_name1):
            base = osp.basename(img_name1)
            base_path = osp.join(self.root_dir, self.dataset_name, self.scene_info["depth_paths"][idx1]).split("/depths/")[0]
            img_name1 = osp.join(base_path, "imgs", base)

        image0, mask0, scale0, origin_img_size0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray
        )
        image1, mask1, scale1, origin_img_size1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray
        )

        if self.mode in {"train", "val", "test"}:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.dataset_name, self.scene_info["depth_paths"][idx0]),
                pad_to=self.depth_max_size,
            )
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.dataset_name, self.scene_info["depth_paths"][idx1]),
                pad_to=self.depth_max_size,
            )
        else:
            depth0 = depth1 = torch.tensor([])

        K0 = torch.tensor(self.scene_info["intrinsics"][idx0].copy(), dtype=torch.float32).reshape(3, 3)
        K1 = torch.tensor(self.scene_info["intrinsics"][idx1].copy(), dtype=torch.float32).reshape(3, 3)

        T0 = self.scene_info["poses"][idx0]
        T1 = self.scene_info["poses"][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float32)[:4, :4]
        T_1to0 = T_0to1.inverse()

        if self.fp16:
            image0 = image0.half()
            image1 = image1.half()
            depth0 = depth0.half()
            depth1 = depth1.half()
            scale0 = scale0.half()
            scale1 = scale1.half()

        data: Dict[str, Any] = {
            "image0": image0,
            "image1": image1,
            "depth0": depth0,
            "depth1": depth1,
            "T_0to1": T_0to1,
            "T_1to0": T_1to0,
            "K0": K0,
            "K1": K1,
            "homography": torch.zeros((3, 3), dtype=torch.float32),
            "norm_pixel_mat": torch.zeros((3, 3), dtype=torch.float32),
            "homo_sample_normed": torch.zeros((3, 3), dtype=torch.float32),
            "origin_img_size0": origin_img_size0,
            "origin_img_size1": origin_img_size1,
            "scale0": scale0,
            "scale1": scale1,
            "dataset_name": "MegaDepth",
            "scene_id": self.scene_id,
            "pair_id": idx,
            "pair_names": (img_name0, img_name1),
            "rel_pair_names": (
                self.scene_info["image_paths"][idx0],
                self.scene_info["image_paths"][idx1],
            ),
        }

        # Downsample masks for coarse supervision.
        if mask0 is not None:
            if self.coarse_scale:
                if self.fix_bias:
                    size = ((image0.shape[1] - 1) // 8 + 1, (image0.shape[2] - 1) // 8 + 1)
                    ts0, ts1 = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(), size=size, mode="nearest")[0].bool()
                else:
                    ts0, ts1 = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(), scale_factor=self.coarse_scale, mode="nearest")[0].bool()
                data.update({"mask0": ts0, "mask1": ts1})

            if self.dino_scale:
                if self.fix_bias:
                    size = ((image0.shape[1] - 1) // 16 + 1, (image0.shape[2] - 1) // 16 + 1)
                    ts0, ts1 = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(), size=size, mode="nearest")[0].bool()
                else:
                    ts0, ts1 = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(), scale_factor=self.dino_scale, mode="nearest")[0].bool()
                data.update({"mask0_dino": ts0, "mask1_dino": ts1})

        if self.load_origin_rgb:
            from PIL import Image

            data.update(
                {
                    "image0_rgb_origin": torch.from_numpy(np.array(Image.open(img_name0).convert("RGB"))).permute(2, 0, 1) / 255.0,
                    "image1_rgb_origin": torch.from_numpy(np.array(Image.open(img_name1).convert("RGB"))).permute(2, 0, 1) / 255.0,
                }
            )

        return data


__all__ = ["MegaDepthDataset"]

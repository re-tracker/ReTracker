from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.eval_benchmark.methods.base import BaseMethod, MethodOutput


class ReTrackerMethod(BaseMethod):
    name = "retracker"

    def __init__(
        self,
        ckpt_path: str | Path,
        interp_shape: tuple[int, int] = (512, 512),
        engine: Any | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.interp_shape = (int(interp_shape[0]), int(interp_shape[1]))
        self._engine = engine
        self._device: str | None = None

    def load(self, device: str) -> None:
        # In unit tests we inject a dummy engine.
        if self._engine is not None:
            self._device = device
            return

        from retracker.inference.engine import ReTrackerEngine

        self._engine = ReTrackerEngine(ckpt_path=str(self.ckpt_path), interp_shape=self.interp_shape)
        self._engine.to(torch.device(device))
        self._device = device

    def predict(self, frames_uint8: np.ndarray, query_points_tyx: np.ndarray) -> MethodOutput:
        if self._engine is None:
            # Default to CUDA when available, otherwise CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.load(device)

        assert self._engine is not None

        frames = np.asarray(frames_uint8)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames_uint8 must be (T,H,W,3), got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        q = np.asarray(query_points_tyx, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"query_points_tyx must be (N,3), got {q.shape}")

        # video: (1,T,3,H,W) uint8
        video_t = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)

        # queries: (1,N,3) float32 in engine convention [t,x,y]
        queries_txy = q[:, [0, 2, 1]]
        queries_t = torch.from_numpy(queries_txy).unsqueeze(0)

        start = time.time()
        with torch.no_grad():
            traj_e, vis_e = self._engine.video_forward(video_t, queries_t)
        end = time.time()

        # traj_e: (1,T,N,2) -> (N,T,2)
        pred_tracks_xy = traj_e[0].permute(1, 0, 2).cpu().numpy().astype(np.float32, copy=False)

        # vis_e: (1,T,N) visible bool -> occluded bool in (N,T)
        vis_nt = vis_e[0].permute(1, 0)
        pred_occluded = (~vis_nt).cpu().numpy().astype(bool, copy=False)

        return MethodOutput(pred_tracks_xy=pred_tracks_xy, pred_occluded=pred_occluded, runtime_sec=float(end - start))

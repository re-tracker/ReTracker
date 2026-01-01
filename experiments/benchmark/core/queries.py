from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Queries


@dataclass(frozen=True)
class GridQueriesConfig:
    grid_size: int
    height: int
    width: int
    t: int = 0


def get_points_on_a_grid(size: int, height: int, width: int) -> np.ndarray:
    """
    Match the grid used by Track-On2 and ReTracker demo:
      margin = width / 64
      ys: linspace(margin, height-margin, size)
      xs: linspace(margin, width-margin, size)
    Row-major order: y outer, x inner.

    Returns:
      (N, 2) array of (x, y) float32.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be > 0, got {height}x{width}")

    if size == 1:
        return np.asarray([[width / 2.0, height / 2.0]], dtype=np.float32)

    margin = float(width) / 64.0
    ys = np.linspace(margin, float(height) - margin, size, dtype=np.float32)
    xs = np.linspace(margin, float(width) - margin, size, dtype=np.float32)

    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)  # (size*size, 2) (x,y)
    return pts.astype(np.float32, copy=False)


def make_grid_queries(cfg: GridQueriesConfig) -> Queries:
    pts_xy = get_points_on_a_grid(cfg.grid_size, cfg.height, cfg.width)  # (N,2)
    t = np.full((pts_xy.shape[0], 1), float(cfg.t), dtype=np.float32)
    txy = np.concatenate([t, pts_xy], axis=1)  # (N,3)
    return Queries(txy=txy)


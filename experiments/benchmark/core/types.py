from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Queries:
    """
    Query points in (t, x, y) pixel coordinates at the visualization resolution.

    Shape: (N, 3), dtype float32.
    """

    txy: np.ndarray  # (N, 3) float32

    def __post_init__(self) -> None:
        txy = np.asarray(self.txy)
        if txy.ndim != 2 or txy.shape[1] != 3:
            raise ValueError(f"Queries must be (N,3), got {txy.shape}")
        if txy.dtype != np.float32:
            object.__setattr__(self, "txy", txy.astype(np.float32, copy=False))

    @property
    def n(self) -> int:
        return int(self.txy.shape[0])

    @staticmethod
    def load_txt(path: str | Path) -> "Queries":
        rows: list[tuple[float, float, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Bad queries line (expected 3 columns): {raw.rstrip()}")
                t, x, y = (float(parts[0]), float(parts[1]), float(parts[2]))
                rows.append((t, x, y))
        if not rows:
            raise ValueError(f"No queries found in: {path}")
        return Queries(txy=np.asarray(rows, dtype=np.float32))

    def save_txt(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("# format: t x y\n")
            f.write(f"# n={self.n}\n")
            for t, x, y in self.txy.tolist():
                f.write(f"{t:.6f} {x:.6f} {y:.6f}\n")


@dataclass(frozen=True)
class TrackSet:
    """
    Tracks and visibility masks at the visualization resolution.

    tracks_xy: (T, N, 2) float32, (x, y) pixels
    visibles:  (T, N) bool
    """

    tracks_xy: np.ndarray
    visibles: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.tracks_xy)
        vis = np.asarray(self.visibles)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"tracks_xy must be (T,N,2), got {xy.shape}")
        if vis.ndim != 2:
            raise ValueError(f"visibles must be (T,N), got {vis.shape}")
        if xy.shape[0] != vis.shape[0] or xy.shape[1] != vis.shape[1]:
            raise ValueError(f"tracks/visibles shape mismatch: {xy.shape} vs {vis.shape}")

        if xy.dtype != np.float32:
            xy = xy.astype(np.float32, copy=False)
            object.__setattr__(self, "tracks_xy", xy)
        if vis.dtype != np.bool_:
            vis = vis.astype(bool, copy=False)
            object.__setattr__(self, "visibles", vis)

    @property
    def t(self) -> int:
        return int(self.tracks_xy.shape[0])

    @property
    def n(self) -> int:
        return int(self.tracks_xy.shape[1])


@dataclass
class TrackingResult:
    """
    Standardized output for any tracker.
    """

    method: str
    video_path: str
    resized_hw: tuple[int, int]  # (H, W)
    queries: Queries
    tracks: TrackSet
    runtime_sec: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": np.asarray([self.method]),
            "video_path": np.asarray([self.video_path]),
            "resized_hw": np.asarray(self.resized_hw, dtype=np.int32),
            "queries": self.queries.txy.astype(np.float32, copy=False),
            "tracks": self.tracks.tracks_xy.astype(np.float32, copy=False),
            "visibles": self.tracks.visibles.astype(bool, copy=False),
            "runtime_sec": np.asarray([self.runtime_sec if self.runtime_sec is not None else -1.0], dtype=np.float32),
            "meta_json": np.asarray([json.dumps(self.meta, ensure_ascii=True)]),
        }
        np.savez_compressed(out, **payload)

    @staticmethod
    def load_npz(path: str | Path) -> "TrackingResult":
        p = Path(path)
        with np.load(p, allow_pickle=False) as z:
            method = str(z["method"][0])
            video_path = str(z["video_path"][0])
            resized_hw_arr = np.asarray(z["resized_hw"], dtype=np.int32).reshape(-1)
            resized_hw = (int(resized_hw_arr[0]), int(resized_hw_arr[1]))

            queries = Queries(txy=np.asarray(z["queries"], dtype=np.float32))
            tracks = TrackSet(
                tracks_xy=np.asarray(z["tracks"], dtype=np.float32),
                visibles=np.asarray(z["visibles"]).astype(bool),
            )

            runtime_sec = float(np.asarray(z["runtime_sec"], dtype=np.float32).reshape(-1)[0])
            if runtime_sec < 0:
                runtime_sec = None

            meta_json = str(z["meta_json"][0]) if "meta_json" in z.files else "{}"
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {"_meta_json_parse_error": True, "_raw": meta_json}

        return TrackingResult(
            method=method,
            video_path=video_path,
            resized_hw=resized_hw,
            queries=queries,
            tracks=tracks,
            runtime_sec=runtime_sec,
            meta=meta,
        )


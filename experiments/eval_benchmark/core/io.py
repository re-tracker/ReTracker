from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PredictionCache:
    pred_tracks_xy: np.ndarray  # (N,T,2) float32
    pred_occluded: np.ndarray  # (N,T) bool
    runtime_sec: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def save_prediction_npz(
    path: str | Path,
    *,
    pred_tracks_xy: np.ndarray,
    pred_occluded: np.ndarray,
    runtime_sec: float | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tracks = np.asarray(pred_tracks_xy, dtype=np.float32)
    occ = np.asarray(pred_occluded).astype(bool, copy=False)

    payload = {
        "pred_tracks_xy": tracks,
        "pred_occluded": occ,
        "runtime_sec": np.asarray([runtime_sec if runtime_sec is not None else -1.0], dtype=np.float64),
        "meta_json": np.asarray([json.dumps(meta or {}, ensure_ascii=True)]),
    }
    np.savez_compressed(out, **payload)


def load_prediction_npz(path: str | Path) -> PredictionCache:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        tracks = np.asarray(z["pred_tracks_xy"], dtype=np.float32)
        occ = np.asarray(z["pred_occluded"]).astype(bool, copy=False)
        runtime = float(np.asarray(z.get("runtime_sec", [-1.0]), dtype=np.float64).reshape(-1)[0])
        if runtime < 0:
            runtime = None
        meta_json = str(z.get("meta_json", np.asarray(["{}"]))[0])
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {"_meta_json_parse_error": True, "_raw": meta_json}

    return PredictionCache(pred_tracks_xy=tracks, pred_occluded=occ, runtime_sec=runtime, meta=meta)

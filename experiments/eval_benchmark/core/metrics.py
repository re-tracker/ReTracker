from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from retracker.evaluation.eval_utils import compute_tapvid_metrics

if TYPE_CHECKING:  # pragma: no cover
    from experiments.benchmark.core.types import TrackingResult


def compute_tapvid_davis_first_metrics(
    *,
    query_points_tyx: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks_xy: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks_xy: np.ndarray,
) -> dict[str, float]:
    """Compute TAP-Vid metrics for the DAVIS "first" protocol for a single video."""

    q = np.asarray(query_points_tyx, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"query_points_tyx must be (N,3), got {q.shape}")

    gt_occ = np.asarray(gt_occluded).astype(bool, copy=False)
    pred_occ = np.asarray(pred_occluded).astype(bool, copy=False)
    if gt_occ.ndim != 2:
        raise ValueError(f"gt_occluded must be (N,T), got {gt_occ.shape}")
    if pred_occ.shape != gt_occ.shape:
        raise ValueError(f"pred_occluded shape mismatch: {pred_occ.shape} vs {gt_occ.shape}")

    gt_tr = np.asarray(gt_tracks_xy, dtype=np.float32)
    pred_tr = np.asarray(pred_tracks_xy, dtype=np.float32)
    if gt_tr.ndim != 3 or gt_tr.shape[-1] != 2:
        raise ValueError(f"gt_tracks_xy must be (N,T,2), got {gt_tr.shape}")
    if pred_tr.shape != gt_tr.shape:
        raise ValueError(f"pred_tracks_xy shape mismatch: {pred_tr.shape} vs {gt_tr.shape}")

    # compute_tapvid_metrics expects batch dimension.
    raw = compute_tapvid_metrics(
        q[None, ...],
        gt_occ[None, ...],
        gt_tr[None, ...],
        pred_occ[None, ...],
        pred_tr[None, ...],
        query_mode="first",
    )

    out: dict[str, float] = {}
    for k, v in raw.items():
        out[k] = float(np.asarray(v).reshape(-1)[0])
    return out


def compute_tapvid_davis_first_metrics_from_trackingresult(
    *,
    pred: "TrackingResult",
    query_points_tyx: np.ndarray,
    gt_tracks_xy: np.ndarray,
    gt_occluded: np.ndarray,
) -> dict[str, float]:
    """
    Convenience wrapper to compute DAVIS-first TAP-Vid metrics from a standardized
    `TrackingResult` produced by benchmark runner scripts.

    TrackingResult stores tracks in (T,N,2) and visibles in (T,N).
    TAP-Vid metrics expect (N,T,2) and occluded (N,T).
    """

    tracks_tn2 = np.asarray(pred.tracks.tracks_xy, dtype=np.float32)
    vis_tn = np.asarray(pred.tracks.visibles).astype(bool, copy=False)
    if tracks_tn2.ndim != 3 or tracks_tn2.shape[-1] != 2:
        raise ValueError(f"pred tracks must be (T,N,2), got {tracks_tn2.shape}")
    if vis_tn.shape != tracks_tn2.shape[:2]:
        raise ValueError(f"pred visibles shape mismatch: {vis_tn.shape} vs {tracks_tn2.shape[:2]}")

    pred_tracks_xy = np.transpose(tracks_tn2, (1, 0, 2))  # (N,T,2)
    pred_occluded = ~np.transpose(vis_tn, (1, 0))  # (N,T)

    return compute_tapvid_davis_first_metrics(
        query_points_tyx=query_points_tyx,
        gt_occluded=gt_occluded,
        gt_tracks_xy=gt_tracks_xy,
        pred_occluded=pred_occluded,
        pred_tracks_xy=pred_tracks_xy,
    )


def aggregate_metrics(per_video: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compute simple mean across videos for each metric key."""

    names = sorted(per_video.keys())
    if not names:
        return {"avg": {}, "per_video": {}, "num_videos": 0}

    keys = sorted({k for v in per_video.values() for k in v.keys()})

    avg: dict[str, float] = {}
    for k in keys:
        vals = [float(per_video[n].get(k)) for n in names if k in per_video[n]]
        if not vals:
            continue
        avg[k] = float(round(sum(vals) / float(len(vals)), 10))

    return {"avg": avg, "per_video": per_video, "num_videos": len(names)}

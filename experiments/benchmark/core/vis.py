from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import TrackingResult


@dataclass(frozen=True)
class VisConfig:
    point_size: int = 100  # matplotlib-like "s"; we convert to an OpenCV radius
    hide_before_query: bool = True
    label: str | None = None


def _point_radius(point_size: int) -> int:
    # Matplotlib's scatter `s` is area-ish. Use a gentle sqrt mapping to radius.
    ps = max(1, int(point_size))
    return max(1, int(round((ps**0.5) / 2.0)))


def _compute_colors_bgr_from_queries(queries_txy: np.ndarray) -> np.ndarray:
    # Color by y at query time (stable across frames). Output (N,3) uint8 BGR.
    q = np.asarray(queries_txy, dtype=np.float32)
    y = q[:, 2]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    denom = max(1e-6, y_max - y_min)
    y_norm = (y - y_min) / denom  # 0..1

    # Hue in OpenCV HSV is [0..179]. Map y -> hue.
    hue = (y_norm * 179.0).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = np.full_like(hue, 255, dtype=np.uint8)
    hsv = np.stack([hue, sat, val], axis=1).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return bgr.astype(np.uint8, copy=False)


def _draw_label(frame_bgr: np.ndarray, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    x0, y0 = 10, 10
    cv2.rectangle(frame_bgr, (x0 - pad, y0 - pad), (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label, (x0, y0 + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_overlay_rgb(video_rgb: np.ndarray, result: TrackingResult, cfg: VisConfig) -> np.ndarray:
    """
    Render an RGB overlay video (uint8) for a single method.
    """
    frames = np.asarray(video_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video_rgb must be (T,H,W,3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)

    tracks = np.asarray(result.tracks.tracks_xy, dtype=np.float32)  # (T,N,2)
    vis = np.asarray(result.tracks.visibles).astype(bool)  # (T,N)
    q = np.asarray(result.queries.txy, dtype=np.float32)  # (N,3)

    T, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if tracks.shape[0] != T:
        raise ValueError(f"Track length mismatch: video T={T} tracks T={tracks.shape[0]}")
    if tracks.shape[1] != q.shape[0]:
        raise ValueError(f"Track N mismatch: tracks N={tracks.shape[1]} queries N={q.shape[0]}")

    colors = _compute_colors_bgr_from_queries(q)  # (N,3) BGR
    radius = _point_radius(cfg.point_size)

    # Per-point query start frame (rounded/clamped).
    q_t = np.clip(np.round(q[:, 0]).astype(np.int32), 0, max(0, T - 1))

    out = np.empty_like(frames)
    for t in range(T):
        frame_bgr = frames[t][..., ::-1].copy()
        pts = tracks[t]  # (N,2) float
        vis_t = vis[t]  # (N,)

        # Clamp to image bounds.
        xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, H - 1)

        for i in range(pts.shape[0]):
            if cfg.hide_before_query and t < int(q_t[i]):
                continue
            col = colors[i].tolist()
            if not bool(vis_t[i]):
                col = [int(c * 0.25) for c in col]  # dim occluded points
            cv2.circle(frame_bgr, (int(xs[i]), int(ys[i])), radius, col, thickness=-1, lineType=cv2.LINE_AA)

        if cfg.label:
            _draw_label(frame_bgr, cfg.label)

        out[t] = frame_bgr[..., ::-1]
    return out


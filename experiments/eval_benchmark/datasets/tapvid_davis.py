from __future__ import annotations

import os
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TapVidDavisRaw = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TapVidDavisSequence:
    """A single TAP-Vid DAVIS sequence for evaluation.

    Frames are uint8 RGB in (T, H, W, 3). Tracks are in pixel coordinates at the
    same resolution.
    """

    name: str
    frames_uint8: np.ndarray  # (T,H,W,3) uint8
    query_points_tyx: np.ndarray  # (N,3) float32 [t, y, x] pixels
    gt_tracks_xy: np.ndarray  # (N,T,2) float32 (x,y) pixels
    gt_occluded: np.ndarray  # (N,T) bool (True = occluded)
    resize_hw: tuple[int, int] | None


def load_tapvid_davis_pickle(pkl_path: str | Path) -> TapVidDavisRaw:
    # Some local dataset layouts store `tapvid_davis.pkl` as a symlink, and the
    # actual pickle is `tapvid_davis.pkl.full` next to the symlink target.
    p = resolve_existing_pickle_path(pkl_path)
    with p.open("rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict from pickle, got {type(raw)}")

    # The pickle is expected to be: {seq_name: {"video": ..., "points": ..., "occluded": ...}}
    return raw  # type: ignore[return-value]


def resolve_existing_pickle_path(pkl_path: str | Path, *, max_depth: int = 8) -> Path:
    """
    Resolve a pickle path that may be a broken symlink, following a short
    symlink chain and checking the `.full` variant at each step.
    """

    p = Path(pkl_path)
    for _ in range(int(max_depth)):
        if p.exists():
            return p

        full = Path(str(p) + ".full")
        if full.exists():
            return full

        if not p.is_symlink():
            break

        try:
            target = os.readlink(str(p))
        except OSError:
            break

        p = Path(os.path.normpath(os.path.join(str(p.parent), target)))

    return Path(pkl_path)


def _decode_jpeg_bytes(frames: Any) -> np.ndarray:
    """Decode a list/array of JPEG-encoded byte frames into uint8 RGB."""

    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required to decode TAP-Vid JPEG bytes frames") from e

    decoded = []
    for f in frames:
        if not isinstance(f, (bytes, bytearray)):
            raise TypeError(f"Expected bytes frames, got: {type(f)}")
        img = Image.open(io.BytesIO(f))
        decoded.append(np.array(img))
    arr = np.asarray(decoded)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Decoded frames must be (T,H,W,3), got {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def _resize_video_uint8(frames_uint8: np.ndarray, resize_hw: tuple[int, int] | None) -> np.ndarray:
    if resize_hw is None:
        return frames_uint8

    frames = np.asarray(frames_uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)

    out_h, out_w = int(resize_hw[0]), int(resize_hw[1])
    if frames.shape[1] == out_h and frames.shape[2] == out_w:
        return frames

    # Use Pillow for resizing to avoid importing OpenCV at eval time.
    #
    # Some environments have an OpenCV build that is incompatible with the host
    # numpy version (e.g., cv2 compiled against numpy 1.x with numpy 2.x
    # installed), which can crash imports and even poison later imports in child
    # processes. Pillow is slower but far more robust for this cache-building
    # step.
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required to resize TAP-Vid videos") from e

    resized = []
    for t in range(frames.shape[0]):
        img = Image.fromarray(frames[t])
        img = img.resize((out_w, out_h), resample=Image.BILINEAR)
        resized.append(np.asarray(img, dtype=np.uint8))
    return np.stack(resized, axis=0)


def _scale_tracks_xy_to_pixels(tracks_xy01: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    tracks = np.asarray(tracks_xy01, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"points must be (N,T,2), got {tracks.shape}")
    h, w = int(hw[0]), int(hw[1])
    scale = np.array([w - 1, h - 1], dtype=np.float32).reshape(1, 1, 2)
    return tracks * scale


def _sample_queries_first(
    gt_occluded: np.ndarray, gt_tracks_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-query protocol: for each track, use its first visible point as the query."""

    occ = np.asarray(gt_occluded).astype(bool, copy=False)
    tracks = np.asarray(gt_tracks_xy, dtype=np.float32)
    if occ.ndim != 2:
        raise ValueError(f"occluded must be (N,T), got {occ.shape}")
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be (N,T,2), got {tracks.shape}")
    if occ.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"shape mismatch: occluded {occ.shape} vs tracks {tracks.shape}")

    # Filter tracks that are never visible.
    valid = np.sum(~occ, axis=1) > 0
    occ = occ[valid]
    tracks = tracks[valid]

    query_points = []
    for i in range(tracks.shape[0]):
        t0 = int(np.where(~occ[i])[0][0])
        x, y = tracks[i, t0, 0], tracks[i, t0, 1]
        query_points.append(np.array([float(t0), float(y), float(x)], dtype=np.float32))

    return np.stack(query_points, axis=0), occ, tracks


def make_tapvid_davis_sequence(
    *,
    name: str,
    item: dict[str, Any],
    resize_hw: tuple[int, int] | None,
) -> TapVidDavisSequence:
    if not isinstance(item, dict):
        raise TypeError(f"Expected dict for sequence {name}, got {type(item)}")

    frames = item["video"]
    points = np.asarray(item["points"], dtype=np.float32)
    occluded = np.asarray(item["occluded"]).astype(bool, copy=False)

    if isinstance(frames, np.ndarray):
        frames_uint8 = frames.astype(np.uint8, copy=False)
    else:
        frames_uint8 = _decode_jpeg_bytes(frames)

    frames_uint8 = _resize_video_uint8(frames_uint8, resize_hw=resize_hw)
    h, w = int(frames_uint8.shape[1]), int(frames_uint8.shape[2])

    gt_tracks_xy = _scale_tracks_xy_to_pixels(points, hw=(h, w))
    query_points_tyx, gt_occluded, gt_tracks_xy = _sample_queries_first(occluded, gt_tracks_xy)

    return TapVidDavisSequence(
        name=str(name),
        frames_uint8=frames_uint8,
        query_points_tyx=query_points_tyx.astype(np.float32, copy=False),
        gt_tracks_xy=gt_tracks_xy.astype(np.float32, copy=False),
        gt_occluded=gt_occluded.astype(bool, copy=False),
        resize_hw=resize_hw,
    )


def load_tapvid_davis_first(
    pkl_path: str | Path,
    resize_hw: tuple[int, int] | None = (256, 256),
    max_videos: int | None = None,
    video_names: list[str] | None = None,
) -> list[TapVidDavisSequence]:
    """Load TAP-Vid DAVIS pickle and build per-sequence evaluation samples."""

    raw = load_tapvid_davis_pickle(pkl_path)

    if video_names is None:
        names = sorted(raw.keys())
    else:
        names = list(video_names)

    if max_videos is not None:
        names = names[: int(max_videos)]

    return [make_tapvid_davis_sequence(name=str(n), item=raw[n], resize_hw=resize_hw) for n in names]

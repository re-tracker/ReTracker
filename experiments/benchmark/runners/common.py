from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Ensure we can import `experiments.*` when this file is imported from a runner
# executed inside an arbitrary conda env.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.benchmark.core.types import Queries, TrackSet, TrackingResult  # noqa: E402


def ensure_repo_root_on_syspath() -> Path:
    """
    Make `import experiments...` work when runners are executed as scripts inside
    arbitrary conda envs.
    """
    return _REPO_ROOT


def load_queries_txt(path: str | Path) -> Queries:
    return Queries.load_txt(path)


def shift_queries_for_clip(queries: Queries, *, start: int, clip_len: int) -> Queries:
    """
    If we load a sub-clip of the original video starting at `start`, convert
    query times from original-frame indices to clip-frame indices.
    """
    txy = queries.txy.copy()
    t = np.round(txy[:, 0]).astype(np.int32)
    t = t - int(start)
    t = np.clip(t, 0, max(0, int(clip_len) - 1))
    txy[:, 0] = t.astype(np.float32)
    return Queries(txy=txy)


def load_video_rgb(
    video_path: str | Path,
    *,
    resized_hw: tuple[int, int],
    start: int = 0,
    max_frames: int | None = None,
) -> np.ndarray:
    vp = Path(video_path)
    if vp.suffix.lower() == ".npz":
        with np.load(vp, allow_pickle=False) as z:
            if "frames" not in z.files:
                raise KeyError(f"NPZ video missing 'frames' key: {vp}")
            frames = np.asarray(z["frames"])

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"NPZ frames must be (T,H,W,3), got {frames.shape} from {vp}")
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        h, w = (int(resized_hw[0]), int(resized_hw[1]))
        if int(frames.shape[1]) != h or int(frames.shape[2]) != w:
            raise ValueError(
                f"NPZ frames resolution mismatch: frames={frames.shape[1]}x{frames.shape[2]} "
                f"but resized_hw={h}x{w} for {vp}"
            )

        s = max(0, int(start))
        e = frames.shape[0] if max_frames is None else min(frames.shape[0], s + int(max_frames))
        out = frames[s:e]
        if out.shape[0] == 0:
            raise RuntimeError(f"No frames extracted from: {vp}")
        return out.astype(np.uint8, copy=False)

    # Import OpenCV lazily. This keeps NPZ-only workflows and unit tests free of
    # a hard cv2 dependency (and avoids binary wheels mismatching the host numpy).
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required to load non-NPZ videos. "
            "Install a compatible opencv-python for your numpy version."
        ) from e

    p = str(vp)
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {p}")

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

    out: list[np.ndarray] = []
    count = 0
    h, w = resized_hw
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        out.append(frame_rgb)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    if not out:
        raise RuntimeError(f"No frames extracted from: {p}")
    return np.stack(out, axis=0).astype(np.uint8, copy=False)


@dataclass(frozen=True)
class Timed:
    runtime_sec: float


def time_it(fn: Callable[[], Any]) -> tuple[Any, Timed]:
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    return out, Timed(runtime_sec=float(dt))


def save_result_npz(
    *,
    out_npz: str | Path,
    method: str,
    video_path: str | Path,
    resized_hw: tuple[int, int],
    queries: Queries,
    tracks_xy_tn2: np.ndarray,
    visibles_tn: np.ndarray,
    runtime_sec: float | None,
    meta: dict[str, Any] | None = None,
) -> None:
    result = TrackingResult(
        method=method,
        video_path=str(video_path),
        resized_hw=(int(resized_hw[0]), int(resized_hw[1])),
        queries=queries,
        tracks=TrackSet(tracks_xy=tracks_xy_tn2, visibles=visibles_tn),
        runtime_sec=runtime_sec,
        meta=meta or {},
    )
    result.save_npz(out_npz)

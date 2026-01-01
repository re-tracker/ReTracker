from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoData:
    path: str
    fps: float
    frames_rgb: np.ndarray  # (T, H, W, 3) uint8

    @property
    def t(self) -> int:
        return int(self.frames_rgb.shape[0])

    @property
    def h(self) -> int:
        return int(self.frames_rgb.shape[1])

    @property
    def w(self) -> int:
        return int(self.frames_rgb.shape[2])


def load_video_rgb(
    video_path: str | Path,
    *,
    resized_hw: tuple[int, int] | None = None,
    start: int = 0,
    max_frames: int | None = None,
) -> VideoData:
    """
    Load a video as RGB uint8 frames.

    resized_hw: (H, W) if provided, frames are resized per-frame using OpenCV.
    start/max_frames: optional sub-sampling (useful for smoke tests).
    """
    p = str(video_path)
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {p}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

    out: list[np.ndarray] = []
    count = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resized_hw is not None:
            h, w = resized_hw
            frame_rgb = cv2.resize(frame_rgb, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        out.append(frame_rgb)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    if not out:
        raise RuntimeError(f"No frames extracted from: {p}")

    frames = np.stack(out, axis=0).astype(np.uint8, copy=False)
    return VideoData(path=p, fps=fps, frames_rgb=frames)


def save_video_rgb(frames_rgb: np.ndarray, out_path: str | Path, *, fps: float, codec: str = "mp4v") -> None:
    frames = np.asarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = int(frames.shape[1]), int(frames.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))

    for frame_rgb in frames:
        writer.write(frame_rgb[..., ::-1])  # RGB -> BGR
    writer.release()


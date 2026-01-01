from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MosaicInput:
    label: str
    video_path: Path


def _label_frame(img_bgr: np.ndarray, label: str) -> np.ndarray:
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    x0, y0 = 10, 10
    cv2.rectangle(out, (x0 - pad, y0 - pad), (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
    cv2.putText(out, label, (x0, y0 + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def compose_mosaic_mp4(
    inputs: list[MosaicInput],
    *,
    out_mp4: Path,
    cols: int,
    fps: float,
) -> None:
    """
    Compose multiple videos into a labeled grid mp4.

    - If a video ends early, we hold its last frame.
    - If resolutions differ, we resize to the max cell size.
    """
    if cols <= 0:
        raise ValueError(f"cols must be > 0, got {cols}")
    if not inputs:
        raise ValueError("No mosaic inputs.")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    caps: list[cv2.VideoCapture] = []
    try:
        for inp in inputs:
            cap = cv2.VideoCapture(str(inp.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {inp.video_path}")
            caps.append(cap)

        widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
        heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
        cell_w = max(widths)
        cell_h = max(heights)
        if cell_w <= 0 or cell_h <= 0:
            raise RuntimeError("Failed to infer cell size from input videos.")

        n = len(inputs)
        rows = int(ceil(n / cols))
        out_w = cell_w * cols
        out_h = cell_h * rows

        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        max_frames = max(counts) if counts and all(x > 0 for x in counts) else None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (out_w, out_h))

        blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        last_frames = [blank for _ in caps]
        ended = [False for _ in caps]

        def read_or_hold(i: int) -> np.ndarray:
            if ended[i]:
                return last_frames[i]
            ok, frame = caps[i].read()
            if (not ok) or frame is None:
                ended[i] = True
                return last_frames[i]
            if frame.shape[0] != cell_h or frame.shape[1] != cell_w:
                frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            last_frames[i] = frame
            return frame

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            if max_frames is None and all(ended) and frame_idx > 0:
                break

            grid = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            for i, inp in enumerate(inputs):
                r = i // cols
                c = i % cols
                frame = read_or_hold(i)
                frame_l = _label_frame(frame, inp.label)
                y0 = r * cell_h
                x0 = c * cell_w
                grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = frame_l

            writer.write(grid)
            frame_idx += 1

        writer.release()
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


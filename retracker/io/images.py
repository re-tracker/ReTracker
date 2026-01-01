from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


VideoLike = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class VideoFramesWriter:
    """Write a tensor/ndarray video to a directory of RGB frames."""

    filename_template: str = "frame_{t:04d}.png"

    def as_uint8_frames(self, video: VideoLike) -> np.ndarray:
        """Convert a video tensor/ndarray to uint8 RGB frames (T, H, W, 3)."""
        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
            if v.ndim == 5 and v.shape[0] == 1:
                v = v.squeeze(0)  # [T,C,H,W]
            if v.ndim == 4 and v.shape[1] in (1, 3):
                v = v.permute(0, 2, 3, 1)  # [T,H,W,C]
            elif v.ndim != 4:
                raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")

            if v.dtype.is_floating_point:
                vmax = float(v.max().item()) if v.numel() else 0.0
                scale = 255.0 if vmax <= 1.0 else 1.0
                v = (v * scale).clamp(0, 255).to(torch.uint8)
            else:
                v = v.clamp(0, 255).to(torch.uint8)
            frames = v.numpy()
        else:
            v = np.asarray(video)
            if v.ndim == 5 and v.shape[0] == 1:
                v = v[0]
            if v.ndim == 4 and v.shape[-1] in (1, 3):
                frames = v
            elif v.ndim == 4 and v.shape[1] in (1, 3):
                frames = np.transpose(v, (0, 2, 3, 1))
            else:
                raise ValueError(f"Unsupported video ndarray shape: {v.shape}")

            if np.issubdtype(frames.dtype, np.floating):
                vmax = float(frames.max()) if frames.size else 0.0
                scale = 255.0 if vmax <= 1.0 else 1.0
                frames = np.clip(frames * scale, 0, 255).astype(np.uint8)
            else:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        return frames

    def save(self, video: VideoLike, out_dir: str) -> None:
        """Save `video` as `frame_0000.png`, ... in `out_dir`."""
        frames = self.as_uint8_frames(video)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(frames):
            Image.fromarray(frame).save(out_path / self.filename_template.format(t=t))


def as_uint8_frames(video: VideoLike) -> np.ndarray:
    """Convert a video tensor/ndarray to uint8 RGB frames (T, H, W, 3)."""
    return VideoFramesWriter().as_uint8_frames(video)


def save_video_frames(video: VideoLike, out_dir: str) -> None:
    """Save a video tensor/ndarray as `frame_0000.png`, ... in `out_dir`."""
    VideoFramesWriter().save(video, out_dir)

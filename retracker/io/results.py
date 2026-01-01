from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from retracker.io.images import VideoFramesWriter, VideoLike
from retracker.utils.rich_utils import CONSOLE


@dataclass
class ResultsDumper:
    """Persist model outputs (tracks/visibility) to disk.

    Notes:
    - `dump_npz` stores tracks/visibility.
    - `dump_images` stores per-frame RGB images extracted from `video`. To keep this
      explicit and avoid hidden dependencies, `video` must be provided.
    """

    save_dir: str
    dump_npz: bool = True
    dump_images: bool = False
    frames_writer: VideoFramesWriter = field(default_factory=VideoFramesWriter)

    def dump(
        self,
        video_name: str,
        pred_tracks: Any,
        pred_visibility: Any,
        *,
        video: VideoLike | None = None,
    ) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        if isinstance(pred_tracks, torch.Tensor):
            pred_tracks = pred_tracks.detach().cpu().numpy()
        if isinstance(pred_visibility, torch.Tensor):
            pred_visibility = pred_visibility.detach().cpu().numpy()

        if self.dump_npz:
            save_path = os.path.join(self.save_dir, f"{video_name}.npz")
            np.savez(
                save_path,
                video_name=video_name,
                pred_tracks=pred_tracks,
                pred_visibility=pred_visibility,
            )
            CONSOLE.print(f"[bold yellow]Tracking results saved: {save_path}")

        if self.dump_images:
            if video is None:
                raise ValueError("dump_images=True requires `video` (a Tensor/ndarray video).")
            images_folder = os.path.join(self.save_dir, video_name)
            self.frames_writer.save(video, images_folder)
            CONSOLE.print(f"[bold yellow]Frames saved: {images_folder}")


def dump_results(
    save_dir: str,
    video_name: str,
    pred_tracks: Any,
    pred_visibility: Any,
    *,
    dump_npz: bool = True,
    dump_images: bool = False,
    video: VideoLike | None = None,
) -> None:
    """Functional wrapper around :class:`ResultsDumper` (convenience)."""
    ResultsDumper(save_dir=save_dir, dump_npz=dump_npz, dump_images=dump_images).dump(
        video_name,
        pred_tracks,
        pred_visibility,
        video=video,
    )

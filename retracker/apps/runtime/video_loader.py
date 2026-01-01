"""Video loading utilities for offline tracking apps.

Notes
-----
These loaders return a full video tensor shaped `(1, T, C, H, W)`. For the apps,
this is OK for short clips, but can be memory-heavy for long videos.

Two practical requirements for a "usable" CLI:
- `max_frames` must stop decoding early (otherwise `--max_frames` is misleading).
- Resize should happen before moving data to the target device to avoid OOM and
  long host->device copies of high-res frames.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import cv2

try:
    import mediapy as media  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dep
    media = None

from ..config import VideoConfig
from ..utils import detect_video_rotation


class BaseVideoLoader(ABC):
    """Abstract base class for video loading."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        
    @abstractmethod
    def load(self, video_path: str, device: str = 'cuda') -> torch.Tensor:
        """
        Load video and return tensor.
        
        Args:
            video_path: Path to video file
            device: Target device
            
        Returns:
            Video tensor of shape (1, T, C, H, W)
        """
        pass
    
    def _needs_180_rotation(self, video_path: str) -> bool:
        """Return True if the video should be rotated 180 degrees."""
        if not self.config.auto_rotate:
            return False
        _, needs_180_rotation = detect_video_rotation(video_path)
        return needs_180_rotation
    
    def _resize_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Resize a single RGB frame to `config.resized_wh` (W, H)."""
        W, H = self.config.resized_wh
        if frame_rgb.shape[1] == W and frame_rgb.shape[0] == H:
            return frame_rgb
        return cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _to_tensor(video: np.ndarray, device: str) -> torch.Tensor:
        """Convert `(T, H, W, C)` numpy video to `(1, T, C, H, W)` float32 torch tensor."""
        video_tensor = torch.from_numpy(video)
        # Keep conversion simple and predictable for downstream code.
        video_tensor = video_tensor.permute(0, 3, 1, 2).contiguous().float()
        return video_tensor.unsqueeze(0).to(device, non_blocking=True)


class OpenCVVideoLoader(BaseVideoLoader):
    """OpenCV-based video loader (handles HEVC, HDR, etc.)."""
    
    def load(self, video_path: str, device: str = 'cuda') -> torch.Tensor:
        """Load video using OpenCV."""
        needs_180_rotation = self._needs_180_rotation(video_path)
        max_frames = self.config.max_frames

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")
        
        frames = []
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if needs_180_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = self._resize_frame(frame_rgb)
            frames.append(frame_rgb)
        
        cap.release()
        
        if not frames:
            raise RuntimeError(f"No frames extracted from video: {video_path}")
        
        # Stack frames (T, H, W, C)
        video = np.stack(frames, axis=0)
        return self._to_tensor(video, device)


class MediaPyVideoLoader(BaseVideoLoader):
    """MediaPy-based video loader (fallback)."""
    
    def load(self, video_path: str, device: str = 'cuda') -> torch.Tensor:
        """Load video using MediaPy."""
        if media is None:
            raise ModuleNotFoundError("mediapy")

        video = media.read_video(video_path)  # (T, H, W, C)

        if self._needs_180_rotation(video_path):
            # Rotate 180 degrees and copy to fix negative strides
            video = np.rot90(video, k=2, axes=(1, 2)).copy()

        if self.config.max_frames is not None:
            video = video[: self.config.max_frames]

        # Resize on CPU (frame-wise) before moving to `device`.
        W, H = self.config.resized_wh
        if video.shape[2] != W or video.shape[1] != H:
            resized = np.empty((video.shape[0], H, W, video.shape[3]), dtype=video.dtype)
            for i in range(video.shape[0]):
                resized[i] = cv2.resize(video[i], (W, H), interpolation=cv2.INTER_LINEAR)
            video = resized

        return self._to_tensor(video, device)


class AutoVideoLoader(BaseVideoLoader):
    """Automatic fallback loader."""
    
    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self.loaders = [
            OpenCVVideoLoader(config),
            MediaPyVideoLoader(config),
        ]
    
    def load(self, video_path: str, device: str = 'cuda') -> torch.Tensor:
        """Try loading with multiple backends."""
        last_error = None
        
        for loader in self.loaders:
            try:
                return loader.load(video_path, device)
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(
            f"All video loaders failed. Last error: {last_error}"
        )


class VideoLoaderFactory:
    """Factory for creating video loaders."""
    
    @staticmethod
    def create(config: VideoConfig) -> BaseVideoLoader:
        """
        Create video loader based on config.
        
        Args:
            config: VideoConfig instance
            
        Returns:
            Video loader instance
        """
        loaders = {
            'opencv': OpenCVVideoLoader,
            'mediapy': MediaPyVideoLoader,
            'auto': AutoVideoLoader,
        }
        
        loader_class = loaders.get(config.backend, AutoVideoLoader)
        return loader_class(config)

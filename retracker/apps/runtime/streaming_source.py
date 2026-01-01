"""Streaming video source implementations."""

import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Iterator, List
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore

from ..config import StreamingSourceConfig
from ..utils import detect_video_rotation
from retracker.utils.rich_utils import CONSOLE


def natural_sort_key(s: str):
    """
    Natural sorting key for strings with numbers.
    E.g., sorts ['1', '2', '10'] correctly instead of ['1', '10', '2'].
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


class BaseStreamingSource(ABC):
    """Abstract base class for streaming video sources."""

    def __init__(self, config: StreamingSourceConfig):
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for streaming sources. Ensure `opencv-python` is installed and compatible."
            )
        self.config = config
        self.frame_count = 0
        self.start_time = None

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """
        Iterate over frames.

        Yields:
            Tuple of (frame, metadata)
            - frame: RGB numpy array (H, W, 3)
            - metadata: dict with timestamp, frame_id, etc.
        """
        pass

    @abstractmethod
    def release(self):
        """Release resources."""
        pass

    def get_total_frames(self) -> int:
        """Get total number of frames. Returns 0 if unknown."""
        return 0

    def get_current_frame(self) -> int:
        """Get current frame position."""
        return self.frame_count

    def seek(self, frame_idx: int) -> bool:
        """Seek to a specific frame. Returns True if successful."""
        return False

    def supports_seek(self) -> bool:
        """Check if this source supports seeking."""
        return False

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame (resize, rotate, etc.)."""
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume OpenCV BGR format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions
        W, H = self.config.resized_wh
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

        return frame

    def _frame_to_tensor(
        self,
        frame: np.ndarray,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Convert frame to torch tensor."""
        # (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(frame).to(device)
        tensor = tensor.permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0)
        return tensor


class CameraSource(BaseStreamingSource):
    """Camera input source."""

    def __init__(self, config: StreamingSourceConfig):
        super().__init__(config)
        self.cap = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Stream frames from camera."""
        self.cap = cv2.VideoCapture(self.config.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.config.camera_id}")

        # Set camera properties if needed
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.start_time = time.time()
        self.frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    CONSOLE.print("Failed to read frame from camera")
                    break

                # Skip frames if configured
                if self.config.skip_frames > 0:
                    for _ in range(self.config.skip_frames):
                        self.cap.read()

                # Preprocess frame
                frame = self._preprocess_frame(frame)

                # Create metadata
                metadata = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time() - self.start_time,
                    'source': 'camera',
                    'camera_id': self.config.camera_id,
                }

                self.frame_count += 1
                yield frame, metadata

        finally:
            self.release()

    def release(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileSource(BaseStreamingSource):
    """Video file source with optional real-time simulation."""

    def __init__(self, config: StreamingSourceConfig):
        super().__init__(config)
        self.cap = None
        self.video_fps = None
        self.needs_180_rotation = False
        self._total_frames = 0
        self._seek_requested = False
        self._seek_target = 0

    def get_total_frames(self) -> int:
        """Get total number of frames in the video."""
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # If cap not open yet, open temporarily to get frame count
        if self._total_frames == 0 and self.config.video_path:
            temp_cap = cv2.VideoCapture(self.config.video_path)
            if temp_cap.isOpened():
                self._total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                temp_cap.release()
        return self._total_frames

    def get_current_frame(self) -> int:
        """Get current frame position."""
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self.frame_count

    def seek(self, frame_idx: int) -> bool:
        """Seek to a specific frame."""
        if self.cap is None:
            return False
        total = self.get_total_frames()
        frame_idx = max(0, min(frame_idx, total - 1))
        self._seek_requested = True
        self._seek_target = frame_idx
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def supports_seek(self) -> bool:
        """Video files support seeking."""
        return True

    def _get_frame_segments(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Get validated frame segments to process.

        Args:
            total_frames: Total frames in the video

        Returns:
            List of (start, end) tuples with validated frame indices
        """
        segments = self.config.frame_segments

        # If no segments specified, process all frames
        if segments is None or len(segments) == 0:
            return [(0, total_frames)]

        # Validate and clip segments
        validated = []
        for start, end in segments:
            # Clip start to valid range
            start = max(0, min(start, total_frames - 1))

            # Handle None end (until end of video)
            if end is None:
                end = total_frames
            else:
                end = max(start + 1, min(end, total_frames))

            # Only add if segment has frames
            if end > start:
                validated.append((start, end))

        # Sort by start frame and merge overlapping segments
        if not validated:
            return [(0, total_frames)]

        validated.sort(key=lambda x: x[0])

        # Merge overlapping segments
        merged = [validated[0]]
        for start, end in validated[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # Overlapping or adjacent
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Stream frames from video file with support for multiple frame segments."""
        if self.config.video_path is None:
            raise ValueError("video_path must be specified for VideoFileSource")

        if not Path(self.config.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.config.video_path}")

        # Detect rotation before opening video
        if self.config.auto_rotate:
            _, self.needs_180_rotation = detect_video_rotation(self.config.video_path)
            if self.needs_180_rotation:
                CONSOLE.print(f"Detected 180-degree rotation, will apply correction")

        self.cap = cv2.VideoCapture(self.config.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.config.video_path}")

        # Get video properties
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get validated frame segments
        frame_segments = self._get_frame_segments(total_frames)

        # Calculate total frames to process
        total_to_process = sum(end - start for start, end in frame_segments)
        CONSOLE.print(f"Frame segments: {frame_segments}")
        CONSOLE.print(f"Total frames to process: {total_to_process} (video has {total_frames} frames)")

        # Calculate frame delay for real-time simulation
        if self.config.simulate_realtime:
            frame_delay = 1.0 / self.config.target_fps
        else:
            frame_delay = 0

        self.start_time = time.time()
        self.frame_count = 0
        last_frame_time = time.time()

        try:
            # Iterate through each segment
            for segment_idx, (seg_start, seg_end) in enumerate(frame_segments):
                # Seek to segment start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
                current_frame_idx = seg_start

                CONSOLE.print(f"Processing segment {segment_idx + 1}/{len(frame_segments)}: frames {seg_start}-{seg_end}")

                while current_frame_idx < seg_end:
                    # Check if seek was requested
                    if self._seek_requested:
                        self._seek_requested = False
                        current_frame_idx = self._seek_target
                        # Update segment end to process until end of video
                        seg_end = total_frames
                        self.frame_count = current_frame_idx

                    ret, frame = self.cap.read()

                    if not ret:
                        break

                    # Apply 180-degree rotation if needed
                    if self.needs_180_rotation:
                        frame = np.rot90(frame, k=2, axes=(0, 1)).copy()

                    # Skip frames if configured
                    if self.config.skip_frames > 0:
                        for _ in range(self.config.skip_frames):
                            self.cap.read()
                            current_frame_idx += 1

                    # Simulate real-time playback
                    if self.config.simulate_realtime:
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        if elapsed < frame_delay:
                            time.sleep(frame_delay - elapsed)
                        last_frame_time = time.time()

                    # Preprocess frame
                    frame = self._preprocess_frame(frame)

                    # Create metadata
                    metadata = {
                        'frame_id': current_frame_idx,
                        'original_frame_idx': current_frame_idx,
                        'timestamp': time.time() - self.start_time,
                        'source': 'video_file',
                        'video_path': self.config.video_path,
                        'video_fps': self.video_fps,
                        'total_frames': total_frames,
                        'segment_idx': segment_idx,
                        'segment_start': seg_start,
                        'segment_end': seg_end,
                        'frame_segments': frame_segments,
                    }

                    self.frame_count = current_frame_idx
                    current_frame_idx += 1
                    yield frame, metadata

        finally:
            self.release()

    def release(self):
        """Release video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RTSPSource(BaseStreamingSource):
    """
    RTSP network camera source with robust streaming support.

    Features:
    - Automatic reconnection on connection loss
    - Threaded frame reading for reduced latency
    - Configurable timeout and retry settings
    - Optional GStreamer backend support
    """

    def __init__(self, config: StreamingSourceConfig):
        super().__init__(config)
        self.cap = None
        self._running = False
        self._frame_queue = None
        self._reader_thread = None
        self._last_frame = None
        self._connection_lost = False

        # RTSP specific settings (with defaults)
        self.max_retries = getattr(config, 'rtsp_max_retries', 5)
        self.retry_delay = getattr(config, 'rtsp_retry_delay', 2.0)
        self.connection_timeout = getattr(config, 'rtsp_timeout', 10.0)
        self.use_threading = getattr(config, 'rtsp_use_threading', True)
        self.use_gstreamer = getattr(config, 'rtsp_use_gstreamer', False)
        self.queue_size = getattr(config, 'rtsp_queue_size', 2)

    def _build_gstreamer_pipeline(self, rtsp_url: str) -> str:
        """Build GStreamer pipeline string for RTSP."""
        # GStreamer pipeline for low-latency RTSP streaming
        pipeline = (
            f'rtspsrc location="{rtsp_url}" latency=0 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            'videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
        )
        return pipeline

    def _open_stream(self) -> bool:
        """
        Open RTSP stream with retry logic.

        Returns:
            True if successfully opened, False otherwise
        """
        rtsp_url = self.config.rtsp_url

        for attempt in range(self.max_retries):
            try:
                # Release existing capture if any
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                # Try GStreamer backend first if enabled
                if self.use_gstreamer:
                    try:
                        pipeline = self._build_gstreamer_pipeline(rtsp_url)
                        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                        if self.cap.isOpened():
                            CONSOLE.print(f"[RTSP] Connected using GStreamer backend")
                            return True
                    except Exception as e:
                        CONSOLE.print(f"[RTSP] GStreamer failed: {e}, falling back to FFmpeg")

                # Use FFmpeg/default backend
                # Set environment for FFmpeg RTSP options
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

                if not self.cap.isOpened():
                    # Try without specifying backend
                    self.cap = cv2.VideoCapture(rtsp_url)

                if self.cap.isOpened():
                    # Configure capture for low latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Try to set other properties (may not be supported)
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    except Exception:
                        pass

                    CONSOLE.print(f"[RTSP] Connected to {rtsp_url}")
                    return True

                CONSOLE.print(f"[RTSP] Connection attempt {attempt + 1}/{self.max_retries} failed")

            except Exception as e:
                CONSOLE.print(f"[RTSP] Error on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                CONSOLE.print(f"[RTSP] Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)

        return False

    def _reader_thread_func(self):
        """Background thread for continuous frame reading."""
        import queue

        consecutive_failures = 0
        max_consecutive_failures = 30  # About 1 second at 30fps

        while self._running:
            if self.cap is None or not self.cap.isOpened():
                self._connection_lost = True
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    CONSOLE.print("[RTSP] Too many consecutive read failures, marking connection as lost")
                    self._connection_lost = True
                    consecutive_failures = 0
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            self._connection_lost = False

            # Update the latest frame (drop old frames to reduce latency)
            try:
                # Clear old frames
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break

                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop frame if queue is full

    def _start_reader_thread(self):
        """Start the background frame reader thread."""
        import queue
        import threading

        self._frame_queue = queue.Queue(maxsize=self.queue_size)
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_thread_func, daemon=True)
        self._reader_thread.start()

    def _stop_reader_thread(self):
        """Stop the background frame reader thread."""
        self._running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._frame_queue = None

    def _get_frame_threaded(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from threaded reader."""
        import queue

        # Check for connection loss and attempt reconnection
        if self._connection_lost:
            CONSOLE.print("[RTSP] Attempting to reconnect...")
            if self._open_stream():
                self._connection_lost = False
            else:
                return False, None

        try:
            frame = self._frame_queue.get(timeout=self.connection_timeout)
            return True, frame
        except queue.Empty:
            CONSOLE.print("[RTSP] Frame read timeout")
            return False, None

    def _get_frame_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame directly (non-threaded)."""
        if self.cap is None or not self.cap.isOpened():
            # Attempt reconnection
            CONSOLE.print("[RTSP] Connection lost, attempting to reconnect...")
            if not self._open_stream():
                return False, None

        ret, frame = self.cap.read()

        if not ret:
            # Try to reconnect
            CONSOLE.print("[RTSP] Read failed, attempting to reconnect...")
            if self._open_stream():
                ret, frame = self.cap.read()

        return ret, frame if ret else None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Stream frames from RTSP source."""
        if self.config.rtsp_url is None:
            raise ValueError("rtsp_url must be specified for RTSPSource")

        CONSOLE.print(f"[RTSP] Connecting to {self.config.rtsp_url}...")
        CONSOLE.print(f"[RTSP] Settings: threading={self.use_threading}, gstreamer={self.use_gstreamer}")

        # Open initial connection
        if not self._open_stream():
            raise RuntimeError(f"Failed to open RTSP stream: {self.config.rtsp_url}")

        # Start threaded reader if enabled
        if self.use_threading:
            self._start_reader_thread()
            # Wait a bit for the thread to start reading
            time.sleep(0.5)

        self.start_time = time.time()
        self.frame_count = 0
        consecutive_failures = 0
        max_failures = 100  # Maximum consecutive failures before giving up

        try:
            while True:
                # Get frame based on threading mode
                if self.use_threading:
                    ret, frame = self._get_frame_threaded()
                else:
                    ret, frame = self._get_frame_direct()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        CONSOLE.print(f"[RTSP] Too many failures ({max_failures}), stopping")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                # Skip frames if configured
                if self.config.skip_frames > 0 and not self.use_threading:
                    for _ in range(self.config.skip_frames):
                        self.cap.read()

                # Preprocess frame
                frame = self._preprocess_frame(frame)

                # Create metadata
                metadata = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time() - self.start_time,
                    'source': 'rtsp',
                    'rtsp_url': self.config.rtsp_url,
                    'threaded': self.use_threading,
                }

                self.frame_count += 1
                yield frame, metadata

        finally:
            self.release()

    def release(self):
        """Release RTSP stream."""
        # Stop reader thread first
        if self.use_threading:
            self._stop_reader_thread()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        CONSOLE.print("[RTSP] Stream released")


class HTTPStreamSource(BaseStreamingSource):
    """
    HTTP video stream source (MJPEG, etc.).

    Supports IP Webcam apps and other HTTP-based camera streams.
    Common URL formats:
    - http://ip:port/video (MJPEG stream)
    - http://ip:port/videofeed
    - http://ip:port/mjpegfeed
    """

    def __init__(self, config: StreamingSourceConfig):
        super().__init__(config)
        self.cap = None

    def _get_stream_url(self) -> str:
        """Get the stream URL, trying common endpoints if needed."""
        base_url = self.config.http_url or self.config.rtsp_url

        # If URL already has a path, use it directly
        if '/' in base_url.split('//')[-1].split(':')[-1]:
            return base_url

        # Common IP Webcam endpoints to try
        return base_url

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Stream frames from HTTP source."""
        url = self._get_stream_url()

        if url is None:
            raise ValueError("http_url or rtsp_url must be specified for HTTPStreamSource")

        CONSOLE.print(f"[HTTP] Connecting to {url}...")

        # Try to open the stream
        self.cap = cv2.VideoCapture(url)

        if not self.cap.isOpened():
            # Try common IP Webcam endpoints
            base_url = url.rstrip('/')
            endpoints = ['/video', '/videofeed', '/mjpegfeed', '']

            for endpoint in endpoints:
                test_url = base_url + endpoint
                CONSOLE.print(f"[HTTP] Trying {test_url}...")
                self.cap = cv2.VideoCapture(test_url)
                if self.cap.isOpened():
                    url = test_url
                    break

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open HTTP stream: {url}")

        CONSOLE.print(f"[HTTP] Connected to {url}")

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.start_time = time.time()
        self.frame_count = 0
        consecutive_failures = 0
        max_failures = 50

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        CONSOLE.print(f"[HTTP] Too many failures, stopping")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                # Skip frames if configured
                if self.config.skip_frames > 0:
                    for _ in range(self.config.skip_frames):
                        self.cap.read()

                # Preprocess frame
                frame = self._preprocess_frame(frame)

                # Create metadata
                metadata = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time() - self.start_time,
                    'source': 'http',
                    'http_url': url,
                }

                self.frame_count += 1
                yield frame, metadata

        finally:
            self.release()

    def release(self):
        """Release HTTP stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        CONSOLE.print("[HTTP] Stream released")


class ImageSequenceSource(BaseStreamingSource):
    """Image sequence source for tracking on ordered images."""

    def __init__(self, config: StreamingSourceConfig):
        super().__init__(config)
        self.image_paths = []

    def _discover_images(self) -> List[Path]:
        """
        Discover and sort images in the directory.

        Returns:
            Sorted list of image file paths
        """
        image_dir = Path(self.config.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not image_dir.is_dir():
            raise ValueError(f"Path is not a directory: {image_dir}")

        # Find all image files
        extensions = self.config.image_extensions
        image_files = []

        for ext in extensions:
            # Support both lowercase and uppercase extensions
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))

        # Remove duplicates (in case of case-insensitive filesystems)
        image_files = list(set(image_files))

        if not image_files:
            raise ValueError(f"No images found in {image_dir} with extensions {extensions}")

        # Sort images based on config
        sort_by = self.config.sort_by
        if sort_by == 'name':
            # Simple alphabetical sort
            image_files.sort(key=lambda p: p.name)
        elif sort_by == 'natural':
            # Natural sort (handles numbers correctly: 1, 2, 10 instead of 1, 10, 2)
            image_files.sort(key=lambda p: natural_sort_key(p.name))
        elif sort_by == 'mtime':
            # Sort by modification time
            image_files.sort(key=lambda p: p.stat().st_mtime)
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}")

        return image_files

    def _get_frame_segments(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Get validated frame segments to process.

        Args:
            total_frames: Total number of images

        Returns:
            List of (start, end) tuples with validated frame indices
        """
        segments = self.config.frame_segments

        # If no segments specified, process all frames
        if segments is None or len(segments) == 0:
            return [(0, total_frames)]

        # Validate and clip segments
        validated = []
        for start, end in segments:
            # Clip start to valid range
            start = max(0, min(start, total_frames - 1))

            # Handle None end (until end)
            if end is None:
                end = total_frames
            else:
                end = max(start + 1, min(end, total_frames))

            # Only add if segment has frames
            if end > start:
                validated.append((start, end))

        # Sort by start frame and merge overlapping segments
        if not validated:
            return [(0, total_frames)]

        validated.sort(key=lambda x: x[0])

        # Merge overlapping segments
        merged = [validated[0]]
        for start, end in validated[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # Overlapping or adjacent
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Stream frames from image sequence."""
        if self.config.image_dir is None:
            raise ValueError("image_dir must be specified for ImageSequenceSource")

        # Discover and sort images
        self.image_paths = self._discover_images()
        total_images = len(self.image_paths)

        CONSOLE.print(f"Found {total_images} images in {self.config.image_dir}")
        CONSOLE.print(f"First image: {self.image_paths[0].name}")
        CONSOLE.print(f"Last image: {self.image_paths[-1].name}")

        # Get validated frame segments
        frame_segments = self._get_frame_segments(total_images)

        # Calculate total frames to process
        total_to_process = sum(end - start for start, end in frame_segments)
        CONSOLE.print(f"Frame segments: {frame_segments}")
        CONSOLE.print(f"Total images to process: {total_to_process}")

        # Calculate frame delay for real-time simulation
        if self.config.simulate_realtime:
            frame_delay = 1.0 / self.config.target_fps
        else:
            frame_delay = 0

        self.start_time = time.time()
        self.frame_count = 0
        last_frame_time = time.time()

        try:
            # Iterate through each segment
            for segment_idx, (seg_start, seg_end) in enumerate(frame_segments):
                CONSOLE.print(f"Processing segment {segment_idx + 1}/{len(frame_segments)}: images {seg_start}-{seg_end}")

                current_idx = seg_start
                while current_idx < seg_end:
                    image_path = self.image_paths[current_idx]

                    # Read image
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        CONSOLE.print(f"Warning: Failed to read image: {image_path}")
                        current_idx += 1
                        continue

                    # Skip frames if configured
                    if self.config.skip_frames > 0:
                        current_idx += self.config.skip_frames

                    # Simulate real-time playback
                    if self.config.simulate_realtime:
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        if elapsed < frame_delay:
                            time.sleep(frame_delay - elapsed)
                        last_frame_time = time.time()

                    # Preprocess frame
                    frame = self._preprocess_frame(frame)

                    # Create metadata
                    metadata = {
                        'frame_id': self.frame_count,
                        'original_idx': current_idx,
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'timestamp': time.time() - self.start_time,
                        'source': 'image_sequence',
                        'image_dir': self.config.image_dir,
                        'total_images': total_images,
                        'segment_idx': segment_idx,
                        'segment_start': seg_start,
                        'segment_end': seg_end,
                        'frame_segments': frame_segments,
                    }

                    self.frame_count += 1
                    current_idx += 1
                    yield frame, metadata

        finally:
            self.release()

    def release(self):
        """Release resources (nothing to release for image sequence)."""
        self.image_paths = []


class StreamingSourceFactory:
    """Factory for creating streaming sources."""

    @staticmethod
    def create(config: StreamingSourceConfig) -> BaseStreamingSource:
        """
        Create streaming source based on config.

        Args:
            config: StreamingSourceConfig instance

        Returns:
            Streaming source instance
        """
        sources = {
            'camera': CameraSource,
            'video_file': VideoFileSource,
            'rtsp': RTSPSource,
            'http': HTTPStreamSource,
            'image_sequence': ImageSequenceSource,
        }

        source_class = sources.get(config.source_type)
        if source_class is None:
            raise ValueError(f"Unknown source type: {config.source_type}")

        return source_class(config)

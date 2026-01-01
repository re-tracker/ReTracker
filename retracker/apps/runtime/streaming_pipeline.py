"""Streaming tracking pipeline for real-time inference."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List

import numpy as np

from ..config import StreamingConfig
from retracker.utils.rich_utils import CONSOLE

if TYPE_CHECKING:
    import torch


def _require_cv2():  # noqa: ANN202
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "OpenCV (cv2) is required for streaming. Ensure `opencv-python` is installed and compatible."
        ) from exc
    return cv2


def _require_imageio():  # noqa: ANN202
    try:
        import imageio  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "imageio is required for --record. Install `retracker[apps]` (or `imageio[ffmpeg]`)."
        ) from exc
    return imageio


class StreamingTrackingPipeline:
    """Real-time streaming tracking pipeline."""

    def __init__(self, config: StreamingConfig):
        """
        Initialize streaming pipeline.

        Args:
            config: StreamingConfig instance
        """
        self.config = config

        # Initialize components
        from .streaming_source import StreamingSourceFactory
        from .query_generator import QueryGeneratorFactory
        from .tracker import Tracker

        self.source = StreamingSourceFactory.create(config.source)
        self.query_generator = QueryGeneratorFactory.create(config.query)
        self.tracker = Tracker(config.model)

        # Frame buffer for accumulating frames
        self.frame_buffer = []
        self.all_frames = []  # Store all frames for global tracking

        # Query points (initialized on first frame)
        self.queries = None
        self.query_frame_id = 0

        # Tracking results (global)
        self.trajectories = None
        self.visibility = None
        self.dense_tracks = None  # Dense matching results (if enabled)
        self.dense_visibility = None

        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.processing_times = []

        # Output video writer
        self.video_writer = None

        # Create output directory
        if config.output.output_dir:
            Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_queries(
        self,
        first_frame: np.ndarray,
        device: str
    ) -> "torch.Tensor":
        """Initialize query points from first frame."""
        import torch

        # Convert frame to tensor (1, 1, C, H, W)
        frame_tensor = torch.from_numpy(first_frame).to(device)
        frame_tensor = frame_tensor.permute(2, 0, 1).float()
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)

        # Generate queries with initial_frame parameter
        queries = self.query_generator.generate(
            frame_tensor,
            seg_mask=None,
            initial_frame=self.config.query.initial_frame
        )

        # Limit number of points if configured
        max_points = self.config.processing.max_points
        if queries.shape[1] > max_points:
            # Random sampling to preserve spatial uniformity
            # (linspace sampling creates patterns due to row-major grid order)
            perm = torch.randperm(queries.shape[1])[:max_points]
            perm = perm.sort().values  # Sort to maintain some order
            queries = queries[:, perm]

        if self.config.verbose:
            CONSOLE.print(f"[green]Initialized {queries.shape[1]} query points")
            CONSOLE.print(f"[green]Query shape: {queries.shape}")
            CONSOLE.print(f"[green]Query range - t: [{queries[0,:,0].min():.1f}, {queries[0,:,0].max():.1f}], x: [{queries[0,:,1].min():.1f}, {queries[0,:,1].max():.1f}], y: [{queries[0,:,2].min():.1f}, {queries[0,:,2].max():.1f}]")

        return queries

    def _process_accumulated_frames(self, device: str) -> Optional[tuple]:
        """Process all accumulated frames with global tracking."""
        import torch

        if len(self.all_frames) == 0:
            return None

        # Stack all frames into video tensor (1, T, C, H, W)
        # Keep on CPU - engine.video_forward will handle GPU transfer
        video_tensor = torch.cat(self.all_frames, dim=1)

        if self.config.verbose:
            CONSOLE.print(f"[cyan]Running tracking on {video_tensor.shape[1]} frames...")

        # Run tracking with queries from first frame
        with torch.no_grad():
            result = self.tracker.track(
                video_tensor,
                self.queries,
                use_aug=False
            )

        # Clean up to free memory
        del video_tensor
        self.all_frames.clear()
        torch.cuda.empty_cache()

        # Handle both dense and non-dense returns
        if len(result) == 4:
            # Dense matching: (traj, vis, dense_tracks, dense_vis)
            trajectories, visibility, dense_tracks, dense_vis = result
            # Store dense results for potential future use
            self.dense_tracks = dense_tracks
            self.dense_visibility = dense_vis
            return trajectories, visibility
        else:
            # Standard: (traj, vis)
            trajectories, visibility = result
            return trajectories, visibility

    def _visualize_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        trajectories: Optional["torch.Tensor"],
        visibility: Optional["torch.Tensor"],
        metadata: Dict[str, Any],
        fps: float
    ) -> np.ndarray:
        """Visualize tracking results on frame."""
        cv2 = _require_cv2()
        vis_frame = frame.copy()

        # Draw FPS counter
        if self.config.visualization.display_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                vis_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        # Draw frame info
        if self.config.visualization.display_info:
            info_text = f"Frame: {metadata['frame_id']}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Draw tracking points
        if trajectories is not None and visibility is not None and frame_idx < trajectories.shape[1]:
            # Get current frame trajectories
            current_traj = trajectories[0, frame_idx].cpu().numpy()  # (N, 2)
            current_vis = visibility[0, frame_idx].cpu().numpy()  # (N,)

            num_high_conf = (current_vis > 0.5).sum()
            num_low_conf = (current_vis <= 0.5).sum()

            # Draw info about tracked points
            if self.config.visualization.display_info:
                if self.dense_tracks is not None:
                    # Show dense point count: N queries * 49 patch points
                    dense_count = num_high_conf * 49
                    points_text = f"Points: {num_high_conf}x49={dense_count} (Dense)"
                else:
                    points_text = f"Points: {num_high_conf}H/{num_low_conf}L/{len(current_vis)}T"
                cv2.putText(
                    vis_frame,
                    points_text,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            # Draw trajectories (trace) first
            if self.config.visualization.tracks_leave_trace > 1:
                trace_start = max(0, frame_idx - self.config.visualization.tracks_leave_trace + 1)
                max_motion = self.config.visualization.max_motion_threshold

                for i in range(trajectories.shape[2]):  # For each point
                    points = []
                    for t in range(trace_start, frame_idx + 1):
                        if visibility[0, t, i] > 0.5:
                            pt = trajectories[0, t, i].cpu().numpy()
                            # Skip invalid points (NaN or Inf)
                            if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                                continue
                            x, y = int(pt[0]), int(pt[1])
                            # Check bounds
                            if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                                points.append((x, y, t))  # Store time index too

                    if len(points) > 1:
                        color = self._get_point_color(i, trajectories.shape[2])
                        for j in range(len(points) - 1):
                            p1 = points[j]
                            p2 = points[j + 1]

                            # Check motion threshold
                            if max_motion > 0:
                                dx = p2[0] - p1[0]
                                dy = p2[1] - p1[1]
                                motion = (dx * dx + dy * dy) ** 0.5
                                if motion > max_motion:
                                    # Skip drawing this line segment (large motion)
                                    continue

                            cv2.line(
                                vis_frame,
                                (p1[0], p1[1]),
                                (p2[0], p2[1]),
                                color,
                                self.config.visualization.linewidth
                            )

            # Draw current points on top
            for i, (point, vis) in enumerate(zip(current_traj, current_vis)):
                # Skip invalid points (NaN or Inf)
                if not np.isfinite(point[0]) or not np.isfinite(point[1]):
                    continue

                x, y = int(point[0]), int(point[1])
                # Ensure point is within frame bounds
                if 0 <= x < vis_frame.shape[1] and 0 <= y < vis_frame.shape[0]:
                    color = self._get_point_color(i, len(current_traj))

                    if vis > 0.5:  # High confidence - filled circle
                        cv2.circle(
                            vis_frame,
                            (x, y),
                            self.config.visualization.point_radius,
                            color,
                            -1  # Filled
                        )
                    elif self.config.visualization.show_low_confidence:
                        # Low confidence - hollow circle (only if show_low_confidence is True)
                        cv2.circle(
                            vis_frame,
                            (x, y),
                            self.config.visualization.point_radius,
                            color,
                            1  # Hollow (thickness=1)
                        )

            # Draw dense matching points if available
            if self.dense_tracks is not None and frame_idx < self.dense_tracks.shape[1]:
                # dense_tracks shape: (B, T, N, 49, 2)
                dense_traj = self.dense_tracks[0, frame_idx].cpu().numpy()  # (N, 49, 2)

                dense_points_drawn = 0

                for i in range(dense_traj.shape[0]):  # For each query point
                    # Skip if center point is not visible
                    if current_vis[i] <= 0.5:
                        continue

                    color = self._get_point_color(i, len(current_traj))

                    # Draw 49 patch points
                    for j in range(dense_traj.shape[1]):  # 49 patch points
                        # Skip center point (index 24 in 7x7 = already drawn above)
                        if j == 24:
                            continue

                        pt = dense_traj[i, j]  # (2,)
                        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                            continue

                        px, py = int(pt[0]), int(pt[1])
                        if 0 <= px < vis_frame.shape[1] and 0 <= py < vis_frame.shape[0]:
                            dense_points_drawn += 1
                            # Draw smaller dots for dense points
                            cv2.circle(
                                vis_frame,
                                (px, py),
                                max(1, self.config.visualization.point_radius - 1),
                                color,
                                -1  # Filled
                            )

                # Debug output for first frame
                if frame_idx == 0:
                    CONSOLE.print(f"[dim][Debug Vis] Frame {frame_idx}: Drew {dense_points_drawn} dense points[/dim]")
                    CONSOLE.print(
                        f"[dim][Debug Vis] dense_traj[0] all 49 points range: "
                        f"x=[{dense_traj[0,:,0].min():.1f}, {dense_traj[0,:,0].max():.1f}], "
                        f"y=[{dense_traj[0,:,1].min():.1f}, {dense_traj[0,:,1].max():.1f}][/dim]"
                    )
            elif frame_idx == 0:
                CONSOLE.print(
                    f"[dim][Debug Vis] Frame {frame_idx}: dense_tracks is None: {self.dense_tracks is None}[/dim]"
                )

        return vis_frame

    def _visualize_pairs_frame(
        self,
        first_frame: np.ndarray,
        current_frame: np.ndarray,
        frame_idx: int,
        trajectories: Optional["torch.Tensor"],
        visibility: Optional["torch.Tensor"],
        metadata: Dict[str, Any],
    ) -> np.ndarray:
        """
        Visualize matching pairs between first frame and current frame.

        Creates a side-by-side visualization with matching lines.

        Args:
            first_frame: The reference frame (query frame)
            current_frame: The current frame being matched
            frame_idx: Current frame index
            trajectories: Tracking trajectories (1, T, N, 2)
            visibility: Visibility scores (1, T, N)
            metadata: Frame metadata

        Returns:
            Visualization image with side-by-side frames and matching lines
        """
        cv2 = _require_cv2()
        H, W = first_frame.shape[:2]

        # Create side-by-side canvas
        canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
        canvas[:, :W] = first_frame
        canvas[:, W:] = current_frame

        # Draw frame labels
        if self.config.visualization.display_info:
            # Left frame label
            cv2.putText(
                canvas,
                f"Frame 0 (Query)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            # Right frame label
            cv2.putText(
                canvas,
                f"Frame {frame_idx}",
                (W + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Draw matching lines and points
        if trajectories is not None and visibility is not None and frame_idx < trajectories.shape[1]:
            # Get first frame positions (queries)
            first_traj = trajectories[0, 0].cpu().numpy()  # (N, 2)
            first_vis = visibility[0, 0].cpu().numpy()  # (N,)

            # Get current frame positions
            current_traj = trajectories[0, frame_idx].cpu().numpy()  # (N, 2)
            current_vis = visibility[0, frame_idx].cpu().numpy()  # (N,)

            # Count matches
            show_low_conf = self.config.visualization.show_low_confidence
            if show_low_conf:
                valid_mask = (first_vis > 0) & (current_vis > 0)
            else:
                valid_mask = (first_vis > 0.5) & (current_vis > 0.5)

            num_valid = valid_mask.sum()

            # Draw info
            if self.config.visualization.display_info:
                info_text = f"Matches: {num_valid}/{len(first_vis)}"
                cv2.putText(
                    canvas,
                    info_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            # Draw matching lines and points
            max_motion = self.config.visualization.max_motion_threshold

            for i in range(len(first_traj)):
                vis_first = first_vis[i]
                vis_curr = current_vis[i]

                # Skip if either point is not visible (based on show_low_confidence)
                if show_low_conf:
                    if vis_first <= 0 or vis_curr <= 0:
                        continue
                else:
                    if vis_first <= 0.5 or vis_curr <= 0.5:
                        continue

                # Get positions - check for NaN/Inf values first
                pt1 = first_traj[i]
                pt2 = current_traj[i]
                if not (np.isfinite(pt1[0]) and np.isfinite(pt1[1]) and
                        np.isfinite(pt2[0]) and np.isfinite(pt2[1])):
                    continue
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])

                # Offset x2 for right panel
                x2_canvas = x2 + W

                # Check bounds
                if not (0 <= x1 < W and 0 <= y1 < H):
                    continue
                if not (0 <= x2 < W and 0 <= y2 < H):
                    continue

                # Get color for this point
                color = self._get_point_color(i, len(first_traj))

                # Determine if this is a high confidence match
                is_high_conf = vis_first > 0.5 and vis_curr > 0.5

                # Check motion threshold
                dx = x2 - x1
                dy = y2 - y1
                motion = (dx * dx + dy * dy) ** 0.5
                draw_line = (max_motion <= 0) or (motion <= max_motion)

                # Draw matching line (only if enabled and motion within threshold)
                if self.config.visualization.show_matching_lines and draw_line:
                    line_thickness = self.config.visualization.linewidth if is_high_conf else 1
                    cv2.line(
                        canvas,
                        (x1, y1),
                        (x2_canvas, y2),
                        color,
                        line_thickness
                    )

                # Always draw points on both frames
                point_radius = self.config.visualization.point_radius
                if is_high_conf:
                    # Filled circles for high confidence
                    cv2.circle(canvas, (x1, y1), point_radius, color, -1)
                    cv2.circle(canvas, (x2_canvas, y2), point_radius, color, -1)
                else:
                    # Hollow circles for low confidence
                    cv2.circle(canvas, (x1, y1), point_radius, color, 1)
                    cv2.circle(canvas, (x2_canvas, y2), point_radius, color, 1)

        return canvas

    def _get_point_color(self, idx: int, total: int) -> tuple:
        """Get color for point based on index."""
        cv2 = _require_cv2()
        # Use HSV colormap
        hue = int(180 * idx / max(total, 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(map(int, color_bgr))

    def _init_video_writer(self, frame_shape: tuple, is_pairs_mode: bool = False):
        """Initialize video writer for recording."""
        if not self.config.visualization.record_output:
            return

        imageio = _require_imageio()

        output_path = self.config.visualization.output_path
        if output_path is None:
            output_path = str(
                Path(self.config.output.output_dir) / "streaming_output.mp4"
            )

        # Use imageio instead of cv2.VideoWriter for better compatibility
        fps = self.config.source.target_fps
        self.video_writer = imageio.get_writer(output_path, fps=fps)

        if self.config.verbose:
            CONSOLE.print(f"[yellow]Recording to: {output_path}")

    def run(self):
        """Run streaming tracking pipeline."""
        cv2 = _require_cv2()
        import torch

        if self.config.verbose:
            CONSOLE.print("\n[bold cyan]" + "="*60)
            CONSOLE.print("[bold cyan]Starting Streaming Tracking Pipeline")
            CONSOLE.print("[bold cyan]" + "="*60 + "\n")
            CONSOLE.print(f"[yellow]Source: {self.config.source.source_type}")
            CONSOLE.print(f"[yellow]Processing mode: Global tracking (all frames)")
            CONSOLE.print(f"[yellow]Device: {self.config.model.device}\n")

        device = self.config.model.device
        frame_count = 0
        start_time = time.time()

        # Phase 1: Collect all frames (keep on CPU to save GPU memory)
        CONSOLE.print("[bold green]Phase 1: Collecting frames...")
        try:
            for frame, metadata in self.source:
                frame_start = time.time()

                # Initialize queries on first frame
                if self.queries is None:
                    self.queries = self._initialize_queries(frame, device)
                    self.query_frame_id = metadata['frame_id']
                    is_pairs_mode = self.config.visualization.plot_mode == 'pairs'
                    self._init_video_writer(frame.shape, is_pairs_mode=is_pairs_mode)

                # Store frame
                self.frame_buffer.append(frame)

                # Convert to tensor but keep on CPU to save GPU memory
                frame_tensor = torch.from_numpy(frame).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
                self.all_frames.append(frame_tensor)  # Keep on CPU

                frame_count += 1

                # Print progress
                if self.config.verbose and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    CONSOLE.print(f"[cyan]Collected {frame_count} frames | Elapsed: {elapsed:.1f}s")

                # Check max duration
                if self.config.max_duration is not None:
                    if time.time() - start_time > self.config.max_duration:
                        CONSOLE.print("\n[yellow]Max duration reached")
                        break

        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Interrupted by user")

        if frame_count == 0:
            CONSOLE.print("[red]No frames collected!")
            return

        # Phase 2: Run tracking on all frames
        CONSOLE.print(f"\n[bold green]Phase 2: Running tracking on {frame_count} frames...")
        tracking_start = time.time()

        result = self._process_accumulated_frames(device)
        if result is not None:
            self.trajectories, self.visibility = result
            tracking_time = time.time() - tracking_start

            CONSOLE.print(f"[green]Tracking completed in {tracking_time:.2f}s")
            CONSOLE.print(f"[green]Trajectories shape: {self.trajectories.shape}")
            CONSOLE.print(f"[green]FPS: {frame_count/tracking_time:.1f}")

            # Debug: Check dense tracks
            if self.dense_tracks is not None:
                CONSOLE.print(f"[green]Dense tracks shape: {self.dense_tracks.shape}")
                # Check if dense tracks have valid data
                sample = self.dense_tracks[0, 0, 0]  # First frame, first query
                CONSOLE.print(f"[green]Dense sample (first query, first frame): min={sample.min():.2f}, max={sample.max():.2f}")
            else:
                CONSOLE.print(f"[yellow]Dense tracks: None")

        # Phase 3: Visualize and save
        CONSOLE.print(f"\n[bold green]Phase 3: Generating visualization...")
        CONSOLE.print(f"[yellow]Plot mode: {self.config.visualization.plot_mode}")
        vis_start = time.time()

        is_pairs_mode = self.config.visualization.plot_mode == 'pairs'
        first_frame = self.frame_buffer[0] if len(self.frame_buffer) > 0 else None

        for idx, frame in enumerate(self.frame_buffer):
            # Calculate FPS for display
            avg_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

            # Visualize frame based on plot mode
            if is_pairs_mode and first_frame is not None:
                vis_frame = self._visualize_pairs_frame(
                    first_frame,
                    frame,
                    idx,
                    self.trajectories,
                    self.visibility,
                    {'frame_id': idx},
                )
            else:
                vis_frame = self._visualize_frame(
                    frame,
                    idx,
                    self.trajectories,
                    self.visibility,
                    {'frame_id': idx},
                    avg_fps
                )

            # Show live window
            if self.config.visualization.show_live:
                cv2.imshow(
                    self.config.visualization.window_name,
                    cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                )

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    CONSOLE.print("\n[yellow]User requested quit")
                    break

            # Record to video
            if self.video_writer is not None:
                self.video_writer.append_data(vis_frame)

            # Print progress
            if self.config.verbose and (idx + 1) % 30 == 0:
                CONSOLE.print(f"[cyan]Visualized {idx + 1}/{frame_count} frames")

        vis_time = time.time() - vis_start
        CONSOLE.print(f"[green]Visualization completed in {vis_time:.2f}s")

        # Cleanup
        self._cleanup()

        # Print summary
        total_time = time.time() - start_time
        if self.config.verbose:
            CONSOLE.print("\n[bold cyan]" + "="*60)
            CONSOLE.print("[bold green]Streaming completed")
            CONSOLE.print("[bold cyan]" + "="*60)
            CONSOLE.print(f"Total frames: {frame_count}")
            CONSOLE.print(f"Collection time: {tracking_start - start_time:.2f}s")
            CONSOLE.print(f"Tracking time: {tracking_time:.2f}s")
            CONSOLE.print(f"Visualization time: {vis_time:.2f}s")
            CONSOLE.print(f"Total time: {total_time:.2f}s")
            CONSOLE.print(f"Overall FPS: {frame_count/total_time:.1f}")
            CONSOLE.print("[bold cyan]" + "="*60 + "\n")

    def _cleanup(self):
        """Clean up resources."""
        # Release source
        self.source.release()

        # Close video writer
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Close windows
        if self.config.visualization.show_live:
            cv2 = _require_cv2()
            cv2.destroyAllWindows()

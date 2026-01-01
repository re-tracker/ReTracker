"""Complete tracking pipeline orchestration."""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from ..config import TrackingConfig
from .video_loader import VideoLoaderFactory
from .query_generator import QueryGeneratorFactory
from .tracker import Tracker
from retracker.utils.rich_utils import CONSOLE
from ..utils import detect_video_rotation


class TrackingPipeline:
    """Complete tracking pipeline orchestration."""
    
    def __init__(self, config: TrackingConfig):
        """
        Initialize pipeline.
        
        Args:
            config: TrackingConfig instance
        """
        self.config = config
        
        # Initialize components
        self.video_loader = VideoLoaderFactory.create(config.video)
        self.query_generator = QueryGeneratorFactory.create(config.query)
        self.tracker = Tracker(config.model)
        self.visualizer = self._build_visualizer() if config.output.save_video else None
        
        # Create output directory
        Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _build_visualizer(self):
        """Initialize visualizer (lazy import to keep CLI startup fast)."""
        from retracker.visualization.visualizer import Visualizer

        return Visualizer(
            save_dir=self.config.output.output_dir,
            fps=self.config.visualization.fps,
            linewidth=self.config.visualization.linewidth,
            tracks_leave_trace=self.config.visualization.tracks_leave_trace,
        )
    
    def _load_segmentation_mask(
        self, 
        seg_path: str
    ) -> Optional[torch.Tensor]:
        """Load segmentation mask."""
        if seg_path is None:
            return None

        import cv2
        
        CONSOLE.print(f"[yellow]Loading segmentation mask: {seg_path}")
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        if seg_mask is None:
            CONSOLE.print(f"[red]Failed to load mask: {seg_path}")
            return None
        
        # Resize to match video dimensions
        W, H = self.config.video.resized_wh
        seg_mask = cv2.resize(seg_mask, (W, H))
        
        # Convert to tensor (1, 1, H, W)
        seg_mask = torch.from_numpy(seg_mask)[None, None].float()
        
        return seg_mask

    def _load_visualization_video(
        self,
        video_path: str,
        resized_wh: tuple[int, int],
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Load a CPU uint8 video tensor for visualization only."""
        import cv2

        W, H = resized_wh

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        _, needs_180_rotation = detect_video_rotation(video_path) if self.config.video.auto_rotate else (0, False)

        frames = []
        while True:
            if num_frames is not None and len(frames) >= num_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if needs_180_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise RuntimeError(f"No frames extracted from video: {video_path}")

        video_np = torch.from_numpy(
            # (T, H, W, C) uint8
            np.stack(frames, axis=0)
        )
        # (1, T, C, H, W) uint8
        return video_np.permute(0, 3, 1, 2).unsqueeze(0).contiguous()

    @staticmethod
    def _scale_trajectories(
        trajectories: torch.Tensor,
        src_wh: tuple[int, int],
        dst_wh: tuple[int, int],
    ) -> torch.Tensor:
        """Scale (B, T, N, 2) trajectories from src (W,H) to dst (W,H)."""
        src_w, src_h = src_wh
        dst_w, dst_h = dst_wh

        if src_w <= 1 or src_h <= 1:
            return trajectories

        scale_x = (dst_w - 1) / float(src_w - 1)
        scale_y = (dst_h - 1) / float(src_h - 1)

        scaled = trajectories.clone()
        scaled[..., 0] *= scale_x
        scaled[..., 1] *= scale_y
        scaled[..., 0].clamp_(0, dst_w - 1)
        scaled[..., 1].clamp_(0, dst_h - 1)
        return scaled
    
    def run(
        self, 
        video_path: Optional[str] = None,
        seg_path: Optional[str] = None,
        use_aug: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete tracking pipeline.
        
        Args:
            video_path: Path to video (overrides config)
            seg_path: Path to segmentation mask (overrides config)
            use_aug: Use test-time augmentation
            
        Returns:
            results: Dictionary with tracking results and metrics
        """
        # Use provided paths or fall back to config
        video_path = video_path or self.config.video_path
        seg_path = seg_path or self.config.seg_path
        
        if video_path is None:
            raise ValueError("video_path must be provided")
        
        CONSOLE.print("\n[bold cyan]" + "="*60)
        CONSOLE.print("[bold cyan]Starting Tracking Pipeline")
        CONSOLE.print("[bold cyan]" + "="*60 + "\n")

        # Reset CUDA peak stats (per-process) so runs are comparable.
        if torch.cuda.is_available() and str(self.config.model.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        
        # Step 1: Load video
        CONSOLE.print("[bold green][1/5] Loading video...")
        CONSOLE.print(f"  Path: {video_path}")
        t_start = time.time()
        video = self.video_loader.load(
            video_path, 
            device=self.config.model.device
        )
        t_load = time.time() - t_start
        CONSOLE.print(f"  Shape: {tuple(video.shape)}")
        CONSOLE.print(f"  Time: {t_load:.2f}s\n")
        
        # Step 2: Load segmentation mask if provided
        seg_mask = self._load_segmentation_mask(seg_path)
        if seg_mask is not None:
            CONSOLE.print(f"  Mask shape: {tuple(seg_mask.shape)}\n")
        
        # Step 3: Generate queries
        CONSOLE.print("[bold green][2/5] Generating query points...")
        CONSOLE.print(f"  Strategy: {self.config.query.strategy}")
        t_start = time.time()
        queries = self.query_generator.generate(
            video, 
            seg_mask=seg_mask,
            initial_frame=self.config.query.initial_frame
        )
        t_query = time.time() - t_start
        CONSOLE.print(f"  Generated: {queries.shape[1]} points")
        CONSOLE.print(f"  Time: {t_query:.3f}s\n")
        
        # Step 4: Perform tracking
        CONSOLE.print("[bold green][3/5] Running tracking...")
        engine_display = getattr(self.tracker.engine, "display_name", self.tracker.engine.__class__.__name__)
        CONSOLE.print(f"  Engine: {engine_display}")
        CONSOLE.print(f"  Device: {self.config.model.device}")
        CONSOLE.print(f"  Dtype: {self.config.model.dtype}")
        t_start = time.time()
        trajectories, visibility = self.tracker.track(
            video, 
            queries, 
            use_aug=use_aug
        )
        t_track = time.time() - t_start
        CONSOLE.print(f"  Trajectories shape: {tuple(trajectories.shape)}")
        CONSOLE.print(f"  Time: {t_track:.2f}s")
        CONSOLE.print(f"  FPS: {video.shape[1] / t_track:.1f}\n")
        
        # Step 5: Visualization
        if self.config.output.save_video:
            if self.visualizer is None:
                self.visualizer = self._build_visualizer()

            CONSOLE.print("[bold green][4/5] Generating visualization...")
            video_name = Path(video_path).stem

            tracker_name = "retracker"
            vis_wh = self.config.visualization.vis_resized_wh
            output_filename = f"{video_name}_{tracker_name}"
            if vis_wh is not None:
                output_filename = f"{output_filename}_vis{vis_wh[0]}x{vis_wh[1]}"

            vis_video = video.cpu()
            vis_trajectories = trajectories.cpu()
            vis_seg_mask = seg_mask

            if vis_wh is not None:
                # Reload video at visualization resolution on CPU to avoid GPU OOM.
                CONSOLE.print(f"  Visualization resize: {vis_wh[0]}x{vis_wh[1]}")
                vis_video = self._load_visualization_video(
                    video_path,
                    resized_wh=vis_wh,
                    num_frames=video.shape[1],
                )
                vis_trajectories = self._scale_trajectories(
                    trajectories.cpu(),
                    src_wh=self.config.video.resized_wh,
                    dst_wh=vis_wh,
                )
                if vis_seg_mask is not None:
                    vis_seg_mask = F.interpolate(
                        vis_seg_mask,
                        size=(vis_wh[1], vis_wh[0]),
                        mode="nearest",
                    )

            self.visualizer.visualize(
                vis_video,
                vis_trajectories,
                visibility=visibility.cpu(),
                segm_mask=vis_seg_mask,
                filename=output_filename,
                hide_occ_points=self.config.visualization.hide_occ_points,
            )
            CONSOLE.print(f"  Saved: {output_filename}.mp4\n")
        
        # Step 6: Save results
        CONSOLE.print("[bold green][5/5] Saving results...")
        results = self._save_results(
            video_path=video_path,
            video=video,
            trajectories=trajectories,
            visibility=visibility,
            metrics={
                'loading_time': t_load,
                'query_time': t_query,
                'tracking_time': t_track,
                'total_time': t_load + t_query + t_track,
            }
        )

        # CUDA peak memory stats.
        if torch.cuda.is_available() and str(self.config.model.device).startswith("cuda"):
            torch.cuda.synchronize()
            results["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
            results["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024**2)
        
        CONSOLE.print("\n[bold cyan]" + "="*60)
        CONSOLE.print("[bold green]Pipeline completed successfully.")
        CONSOLE.print("[bold cyan]" + "="*60 + "\n")
        
        return results
    
    def _save_results(
        self, 
        video_path: str,
        video: Optional[torch.Tensor],
        trajectories: torch.Tensor,
        visibility: torch.Tensor,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Save results in various formats."""
        video_name = Path(video_path).stem
        tracker_name = "retracker"
        vis_wh = self.config.visualization.vis_resized_wh
        seq_name = f"{video_name}_{tracker_name}"
        if vis_wh is not None:
            seq_name = f"{seq_name}_vis{vis_wh[0]}x{vis_wh[1]}"

        # Save NPZ/images if requested
        if self.config.output.save_npz or self.config.output.save_images:
            if self.config.output.save_images and video is None:
                raise ValueError("save_images=True requires `video` to be provided.")
            from retracker.io.results import ResultsDumper

            ResultsDumper(
                save_dir=self.config.output.output_dir,
                dump_npz=self.config.output.save_npz,
                dump_images=self.config.output.save_images,
            ).dump(seq_name, trajectories, visibility, video=video)
            if self.config.output.save_npz:
                CONSOLE.print(f"  NPZ: {seq_name}.npz")
            if self.config.output.save_images:
                CONSOLE.print(f"  Images: {seq_name}/ directory")
        
        # Compile results
        results = {
            'video_path': video_path,
            'output_dir': self.config.output.output_dir,
            'trajectories': trajectories,
            'visibility': visibility,
            'num_points': trajectories.shape[2],
            'num_frames': trajectories.shape[1],
            **metrics,
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print tracking summary."""
        CONSOLE.print("\n[bold]TRACKING SUMMARY[/bold]")
        CONSOLE.print("─" * 50)
        CONSOLE.print(f"Video: {results['video_path']}")
        CONSOLE.print(f"Tracked points: {results['num_points']}")
        CONSOLE.print(f"Frames: {results['num_frames']}")
        CONSOLE.print(f"Tracking time: {results['tracking_time']:.2f}s")
        CONSOLE.print(f"FPS: {results['num_frames']/results['tracking_time']:.1f}")
        CONSOLE.print(f"Total time: {results['total_time']:.2f}s")
        if "gpu_peak_reserved_mib" in results:
            alloc = results.get("gpu_peak_allocated_mib", 0.0)
            resv = results.get("gpu_peak_reserved_mib", 0.0)
            CONSOLE.print(f"Peak GPU memory: {alloc:.0f} MiB alloc, {resv:.0f} MiB reserved")
        CONSOLE.print(f"Output: {results['output_dir']}")
        CONSOLE.print("─" * 50)

    # ========== Profiling Methods ==========

    def enable_profiling(self) -> None:
        """Enable profiling for inference bottleneck analysis."""
        self.tracker.enable_profiling()
        self._profiling_enabled = True

    def disable_profiling(self) -> None:
        """Disable profiling."""
        self.tracker.disable_profiling()
        self._profiling_enabled = False

    def print_profile_summary(self) -> None:
        """Print profiling summary."""
        self.tracker.print_profile_summary()

    def export_profile_csv(self, filepath: str) -> None:
        """Export profiling results to CSV file."""
        self.tracker.export_profile_csv(filepath)

    # ========== torch.compile Methods ==========

    def compile_model(self, mode: str = "reduce-overhead") -> None:
        """Compile the model using torch.compile for faster inference."""
        self.tracker.compile_model(mode=mode)

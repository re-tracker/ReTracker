#!/usr/bin/env python3
"""
Multi-view triangulation pipeline.

This pipeline:
1. Loads camera intrinsics and extrinsics
2. Runs multi-view tracking to get 2D correspondences
3. Triangulates 2D points to 3D using robust triangulation
4. Renders point clouds as video

Each step supports saving/loading intermediate results for debugging.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

from rich.console import Console

CONSOLE = Console()


@dataclass
class CameraParams:
    """Camera parameters for a single view."""
    view_id: str
    K: np.ndarray  # 3x3 intrinsic matrix
    R: np.ndarray  # 3x3 rotation matrix
    T: np.ndarray  # 3x1 translation vector
    D: np.ndarray  # distortion coefficients
    H: int  # image height
    W: int  # image width

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix P = K @ [R | T]"""
        RT = np.hstack([self.R, self.T.reshape(3, 1)])
        return self.K @ RT

    @property
    def extrinsic(self) -> np.ndarray:
        """4x4 extrinsic matrix [R | T; 0 0 0 1]"""
        ext = np.eye(4)
        ext[:3, :3] = self.R
        ext[:3, 3] = self.T.flatten()
        return ext

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T @ T"""
        return -self.R.T @ self.T.flatten()


@dataclass
class TriangulationConfig:
    """Configuration for triangulation pipeline."""
    # Data paths
    data_root: Path = None
    cameras_path: Path = None  # Path to cameras directory containing extri.yml, intri.yml
    output_base: Path = field(default_factory=lambda: Path("outputs/multiview_triangulation"))

    # View configuration
    view_ids: List[str] = field(default_factory=lambda: ["19", "25", "28"])
    reference_view: str = "25"

    # Tracking configuration
    num_points: int = 400

    # Frame range
    start_frame: int = 0
    end_frame: Optional[int] = None  # None = process all frames

    # Matching settings (for cross-view point matching)
    confidence_threshold: float = 0.3  # Minimum confidence for valid matches
    min_visible_views: int = 2  # Minimum views a point must be visible in for tracking
    # Matching strategy: 'star' (all views to ref) or 'chain' (sequential) or 'ring' (chain + close loop)
    # 'ring' is best for surrounding cameras as it closes the loop
    matching_strategy: str = 'ring'
    # Local feature matching: each view detects its own features and matches only with adjacent neighbors
    # If True: no propagation through chain, each track only spans views with real direct matches
    # If False: features propagate from first view through the chain (original behavior)
    local_feature_matching: bool = True

    # Triangulation settings
    min_views_for_triangulation: int = 2  # Minimum views a point must be visible in for triangulation
    reprojection_error_threshold: float = 5.0  # Max reprojection error in pixels

    # Outlier filtering settings
    filter_outliers: bool = True  # Enable outlier filtering
    outlier_std_threshold: float = 2.0  # Points beyond N std from median are outliers
    outlier_neighbor_count: int = 10  # Number of neighbors for statistical filtering

    # Multi-GPU settings for tracking
    # When devices has multiple GPUs, queries are split across them for parallel processing
    # Example: devices=('cuda:0', 'cuda:1') will use both GPUs for tracking
    # If None, uses single GPU
    devices: Optional[Tuple[str, ...]] = None

    # Visualization settings
    point_size: float = 5.0
    render_fps: int = 30
    # Camera distance scale: >1.0 moves camera further out for better overview
    camera_distance_scale: float = 1.5
    # Number of views to traverse per video cycle (slower = easier to observe)
    # e.g., views_per_cycle=3 means camera moves through 3 views during entire video
    views_per_cycle: float = 3.0

    @property
    def dataset_name(self) -> str:
        """Extract dataset name from data_root (e.g., '0172_05')."""
        if self.data_root is None:
            return "unknown"
        return self.data_root.name

    @property
    def output_dir(self) -> Path:
        """Output directory: outputs/multiview_triangulation/{dataset_name}/"""
        return self.output_base / self.dataset_name

    @property
    def views_str(self) -> str:
        """View IDs as string for filenames (e.g., 'v19_v25_v28')."""
        return "_".join([f"v{vid}" for vid in self.view_ids])

    def get_tracking_result_path(self) -> Path:
        """Path for tracking results: {output_dir}/tracking_{views}_{start}-{end}.pkl"""
        end_str = self.end_frame if self.end_frame is not None else "end"
        filename = f"tracking_{self.views_str}_f{self.start_frame}-{end_str}.pkl"
        return self.output_dir / filename

    def get_triangulation_result_path(self) -> Path:
        """Path for triangulation results."""
        end_str = self.end_frame if self.end_frame is not None else "end"
        filename = f"triangulation_{self.views_str}_f{self.start_frame}-{end_str}.pkl"
        return self.output_dir / filename

    def get_video_output_path(self) -> Path:
        """Path for output video."""
        end_str = self.end_frame if self.end_frame is not None else "end"
        filename = f"pointcloud_{self.views_str}_f{self.start_frame}-{end_str}.mp4"
        return self.output_dir / filename

    def get_debug_dir(self) -> Path:
        """Path for debug visualizations."""
        return self.output_dir / "debug"


class CameraLoader:
    """Load camera parameters from YAML files."""

    def __init__(self, cameras_path: Path):
        self.cameras_path = Path(cameras_path)
        self.extri_path = self.cameras_path / "extri.yml"
        self.intri_path = self.cameras_path / "intri.yml"

        if not self.extri_path.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {self.extri_path}")
        if not self.intri_path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {self.intri_path}")

    def load_camera(self, view_id: str) -> CameraParams:
        """Load camera parameters for a specific view."""
        # Load extrinsics
        fs_extri = cv2.FileStorage(str(self.extri_path), cv2.FILE_STORAGE_READ)

        # Get rotation matrix and translation
        Rot = fs_extri.getNode(f"Rot_{view_id}").mat()
        T = fs_extri.getNode(f"T_{view_id}").mat()
        fs_extri.release()

        if Rot is None or T is None:
            raise ValueError(f"Camera {view_id} not found in extrinsics file")

        # Load intrinsics
        fs_intri = cv2.FileStorage(str(self.intri_path), cv2.FILE_STORAGE_READ)

        K = fs_intri.getNode(f"K_{view_id}").mat()
        D = fs_intri.getNode(f"D_{view_id}").mat()
        H = fs_intri.getNode(f"H_{view_id}").real()
        W = fs_intri.getNode(f"W_{view_id}").real()
        fs_intri.release()

        if K is None:
            raise ValueError(f"Camera {view_id} not found in intrinsics file")

        return CameraParams(
            view_id=view_id,
            K=K,
            R=Rot,
            T=T.flatten(),
            D=D.flatten() if D is not None else np.zeros(5),
            H=int(H),
            W=int(W)
        )

    def load_cameras(self, view_ids: List[str]) -> Dict[str, CameraParams]:
        """Load camera parameters for multiple views."""
        cameras = {}
        for view_id in view_ids:
            cameras[view_id] = self.load_camera(view_id)
            CONSOLE.print(f"[green]Loaded camera {view_id}: K shape={cameras[view_id].K.shape}, "
                         f"R shape={cameras[view_id].R.shape}, T shape={cameras[view_id].T.shape}")
        return cameras


class RobustTriangulator:
    """Robust triangulation of 2D points to 3D."""

    def __init__(self, cameras: Dict[str, CameraParams], config: TriangulationConfig):
        self.cameras = cameras
        self.config = config
        self.view_ids = list(cameras.keys())

    def triangulate_point(
        self,
        points_2d: Dict[str, np.ndarray],
        visibility: Dict[str, bool]
    ) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Triangulate a single point from multiple views.

        Args:
            points_2d: Dict mapping view_id -> (x, y) 2D point
            visibility: Dict mapping view_id -> visibility flag

        Returns:
            point_3d: [3] 3D point or None if triangulation failed
            reprojection_error: Mean reprojection error
            num_views: Number of views used
        """
        # Collect visible views
        visible_views = [vid for vid in self.view_ids if visibility.get(vid, False)]

        if len(visible_views) < self.config.min_views_for_triangulation:
            return None, float('inf'), len(visible_views)

        # Build projection matrices and 2D points
        proj_matrices = []
        pts_2d = []

        for vid in visible_views:
            proj_matrices.append(self.cameras[vid].P)
            pts_2d.append(points_2d[vid])

        # Triangulate using DLT
        point_3d = self._triangulate_dlt(proj_matrices, pts_2d)

        if point_3d is None:
            return None, float('inf'), len(visible_views)

        # Compute reprojection error
        total_error = 0.0
        for i, vid in enumerate(visible_views):
            P = proj_matrices[i]
            pt_2d = pts_2d[i]

            # Project 3D point
            pt_homo = P @ np.append(point_3d, 1.0)
            pt_proj = pt_homo[:2] / pt_homo[2]

            # Compute error
            error = np.linalg.norm(pt_proj - pt_2d)
            total_error += error

        mean_error = total_error / len(visible_views)

        # Reject if reprojection error is too high
        if mean_error > self.config.reprojection_error_threshold:
            return None, mean_error, len(visible_views)

        return point_3d, mean_error, len(visible_views)

    def _triangulate_dlt(
        self,
        proj_matrices: List[np.ndarray],
        pts_2d: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Triangulate using Direct Linear Transform (DLT).

        Args:
            proj_matrices: List of 3x4 projection matrices
            pts_2d: List of (x, y) 2D points

        Returns:
            [3] 3D point or None
        """
        n_views = len(proj_matrices)

        # Build linear system A @ X = 0
        A = np.zeros((2 * n_views, 4))

        for i, (P, pt) in enumerate(zip(proj_matrices, pts_2d)):
            x, y = pt[0], pt[1]
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]

        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]

            # Convert from homogeneous
            if abs(X[3]) < 1e-10:
                return None

            point_3d = X[:3] / X[3]

            # Check if point is in front of all cameras
            for vid, P in zip(self.view_ids[:n_views], proj_matrices):
                # Check depth (z in camera frame)
                pt_cam = P @ np.append(point_3d, 1.0)
                if pt_cam[2] <= 0:
                    return None

            return point_3d

        except np.linalg.LinAlgError:
            return None

    def triangulate_frame(
        self,
        tracks: Dict[str, np.ndarray],
        visibility: Dict[str, np.ndarray],
        scale_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate all points in a frame.

        Args:
            tracks: Dict mapping view_id -> [N, 2] 2D points
            visibility: Dict mapping view_id -> [N] visibility flags
            scale_factor: Scale factor for 2D coordinates (if images were resized)

        Returns:
            points_3d: [M, 3] valid 3D points
            colors: [M, 3] colors (from reference view)
            point_indices: [M] original point indices
        """
        ref_view = self.config.reference_view
        n_points = len(tracks[ref_view])

        points_3d = []
        point_indices = []
        reprojection_errors = []

        for i in range(n_points):
            # Get 2D points and visibility for this point
            pts_2d = {}
            vis = {}

            for vid in self.view_ids:
                # Scale coordinates if needed
                pts_2d[vid] = tracks[vid][i] * scale_factor
                # visibility can be boolean array or float array
                v = visibility[vid][i]
                vis[vid] = bool(v) if isinstance(v, (bool, np.bool_)) else v > 0.5

            # Triangulate
            pt_3d, error, n_views = self.triangulate_point(pts_2d, vis)

            if pt_3d is not None:
                points_3d.append(pt_3d)
                point_indices.append(i)
                reprojection_errors.append(error)

        if len(points_3d) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])

        points_3d = np.array(points_3d)
        point_indices = np.array(point_indices)

        # Filter outliers if enabled
        if self.config.filter_outliers and len(points_3d) > self.config.outlier_neighbor_count:
            inlier_mask = self._filter_outliers(points_3d)
            n_before = len(points_3d)
            points_3d = points_3d[inlier_mask]
            point_indices = point_indices[inlier_mask]
            n_after = len(points_3d)
            if n_before != n_after:
                CONSOLE.print(f"    [yellow]Filtered {n_before - n_after} outliers, {n_after} points remain")

        # Generate colors based on point index (consistent across frames)
        colors = np.zeros((len(point_indices), 3))
        for i, idx in enumerate(point_indices):
            hue = int(180 * idx / n_points)
            color_hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)
            colors[i] = color_rgb[0, 0] / 255.0

        return points_3d, colors, point_indices

    def _filter_outliers(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Filter outlier points using statistical outlier removal.

        Args:
            points_3d: [N, 3] 3D points

        Returns:
            [N] boolean mask of inliers
        """
        n_points = len(points_3d)
        k = min(self.config.outlier_neighbor_count, n_points - 1)

        if k < 1:
            return np.ones(n_points, dtype=bool)

        # Compute distances to k nearest neighbors for each point
        mean_distances = np.zeros(n_points)

        for i in range(n_points):
            # Compute distance to all other points
            dists = np.linalg.norm(points_3d - points_3d[i], axis=1)
            # Get k nearest (excluding self)
            dists_sorted = np.sort(dists)[1:k+1]
            mean_distances[i] = dists_sorted.mean()

        # Compute statistics
        median_dist = np.median(mean_distances)
        std_dist = np.std(mean_distances)

        # Mark outliers as points with mean distance > median + threshold * std
        threshold = self.config.outlier_std_threshold
        inlier_mask = mean_distances < (median_dist + threshold * std_dist)

        return inlier_mask


class PointCloudRenderer:
    """Render 3D point clouds as video with camera trajectory through given views."""

    def __init__(self, cameras: Dict[str, CameraParams], config: TriangulationConfig):
        self.cameras = cameras
        self.config = config
        self.view_ids = config.view_ids

        # Get camera centers and rotations for interpolation
        self.camera_centers = [cameras[vid].camera_center for vid in self.view_ids]
        self.camera_rotations = [cameras[vid].R for vid in self.view_ids]

        # Compute scene center from camera positions
        all_centers = np.array(self.camera_centers)
        self.scene_center = all_centers.mean(axis=0)

        # Camera distance scale: move cameras further from scene center
        self.camera_distance_scale = config.camera_distance_scale
        # Views per cycle: how many views to traverse in one video playback
        self.views_per_cycle = config.views_per_cycle

        # Output image size
        self.render_size = (720, 1280)  # H, W

    def _interpolate_camera(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate camera position and rotation along the path of real cameras.

        Args:
            t: Interpolation parameter in [0, 1] (loops through views_per_cycle cameras)

        Returns:
            cam_pos: [3] camera position
            R: [3, 3] camera rotation matrix
        """
        n_cameras = len(self.camera_centers)

        # Map t to camera index
        # t=0 -> camera 0, t=1 -> camera (views_per_cycle)
        # This slows down the camera movement
        t_scaled = t * self.views_per_cycle
        idx0 = int(t_scaled) % n_cameras
        idx1 = (idx0 + 1) % n_cameras
        alpha = t_scaled - int(t_scaled)

        # Interpolate position (linear)
        pos0 = self.camera_centers[idx0]
        pos1 = self.camera_centers[idx1]
        cam_pos = (1 - alpha) * pos0 + alpha * pos1

        # Apply camera distance scale: move camera away from scene center
        direction = cam_pos - self.scene_center
        cam_pos = self.scene_center + direction * self.camera_distance_scale

        # Interpolate rotation (simple linear interpolation of matrices)
        # For better results, should use SLERP, but this is simpler
        R0 = self.camera_rotations[idx0]
        R1 = self.camera_rotations[idx1]
        R = (1 - alpha) * R0 + alpha * R1

        # Re-orthogonalize R
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        return cam_pos, R

    def _project_points(
        self,
        points_3d: np.ndarray,
        cam_pos: np.ndarray,
        R: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D using camera parameters.

        Returns:
            pts_2d: [N, 2] projected points
            depths: [N] depth values
            valid_mask: [N] boolean mask for valid projections
        """
        # Transform to camera frame
        points_cam = (points_3d - cam_pos) @ R.T

        # Check visibility (in front of camera)
        valid_mask = points_cam[:, 2] > 0.1

        # Project using intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        pts_2d = np.zeros((len(points_3d), 2))
        depths = points_cam[:, 2].copy()

        valid_indices = np.where(valid_mask)[0]
        for i in valid_indices:
            z = points_cam[i, 2]
            pts_2d[i, 0] = fx * points_cam[i, 0] / z + cx
            pts_2d[i, 1] = fy * points_cam[i, 1] / z + cy

        return pts_2d, depths, valid_mask

    def render_pointcloud_3d(
        self,
        points_3d: np.ndarray,
        colors: np.ndarray,
        frame_idx: int,
        total_frames: int,
        show_cameras: bool = True
    ) -> np.ndarray:
        """
        Render 3D point cloud from an interpolated camera position.

        Args:
            points_3d: [N, 3] 3D points
            colors: [N, 3] RGB colors (0-1)
            frame_idx: Current frame index
            total_frames: Total number of frames
            show_cameras: Whether to show camera positions

        Returns:
            Rendered image [H, W, 3]
        """
        H, W = self.render_size
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background

        if len(points_3d) == 0:
            return canvas

        # Calculate interpolation parameter
        t = frame_idx / max(total_frames - 1, 1)

        # Get interpolated camera
        cam_pos, R = self._interpolate_camera(t)

        # Use reference camera intrinsics (scaled for render size)
        ref_cam = self.cameras[self.config.reference_view]
        scale_x = W / ref_cam.W
        scale_y = H / ref_cam.H
        K = ref_cam.K.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] = W / 2  # cx
        K[1, 2] = H / 2  # cy

        # Project points
        pts_2d, depths, valid_mask = self._project_points(points_3d, cam_pos, R, K)

        if not valid_mask.any():
            return canvas

        # Sort by depth (far first)
        valid_indices = np.where(valid_mask)[0]
        depths_valid = depths[valid_indices]
        sort_order = np.argsort(depths_valid)[::-1]
        sorted_indices = valid_indices[sort_order]

        # Draw points
        base_size = self.config.point_size
        median_depth = np.median(depths_valid)

        for idx in sorted_indices:
            x, y = int(pts_2d[idx, 0]), int(pts_2d[idx, 1])

            if 0 <= x < W and 0 <= y < H:
                color = colors[idx]
                color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                # Size varies with depth
                depth_factor = median_depth / max(depths[idx], 0.1)
                size = max(2, int(base_size * depth_factor * 0.5))

                cv2.circle(canvas, (x, y), size, color_bgr, -1)

        # Draw camera positions
        if show_cameras:
            for vid in self.view_ids:
                cam_world = self.cameras[vid].camera_center
                cam_pt = np.array([cam_world])
                pt_2d, _, valid = self._project_points(cam_pt, cam_pos, R, K)

                if valid[0]:
                    x_c, y_c = int(pt_2d[0, 0]), int(pt_2d[0, 1])
                    if 0 <= x_c < W and 0 <= y_c < H:
                        cv2.circle(canvas, (x_c, y_c), 8, (0, 255, 255), -1)
                        cv2.putText(canvas, vid, (x_c + 10, y_c + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Add info text
        t_scaled = t * self.views_per_cycle
        current_view_idx = int(t_scaled) % len(self.view_ids)
        cv2.putText(canvas, f"Frame {frame_idx} - {len(points_3d)} points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"View: {self.view_ids[current_view_idx]} (dist_scale={self.camera_distance_scale:.1f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return canvas

    def render_combined_view(
        self,
        points_3d: np.ndarray,
        colors: np.ndarray,
        images: Dict[str, np.ndarray],
        frame_idx: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Render combined view: 3D point cloud + 2D projections.

        Args:
            points_3d: [N, 3] 3D points
            colors: [N, 3] RGB colors (0-1)
            images: Dict mapping view_id -> background image
            frame_idx: Frame index
            total_frames: Total frames

        Returns:
            Combined rendered image
        """
        # Render 3D view
        view_3d = self.render_pointcloud_3d(points_3d, colors, frame_idx, total_frames)
        H_3d, W_3d = view_3d.shape[:2]

        # Calculate scale for 2D views to fit all views vertically within H_3d
        n_views = len(self.config.view_ids)
        available_height_per_view = H_3d // n_views

        views_2d = []
        for view_id in self.config.view_ids:
            image = images.get(view_id)
            if image is None:
                continue

            H_img, W_img = image.shape[:2]

            # Scale to fit within available height
            scale = available_height_per_view / H_img
            small_h = int(H_img * scale)
            small_w = int(W_img * scale)

            # Resize image
            small_img = cv2.resize(image, (small_w, small_h))

            # Project points onto this view
            camera = self.cameras[view_id]
            P = camera.P

            for i, (pt_3d, color) in enumerate(zip(points_3d, colors)):
                pt_homo = P @ np.append(pt_3d, 1.0)
                if pt_homo[2] <= 0:
                    continue
                pt_2d = pt_homo[:2] / pt_homo[2]
                x, y = int(pt_2d[0] * scale), int(pt_2d[1] * scale)

                if 0 <= x < small_w and 0 <= y < small_h:
                    color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                    cv2.circle(small_img, (x, y), 2, color_bgr, -1)

            # Add label
            cv2.putText(small_img, f"View {view_id}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            views_2d.append(small_img)

        # Combine: 3D view on left, 2D views stacked on right
        if views_2d:
            # Ensure all 2D views have the same width
            max_width = max(v.shape[1] for v in views_2d)
            views_2d_padded = []
            for v in views_2d:
                if v.shape[1] < max_width:
                    pad = np.zeros((v.shape[0], max_width - v.shape[1], 3), dtype=np.uint8)
                    v = np.hstack([v, pad])
                views_2d_padded.append(v)

            # Stack 2D views vertically
            views_2d_combined = np.vstack(views_2d_padded)

            # Adjust height to match 3D view exactly
            h_2d = views_2d_combined.shape[0]
            if h_2d < H_3d:
                pad = np.zeros((H_3d - h_2d, views_2d_combined.shape[1], 3), dtype=np.uint8)
                views_2d_combined = np.vstack([views_2d_combined, pad])
            elif h_2d > H_3d:
                views_2d_combined = views_2d_combined[:H_3d]

            combined = np.hstack([view_3d, views_2d_combined])
        else:
            combined = view_3d

        return combined


class TriangulationPipeline:
    """
    Main pipeline for multi-view triangulation.

    Supports saving/loading intermediate results at each step.
    """

    def __init__(self, config: TriangulationConfig):
        self.config = config

        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.get_debug_dir().mkdir(parents=True, exist_ok=True)

        CONSOLE.print(f"\n[cyan]Output directory: {config.output_dir}")

        # Load cameras
        CONSOLE.print("[bold cyan]Loading camera parameters...")
        self.camera_loader = CameraLoader(config.cameras_path)
        self.cameras = self.camera_loader.load_cameras(config.view_ids)

        # Compute scale factor (images are resized to 512x512 for tracking)
        # Original size is from camera params
        ref_cam = self.cameras[config.reference_view]
        self.tracking_size = (512, 512)  # H, W used in tracking
        self.scale_factor = ref_cam.W / self.tracking_size[1]  # Scale from tracking coords to original
        CONSOLE.print(f"[cyan]Scale factor: {self.scale_factor} (tracking {self.tracking_size} -> original {ref_cam.W}x{ref_cam.H})")

    def step1_tracking(self, force_rerun: bool = False) -> Dict:
        """
        Step 1: Run multi-view tracking to get 2D correspondences.

        Args:
            force_rerun: If True, rerun even if results exist

        Returns:
            Dict with 'tracks' and 'visibility' per view
        """
        result_path = self.config.get_tracking_result_path()

        if result_path.exists() and not force_rerun:
            CONSOLE.print(f"\n[yellow]Step 1: Loading cached tracking results from {result_path}")
            with open(result_path, 'rb') as f:
                # Trusted source: internal cache file
                return pickle.load(f)

        CONSOLE.print("\n[bold cyan]Step 1: Running multi-view tracking...")

        from .config import MultiViewConfig
        from .multiview_tracker import MultiViewTracker

        # Create tracking config
        tracking_config = MultiViewConfig(
            data_root=self.config.data_root / "images",
            view_ids=self.config.view_ids,
            reference_view=self.config.reference_view,
            num_points=self.config.num_points,
            start_frame=self.config.start_frame,
            end_frame=self.config.end_frame,
        )
        tracking_config.visualization.save_video = False
        tracking_config.visualization.show_live = False

        # Set matching config
        tracking_config.matching.confidence_threshold = self.config.confidence_threshold
        tracking_config.matching.min_visible_views = self.config.min_visible_views
        tracking_config.matching.require_visible_in_all_views = False  # Use min_visible_views instead
        tracking_config.matching.matching_strategy = self.config.matching_strategy
        tracking_config.matching.local_feature_matching = self.config.local_feature_matching

        # Set multi-GPU if configured
        if self.config.devices:
            tracking_config.model.devices = self.config.devices

        # Run tracking
        tracker = MultiViewTracker(tracking_config)
        all_tracks, all_visibility, initial_confidence = tracker.run()

        # Use first view to get n_frames and n_points
        first_view = self.config.view_ids[0]

        # Package results
        results = {
            'tracks': all_tracks,  # view_id -> list of [N, 2] per frame
            'visibility': all_visibility,  # view_id -> list of [N] per frame (temporal tracking visibility)
            'initial_confidence': initial_confidence,  # view_id -> [N] (cross-view matching confidence)
            'n_frames': len(all_tracks[first_view]),
            'n_points': len(all_tracks[first_view][0]) if all_tracks[first_view] else 0,
            'config': {
                'view_ids': self.config.view_ids,
                'start_frame': self.config.start_frame,
                'end_frame': self.config.end_frame,
            }
        }

        # Save results
        CONSOLE.print(f"[green]Saving tracking results to {result_path}")
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)

        return results

    def step2_triangulation(self, tracking_results: Dict, force_rerun: bool = False) -> Dict:
        """
        Step 2: Triangulate 2D points to 3D.

        Args:
            tracking_results: Results from step 1
            force_rerun: If True, rerun even if results exist

        Returns:
            Dict with 3D points per frame
        """
        result_path = self.config.get_triangulation_result_path()

        if result_path.exists() and not force_rerun:
            CONSOLE.print(f"\n[yellow]Step 2: Loading cached triangulation results from {result_path}")
            with open(result_path, 'rb') as f:
                # Trusted source: internal cache file
                return pickle.load(f)

        CONSOLE.print("\n[bold cyan]Step 2: Triangulating points...")

        triangulator = RobustTriangulator(self.cameras, self.config)

        all_tracks = tracking_results['tracks']
        all_visibility = tracking_results['visibility']
        # initial_confidence tells us which views have valid cross-view matches
        initial_confidence = tracking_results.get('initial_confidence', None)
        n_frames = tracking_results['n_frames']

        frame_results = []

        for frame_idx in range(n_frames):
            # Get tracks and visibility for this frame
            tracks = {vid: all_tracks[vid][frame_idx] for vid in self.config.view_ids}

            # Combine initial matching confidence with temporal tracking visibility
            # A point is valid in a view if: (1) it had a confident cross-view match AND (2) tracking succeeded
            visibility = {}
            for vid in self.config.view_ids:
                temporal_vis = all_visibility[vid][frame_idx]
                if initial_confidence is not None:
                    # Use initial confidence - only trust matches with high confidence
                    init_conf = initial_confidence[vid]
                    # Point is visible if it had a confident match AND tracking succeeded
                    visibility[vid] = (init_conf > self.config.confidence_threshold) & (temporal_vis > 0.5)
                else:
                    # Fallback: just use temporal visibility
                    visibility[vid] = temporal_vis > 0.5

            # Triangulate
            points_3d, colors, point_indices = triangulator.triangulate_frame(
                tracks, visibility, scale_factor=self.scale_factor
            )

            frame_results.append({
                'points_3d': points_3d,
                'colors': colors,
                'point_indices': point_indices,
                'n_valid': len(points_3d)
            })

            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                CONSOLE.print(f"  Frame {frame_idx}: {len(points_3d)} valid 3D points")

        results = {
            'frames': frame_results,
            'n_frames': n_frames,
            'scale_factor': self.scale_factor,
            'config': {
                'reprojection_error_threshold': self.config.reprojection_error_threshold,
                'min_views_for_triangulation': self.config.min_views_for_triangulation,
            }
        }

        # Save results
        CONSOLE.print(f"[green]Saving triangulation results to {result_path}")
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)

        # Print summary
        n_points_per_frame = [fr['n_valid'] for fr in frame_results]
        CONSOLE.print(f"\n[cyan]Triangulation summary:")
        CONSOLE.print(f"  Total frames: {n_frames}")
        CONSOLE.print(f"  Points per frame: min={min(n_points_per_frame)}, max={max(n_points_per_frame)}, avg={np.mean(n_points_per_frame):.1f}")

        return results

    def step3_render_video(self, triangulation_results: Dict, force_rerun: bool = False) -> Path:
        """
        Step 3: Render point clouds as video.

        Args:
            triangulation_results: Results from step 2
            force_rerun: If True, rerun even if video exists

        Returns:
            Path to output video
        """
        output_path = self.config.get_video_output_path()

        if output_path.exists() and not force_rerun:
            CONSOLE.print(f"\n[yellow]Step 3: Video already exists at {output_path}")
            return output_path

        CONSOLE.print("\n[bold cyan]Step 3: Rendering point cloud video...")

        import imageio

        renderer = PointCloudRenderer(self.cameras, self.config)
        frame_results = triangulation_results['frames']
        n_frames = triangulation_results['n_frames']

        # Create image loaders for each view
        from .multiview_tracker import ViewTracker
        from .config import MultiViewConfig, ViewConfig

        # We just need to load frames, not track
        frame_loaders = {}
        for view_id in self.config.view_ids:
            view_config = ViewConfig(
                view_id=view_id,
                images_dir=self.config.data_root / "images" / view_id,
                is_reference=(view_id == self.config.reference_view)
            )
            # Just load frame paths
            extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
            paths = []
            for ext in extensions:
                paths.extend(view_config.images_dir.glob(f'*{ext}'))
                paths.extend(view_config.images_dir.glob(f'*{ext.upper()}'))
            frame_loaders[view_id] = sorted(paths)

        # Initialize video writer
        # Get frame size from first camera
        ref_cam = self.cameras[self.config.reference_view]
        H, W = ref_cam.H, ref_cam.W
        n_views = len(self.config.view_ids)

        writer = imageio.get_writer(
            str(output_path),
            fps=self.config.render_fps,
            codec='libx264',
            quality=8
        )

        for frame_idx in range(n_frames):
            actual_frame_idx = self.config.start_frame + frame_idx

            # Load images for this frame
            images = {}
            for view_id in self.config.view_ids:
                if actual_frame_idx < len(frame_loaders[view_id]):
                    img = cv2.imread(str(frame_loaders[view_id][actual_frame_idx]))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize to camera resolution if needed
                        if img.shape[0] != H or img.shape[1] != W:
                            img = cv2.resize(img, (W, H))
                        images[view_id] = img

            # Get 3D points for this frame
            points_3d = frame_results[frame_idx]['points_3d']
            colors = frame_results[frame_idx]['colors']

            # Render 3D point cloud with rotating viewpoint
            rendered = renderer.render_combined_view(
                points_3d, colors, images, actual_frame_idx, n_frames
            )

            writer.append_data(rendered)

            if (frame_idx + 1) % 10 == 0:
                CONSOLE.print(f"  Rendered frame {frame_idx + 1}/{n_frames}")

        writer.close()
        CONSOLE.print(f"[green]Video saved to {output_path}")

        return output_path

    def run(self, force_rerun_all: bool = False) -> Path:
        """
        Run the full pipeline.

        Args:
            force_rerun_all: If True, rerun all steps even if cached results exist

        Returns:
            Path to output video
        """
        CONSOLE.print("\n" + "="*60)
        CONSOLE.print("[bold cyan]Multi-View Triangulation Pipeline")
        CONSOLE.print("="*60)

        # Step 1: Tracking
        tracking_results = self.step1_tracking(force_rerun=force_rerun_all)

        # Step 2: Triangulation
        triangulation_results = self.step2_triangulation(tracking_results, force_rerun=force_rerun_all)

        # Step 3: Render video
        video_path = self.step3_render_video(triangulation_results, force_rerun=force_rerun_all)

        CONSOLE.print("\n" + "="*60)
        CONSOLE.print("[bold green]Pipeline complete!")
        CONSOLE.print(f"Output video: {video_path}")
        CONSOLE.print("="*60)

        return video_path


def main():
    """Run triangulation pipeline with default settings."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-view triangulation pipeline")
    parser.add_argument('--data_root', type=str, default='data/multiview_tracker/0172_05',
                       help='Path to data root')
    parser.add_argument('--views', type=str, nargs='+', default=['19', '25', '28'],
                       help='View IDs to use')
    parser.add_argument('--ref_view', type=str, default='25',
                       help='Reference view')
    parser.add_argument('--num_points', type=int, default=400,
                       help='Number of points to track')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='End frame (default: None = all frames)')
    parser.add_argument('--output_base', type=str, default='outputs/multiview_triangulation',
                       help='Base output directory')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun all steps')
    parser.add_argument('--step', type=str, choices=['tracking', 'triangulation', 'render', 'all'],
                       default='all', help='Which step to run')

    args = parser.parse_args()

    data_root = Path(args.data_root)

    config = TriangulationConfig(
        data_root=data_root,
        cameras_path=data_root / "cameras",
        output_base=Path(args.output_base),
        view_ids=args.views,
        reference_view=args.ref_view,
        num_points=args.num_points,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    pipeline = TriangulationPipeline(config)

    if args.step == 'all':
        pipeline.run(force_rerun_all=args.force_rerun)
    elif args.step == 'tracking':
        pipeline.step1_tracking(force_rerun=args.force_rerun)
    elif args.step == 'triangulation':
        tracking_results = pipeline.step1_tracking(force_rerun=False)
        pipeline.step2_triangulation(tracking_results, force_rerun=args.force_rerun)
    elif args.step == 'render':
        tracking_results = pipeline.step1_tracking(force_rerun=False)
        triangulation_results = pipeline.step2_triangulation(tracking_results, force_rerun=False)
        pipeline.step3_render_video(triangulation_results, force_rerun=args.force_rerun)


if __name__ == '__main__':
    main()

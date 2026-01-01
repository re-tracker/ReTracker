#!/usr/bin/env python3
"""Quick test to verify camera loading and triangulation setup."""

from pathlib import Path
import numpy as np

from retracker.apps.multiview.triangulation_pipeline import CameraLoader, CameraParams

def test_camera_loading():
    """Test camera parameter loading."""
    print("="*60)
    print("Testing camera loading...")
    print("="*60)

    cameras_path = Path("data/multiview_tracker/0172_05/cameras")
    view_ids = ["19", "25", "28"]

    try:
        loader = CameraLoader(cameras_path)
        cameras = loader.load_cameras(view_ids)

        print(f"\nLoaded {len(cameras)} cameras")

        for view_id, cam in cameras.items():
            print(f"\nCamera {view_id}:")
            print(f"  K (intrinsic):\n{cam.K}")
            print(f"  R (rotation):\n{cam.R}")
            print(f"  T (translation): {cam.T}")
            print(f"  Image size: {cam.W}x{cam.H}")
            print(f"  Camera center: {cam.camera_center}")

            # Verify rotation matrix is orthogonal
            RRT = cam.R @ cam.R.T
            is_orthogonal = np.allclose(RRT, np.eye(3), atol=1e-6)
            print(f"  Rotation orthogonal: {is_orthogonal}")

            # Verify projection matrix shape
            print(f"  Projection matrix P shape: {cam.P.shape}")

        # Test triangulation geometry
        print("\n" + "="*60)
        print("Testing triangulation geometry...")
        print("="*60)

        # Camera centers should be different
        centers = [cameras[vid].camera_center for vid in view_ids]
        for i, vid1 in enumerate(view_ids):
            for j, vid2 in enumerate(view_ids):
                if i < j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    print(f"  Distance {vid1} <-> {vid2}: {dist:.4f}")

        print("\n[SUCCESS] Camera loading test passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Camera loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_triangulation():
    """Test triangulation with synthetic data."""
    print("\n" + "="*60)
    print("Testing simple triangulation...")
    print("="*60)

    cameras_path = Path("data/multiview_tracker/0172_05/cameras")
    view_ids = ["19", "25", "28"]

    loader = CameraLoader(cameras_path)
    cameras = loader.load_cameras(view_ids)

    # Create a synthetic 3D point
    point_3d_gt = np.array([0.0, 0.0, 2.0])  # A point in front of cameras
    print(f"Ground truth 3D point: {point_3d_gt}")

    # Project to each view
    projections = {}
    for vid in view_ids:
        P = cameras[vid].P
        pt_homo = P @ np.append(point_3d_gt, 1.0)
        pt_2d = pt_homo[:2] / pt_homo[2]
        projections[vid] = pt_2d
        print(f"  Projection in view {vid}: {pt_2d}")

    # Triangulate back
    from retracker.apps.multiview.triangulation_pipeline import RobustTriangulator, TriangulationConfig

    config = TriangulationConfig(
        data_root=Path("data/multiview_tracker/0172_05"),
        cameras_path=cameras_path,
        output_base=Path("outputs/multiview_triangulation_test"),
        view_ids=view_ids,
    )

    triangulator = RobustTriangulator(cameras, config)

    visibility = {vid: True for vid in view_ids}
    point_3d_est, error, n_views = triangulator.triangulate_point(projections, visibility)

    if point_3d_est is not None:
        print(f"\nTriangulated 3D point: {point_3d_est}")
        print(f"Reprojection error: {error:.6f}")
        print(f"Number of views used: {n_views}")

        reconstruction_error = np.linalg.norm(point_3d_est - point_3d_gt)
        print(f"Reconstruction error: {reconstruction_error:.6f}")

        if reconstruction_error < 0.01:
            print("\n[SUCCESS] Triangulation test passed!")
            return True
        else:
            print("\n[WARNING] Large reconstruction error")
            return False
    else:
        print("\n[ERROR] Triangulation failed!")
        return False


if __name__ == "__main__":
    success1 = test_camera_loading()
    success2 = test_simple_triangulation() if success1 else False

    print("\n" + "="*60)
    if success1 and success2:
        print("[ALL TESTS PASSED] Pipeline is ready to run!")
    else:
        print("[SOME TESTS FAILED] Please check the errors above")
    print("="*60)

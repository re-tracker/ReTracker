#!/usr/bin/env python3
"""Debug script to test cross-view matching on multiview tracker images (view 19, 25, 28)."""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import torch
from pathlib import Path

# Setup - use multiview tracker data
DATA_ROOT = Path("data/multiview_tracker/0172_05")
DATA_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "fmasks"
VIEW_IDS = ["19", "25", "28"]
REF_VIEW = "25"
CKPT_PATH = None  # Required: set via environment variable or command line
DEVICE = "cuda"
NUM_POINTS = 500
CONFIDENCE_THRESHOLD = 0.3

# Image size - interp_shape should match this
TARGET_SIZE = (512, 512)  # (W, H)
INTERP_SHAPE = (512, 512)  # (H, W) - must match TARGET_SIZE

def load_image(path, target_size=(512, 512)):
    """Load image as RGB numpy array and resize to target size."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img

def load_mask(image_path, target_size=(512, 512)):
    """Load mask corresponding to image path and resize to target size."""
    # Convert image path to mask path
    # images/19/000000.webp -> fmasks/19/000000.png
    image_path = Path(image_path)
    mask_path = MASK_DIR / image_path.parent.name / (image_path.stem + ".png")

    if not mask_path.exists():
        print(f"[WARNING] Mask not found: {mask_path}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if target_size is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask

def filter_points_by_mask(points, mask):
    """Filter points to keep only those within the mask (mask > 0)."""
    if mask is None or len(points) == 0:
        return points

    H, W = mask.shape
    valid_mask = np.zeros(len(points), dtype=bool)

    for i, pt in enumerate(points):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < W and 0 <= y < H:
            valid_mask[i] = mask[y, x] > 0

    return points[valid_mask]

def grid_sample_in_mask(mask, num_points, existing_points=None):
    """
    Sample grid points within the mask region.

    Args:
        mask: [H, W] binary mask
        num_points: number of points to sample
        existing_points: [N, 2] existing points to avoid duplicates

    Returns:
        [M, 2] array of sampled points (x, y)
    """
    if mask is None:
        return np.array([]).reshape(0, 2)

    H, W = mask.shape

    # Find valid mask region
    valid_y, valid_x = np.where(mask > 0)
    if len(valid_x) == 0:
        return np.array([]).reshape(0, 2)

    # Calculate grid size based on num_points and mask area
    mask_area = len(valid_x)
    grid_step = max(1, int(np.sqrt(mask_area / num_points)))

    # Generate grid points
    grid_points = []
    for y in range(grid_step // 2, H, grid_step):
        for x in range(grid_step // 2, W, grid_step):
            if mask[y, x] > 0:
                grid_points.append([x, y])

    grid_points = np.array(grid_points, dtype=np.float32)

    # Remove points too close to existing points
    if existing_points is not None and len(existing_points) > 0 and len(grid_points) > 0:
        min_dist = 10  # minimum distance between points
        filtered = []
        for gp in grid_points:
            dists = np.linalg.norm(existing_points - gp, axis=1)
            if dists.min() > min_dist:
                filtered.append(gp)
        grid_points = np.array(filtered, dtype=np.float32) if filtered else np.array([]).reshape(0, 2)

    return grid_points

def detect_sift_points_with_mask(frame, mask, num_points, n_features=500):
    """
    Detect SIFT keypoints within mask region.
    If not enough points, supplement with grid sampling.

    Args:
        frame: [H, W, 3] RGB image
        mask: [H, W] binary mask (points must be where mask > 0)
        num_points: target number of points
        n_features: max SIFT features to detect

    Returns:
        [N, 2] array of (x, y) coordinates
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    keypoints = sift.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

    # Extract all SIFT points
    all_sift_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

    # Filter by mask
    if mask is not None and len(all_sift_points) > 0:
        sift_points = filter_points_by_mask(all_sift_points, mask)
    else:
        sift_points = all_sift_points

    # Take top num_points SIFT points
    sift_points = sift_points[:num_points]

    print(f"  SIFT: {len(all_sift_points)} detected, {len(sift_points)} in mask")

    # If not enough, supplement with grid sampling
    if len(sift_points) < num_points and mask is not None:
        need_more = num_points - len(sift_points)
        grid_points = grid_sample_in_mask(mask, need_more * 2, existing_points=sift_points)

        if len(grid_points) > 0:
            # Take only what we need
            grid_points = grid_points[:need_more]
            sift_points = np.vstack([sift_points, grid_points]) if len(sift_points) > 0 else grid_points
            print(f"  Added {len(grid_points)} grid points, total: {len(sift_points)}")

    return sift_points.astype(np.float32) if len(sift_points) > 0 else np.array([]).reshape(0, 2)

def match_with_model(tracker, ref_frame, target_frame, ref_points):
    """Match points using the tracking model."""
    N = len(ref_points)
    H, W = ref_frame.shape[:2]

    # Create video tensor [1, 2, C, H, W]
    video = np.stack([ref_frame, target_frame], axis=0)  # [2, H, W, C]
    video = torch.from_numpy(video).float().permute(0, 3, 1, 2)  # [2, C, H, W]
    video = video.unsqueeze(0)  # [1, 2, C, H, W] - keep on CPU, engine moves to GPU

    # Create queries [1, N, 3] with (t=0, x, y)
    queries = torch.zeros((1, N, 3), dtype=torch.float32)
    queries[0, :, 0] = 0  # t = 0, queries start at frame 0
    queries[0, :, 1] = torch.from_numpy(ref_points[:, 0]).float()  # x
    queries[0, :, 2] = torch.from_numpy(ref_points[:, 1]).float()  # y

    print(f"  Image size: H={H}, W={W}")
    print(f"  Video shape: {video.shape}, dtype: {video.dtype}")
    print(f"  Video value range: [{video.min():.1f}, {video.max():.1f}]")
    print(f"  Queries shape: {queries.shape}")
    print(f"  Query points (first 5, x y format):")
    for i in range(min(5, N)):
        print(f"    Point {i}: x={ref_points[i, 0]:.1f}, y={ref_points[i, 1]:.1f}")

    # Run tracking
    with torch.no_grad():
        trajectories, visibility = tracker.track(video, queries)
        print(f"  Trajectories shape: {trajectories.shape}")
        print(f"  Visibility shape: {visibility.shape}")

        # Get raw visibility scores (float, not thresholded)
        visibility_scores = tracker.get_visibility_scores()
        if visibility_scores is not None:
            print(f"  Visibility scores shape: {visibility_scores.shape}")

    # Get positions at frame 0 (should be same as input)
    frame0_points = trajectories[0, 0].cpu().numpy()
    print(f"  Frame 0 output (should match input, first 5):")
    for i in range(min(5, N)):
        print(f"    Point {i}: x={frame0_points[i, 0]:.1f}, y={frame0_points[i, 1]:.1f}")

    # Get positions at target frame (frame 1)
    matched_points = trajectories[0, 1].cpu().numpy()

    # Use raw visibility scores (float) instead of boolean
    if visibility_scores is not None:
        confidence = visibility_scores[0, 1].cpu().numpy()  # Raw float scores
    else:
        confidence = visibility[0, 1].cpu().numpy().astype(float)  # Fallback to boolean

    print(f"  Frame 1 output (matched points, first 5):")
    for i in range(min(5, N)):
        print(f"    Point {i}: x={matched_points[i, 0]:.1f}, y={matched_points[i, 1]:.1f}, conf={confidence[i]:.3f}")

    # Check for anomalies
    displacement = np.linalg.norm(matched_points - ref_points, axis=1)
    print(f"  Displacement stats: min={displacement.min():.1f}, max={displacement.max():.1f}, mean={displacement.mean():.1f}")

    # Check if points went outside image bounds
    out_of_bounds = ((matched_points[:, 0] < 0) | (matched_points[:, 0] >= W) |
                     (matched_points[:, 1] < 0) | (matched_points[:, 1] >= H)).sum()
    print(f"  Points out of bounds: {out_of_bounds}/{N}")

    return matched_points, confidence

def draw_matches(ref_frame, target_frame, ref_points, target_points, confidence, title, threshold=0.3):
    """Draw matching visualization. Only draws points with confidence > threshold."""
    H, W = ref_frame.shape[:2]
    combined = np.zeros((H, W * 2, 3), dtype=np.uint8)
    combined[:, :W] = ref_frame
    combined[:, W:] = target_frame

    # Filter to only valid matches (confidence > threshold)
    valid_mask = confidence > threshold
    n_total = len(ref_points)
    n_valid = valid_mask.sum()

    # Generate colors for valid points only
    colors = []
    for i in range(n_valid):
        hue = int(180 * i / max(n_valid, 1))
        color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
        colors.append(tuple(map(int, color[0, 0])))

    # Draw only valid matches
    valid_idx = 0
    for i, (ref_pt, tgt_pt, conf) in enumerate(zip(ref_points, target_points, confidence)):
        if conf <= threshold:
            continue  # Skip low confidence points

        color = colors[valid_idx]
        valid_idx += 1

        ref_x, ref_y = int(ref_pt[0]), int(ref_pt[1])
        tgt_x, tgt_y = int(tgt_pt[0]) + W, int(tgt_pt[1])

        cv2.circle(combined, (ref_x, ref_y), 5, color, -1)
        cv2.circle(combined, (tgt_x, tgt_y), 5, color, -1)

        # Draw confidence text next to reference point
        conf_text = f"{conf:.2f}"
        cv2.putText(combined, conf_text, (ref_x + 6, ref_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw line connecting matches
        cv2.line(combined, (ref_x, ref_y), (tgt_x, tgt_y), color, 1)

    # Labels
    cv2.putText(combined, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    mean_conf = confidence[valid_mask].mean() if n_valid > 0 else 0.0
    cv2.putText(combined, f"Valid matches: {n_valid}/{n_total} (threshold={threshold})", (10, H - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, f"Mean conf (valid): {mean_conf:.3f}", (10, H - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return combined

def get_first_image(view_id):
    """Get path to first image in a view directory."""
    view_dir = DATA_DIR / view_id
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    paths = []
    for ext in extensions:
        paths.extend(view_dir.glob(f'*{ext}'))
        paths.extend(view_dir.glob(f'*{ext.upper()}'))
    paths = sorted(paths)
    return paths[0] if paths else None


def main():
    print("="*60)
    print("Debug Matching Test on multiview images (view 19, 25, 28)")
    print("="*60)

    # Load first frame and mask from each view
    frames = {}
    masks = {}
    img_paths = {}
    for view_id in VIEW_IDS:
        img_path = get_first_image(view_id)
        img_paths[view_id] = img_path
        print(f"Loading view {view_id}: {img_path}")
        frames[view_id] = load_image(img_path, target_size=TARGET_SIZE)
        masks[view_id] = load_mask(img_path, target_size=TARGET_SIZE)
        if masks[view_id] is not None:
            mask_coverage = (masks[view_id] > 0).sum() / masks[view_id].size * 100
            print(f"  Mask loaded, coverage: {mask_coverage:.1f}%")

    print(f"Loaded and resized images to: {frames[REF_VIEW].shape} (TARGET_SIZE={TARGET_SIZE})")

    # Build tracker with correct settings
    print("\nBuilding tracker...")
    from retracker.apps.runtime.tracker import Tracker
    from retracker.apps.config.base_config import ModelConfig

    model_config = ModelConfig(
        ckpt_path=CKPT_PATH,
        device=DEVICE,
        interp_shape=INTERP_SHAPE,
        compile=False,
        use_amp=True,
        dtype='bf16',
        enable_highres_inference=True,
        coarse_resolution=(512, 512),
    )
    tracker = Tracker(model_config)
    print(f"Tracker built")
    print(f"  interp_shape: {model_config.interp_shape}")
    print(f"  enable_highres_inference: {model_config.enable_highres_inference}")
    print(f"  coarse_resolution: {model_config.coarse_resolution}")

    # Print model config for debugging
    model = tracker.engine.model
    print(f"\nModel config:")
    print(f"  model_task_type: {model.model_task_type}")
    print(f"  train_coarse: {model.train_coarse}")
    print(f"  using_matching_prior: {model.using_matching_prior}")
    print(f"  enable_highres_inference: {model.enable_highres_inference}")

    # Detect SIFT on reference view (with mask filtering)
    ref_frame = frames[REF_VIEW]
    ref_mask = masks[REF_VIEW]
    print(f"\nDetecting SIFT points on view {REF_VIEW} (with mask)...")
    ref_points = detect_sift_points_with_mask(ref_frame, ref_mask, NUM_POINTS, n_features=500)
    print(f"Total query points: {len(ref_points)}")

    # Match to other views
    for view_id in VIEW_IDS:
        if view_id == REF_VIEW:
            continue

        print(f"\nMatching view {REF_VIEW} -> view {view_id}...")
        target_frame = frames[view_id]
        matched_points, confidence = match_with_model(tracker, ref_frame, target_frame, ref_points)

        # Draw and save
        vis = draw_matches(ref_frame, target_frame, ref_points, matched_points, confidence,
                          f"View {REF_VIEW} -> View {view_id}", threshold=CONFIDENCE_THRESHOLD)
        output_path = f"debug_view_{REF_VIEW}_to_{view_id}.png"
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

        # Print stats
        high_conf = (confidence > CONFIDENCE_THRESHOLD).sum()
        print(f"Valid matches (conf > {CONFIDENCE_THRESHOLD}): {high_conf}/{len(ref_points)}")

    # Save reference with SIFT points and mask overlay
    ref_vis = ref_frame.copy()
    # Draw mask boundary
    if ref_mask is not None:
        mask_contours, _ = cv2.findContours((ref_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ref_vis, mask_contours, -1, (0, 255, 0), 1)

    for i, pt in enumerate(ref_points):
        x, y = int(pt[0]), int(pt[1])
        hue = int(180 * i / max(len(ref_points), 1))
        color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
        color = tuple(map(int, color[0, 0]))
        cv2.circle(ref_vis, (x, y), 5, color, -1)
    cv2.imwrite(f"debug_view_{REF_VIEW}_sift.png", cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: debug_view_{REF_VIEW}_sift.png")

    print("\nDone!")

if __name__ == "__main__":
    main()

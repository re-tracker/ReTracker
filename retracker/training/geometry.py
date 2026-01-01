import torch
import torch.nn.functional as F

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """
    Warp kpts0 from I0 to I1 with depth, K and Rt, using bilinear interpolation
    for higher accuracy.
    Also checks covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).

    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y> keypoints.
        depth0 (torch.Tensor): [N, H, W] source depth map.
        depth1 (torch.Tensor): [N, H, W] target depth map.
        T_0to1 (torch.Tensor): [N, 3, 4] transformation matrix from frame 0 to 1.
        K0 (torch.Tensor): [N, 3, 3] intrinsics for camera 0.
        K1 (torch.Tensor): [N, 3, 3] intrinsics for camera 1.

    Returns:
        valid_mask (torch.Tensor): [N, L] boolean mask for valid points.
        warped_keypoints0 (torch.Tensor): [N, L, 2] warped keypoints in I1.
    """
    _, H, W = depth0.shape

    # --- MODIFICATION START ---
    # 1. Sample depth at kpts0 using bilinear interpolation
    # Normalize keypoint coordinates to [-1, 1] for grid_sample
    kpts0_normalized = kpts0.clone()
    kpts0_normalized[..., 0] = 2.0 * kpts0[..., 0] / (W - 1) - 1.0
    kpts0_normalized[..., 1] = 2.0 * kpts0[..., 1] / (H - 1) - 1.0
    kpts0_normalized = kpts0_normalized.unsqueeze(1)  # [N, 1, L, 2]

    # Sample depth using grid_sample
    kpts0_depth = F.grid_sample(
        depth0.unsqueeze(1),  # [N, 1, H, W]
        kpts0_normalized,     # [N, 1, L, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(1).squeeze(1)  # [N, L]
    # --- MODIFICATION END ---
    
    nonzero_mask = kpts0_depth > 0

    # 2. Unproject keypoints to 3D camera space
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = torch.inverse(K0) @ kpts0_h.transpose(1, 2)  # (N, 3, L)

    # 3. Rigid Transform to camera 1 space
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]  # Computed depth in camera 1

    # 4. Project to image 1 plane
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(1, 2)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-6)  # (N, L, 2)

    # 5. Covisibility Check
    # Use the same height and width from depth1 for checking bounds
    H1, W1 = depth1.shape[1:3]
    covisible_mask = (w_kpts0[..., 0] > 0) & (w_kpts0[..., 0] < W1 - 1) & \
                     (w_kpts0[..., 1] > 0) & (w_kpts0[..., 1] < H1 - 1)
    
    # --- MODIFICATION START ---
    # 6. Depth Consistency Check using bilinear interpolation
    # Normalize warped keypoint coordinates to [-1, 1]
    w_kpts0_normalized = w_kpts0.clone()
    w_kpts0_normalized[..., 0] = 2.0 * w_kpts0_normalized[..., 0] / (W1 - 1) - 1.0
    w_kpts0_normalized[..., 1] = 2.0 * w_kpts0_normalized[..., 1] / (H1 - 1) - 1.0
    w_kpts0_normalized = w_kpts0_normalized.unsqueeze(1) # [N, 1, L, 2]

    # Sample depth from depth1 at warped locations
    # We use a mask here to avoid sampling outside the image bounds, which grid_sample would pad.
    # While padding_mode='zeros' handles it, explicit masking is clearer.
    w_kpts0_normalized[~covisible_mask.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 2)] = -2 # Set out-of-bounds to a value that grid_sample ignores
    
    w_kpts0_depth = F.grid_sample(
        depth1.unsqueeze(1),  # [N, 1, H, W]
        w_kpts0_normalized,   # [N, 1, L, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(1).squeeze(1) # [N, L]
    # --- MODIFICATION END ---
    
    # Calculate consistency mask where both depths are valid
    consistency_mask = ((w_kpts0_depth - w_kpts0_depth_computed).abs() / w_kpts0_depth_computed).abs() < 0.2
    # The mask should only be true where the sampled depth from depth1 is also valid
    consistency_mask = consistency_mask & (w_kpts0_depth > 0)

    # 7. Final valid mask
    valid_mask = nonzero_mask & covisible_mask & consistency_mask

    return valid_mask, w_kpts0

@torch.no_grad()
def _warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


@torch.no_grad()
def warp_grid(grid0, depth0, depth1, T_0to1, K0, K1):
    """ Warp a grid of points from I0 to I1 with depth, K and Rt using grid_sample
    for dense flow calculation.

    Args:
        grid0 (torch.Tensor): [N, L, 2] - <x, y> pixel coordinates of the grid.
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        valid_mask (torch.Tensor): [N, L]
        warped_grid0 (torch.Tensor): [N, L, 2] <x, y>
    """
    _, H, W = depth0.shape

    # 1. Sample depth at grid locations using grid_sample for sub-pixel accuracy
    grid0_normalized = grid0.clone()
    grid0_normalized[..., 0] = 2.0 * grid0[..., 0] / (W - 1) - 1.0
    grid0_normalized[..., 1] = 2.0 * grid0[..., 1] / (H - 1) - 1.0
    grid0_normalized = grid0_normalized.unsqueeze(1)  # (B, 1, L, 2)

    grid0_depth = F.grid_sample(
        depth0.unsqueeze(1), grid0_normalized, mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze(1).squeeze(1)  # (B, L)

    nonzero_mask = grid0_depth > 0

    # 2. Unproject
    grid0_h = torch.cat([grid0, torch.ones_like(grid0[:, :, [0]])], dim=-1) * grid0_depth.unsqueeze(-1)  # (B, L, 3)
    grid0_cam = torch.inverse(K0) @ grid0_h.transpose(1, 2)  # (B, 3, L)

    # 3. Rigid Transform
    w_grid0_cam = T_0to1[:, :3, :3] @ grid0_cam + T_0to1[:, :3, [3]]  # (B, 3, L)
    w_grid0_depth_computed = w_grid0_cam[:, 2, :]

    # 4. Project
    w_grid0_h = (K1 @ w_grid0_cam).transpose(1, 2)  # (B, L, 3)
    w_grid0 = w_grid0_h[:, :, :2] / (w_grid0_h[:, :, [2]] + 1e-6)  # (B, L, 2)

    # 5. Covisibility Check
    H1, W1 = depth1.shape[1:3]
    covisible_mask = (w_grid0[:, :, 0] >= 0) & (w_grid0[:, :, 0] < W1-1) & \
                     (w_grid0[:, :, 1] >= 0) & (w_grid0[:, :, 1] < H1-1)

    # 6. Depth Consistency Check (using grid_sample again)
    w_grid0_normalized = w_grid0.clone()
    w_grid0_normalized[..., 0] = 2.0 * w_grid0_normalized[..., 0] / (W1 - 1) - 1.0
    w_grid0_normalized[..., 1] = 2.0 * w_grid0_normalized[..., 1] / (H1 - 1) - 1.0
    w_grid0_normalized = w_grid0_normalized.unsqueeze(1)  # (B, 1, L, 2)

    w_grid0_depth = F.grid_sample(
        depth1.unsqueeze(1), w_grid0_normalized, mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze(1).squeeze(1) # (B, L)

    consistent_mask = ((w_grid0_depth - w_grid0_depth_computed).abs() / w_grid0_depth_computed).abs() < 0.2
    
    # 7. Final valid mask
    valid_mask = nonzero_mask & covisible_mask & consistent_mask

    return valid_mask, w_grid0


@torch.no_grad()
def homo_warp_kpts(kpts0, norm_pixel_mat, homo_sample_normed, original_size0=None, original_size1=None):
    """
    original_size1: N * 2, (h, w)
    """
    normed_kpts0_h = norm_pixel_mat @ torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1).transpose(2, 1) # (N * 3 * L)
    kpts_warpped_h = (torch.linalg.inv(norm_pixel_mat) @ homo_sample_normed @ normed_kpts0_h).transpose(2, 1) # (N * L * 3)
    kpts_warpped = kpts_warpped_h[..., :2] / kpts_warpped_h[..., [2]] # N * L * 2

    valid_mask = (kpts_warpped[..., 0] > 0) & (kpts_warpped[..., 0] < original_size1[:, [1]]) & (kpts_warpped[..., 1] > 0) & (kpts_warpped[..., 1] < original_size1[:, [0]])  # N * L
    if original_size0 is not None:
        valid_mask *= (kpts0[..., 0] > 0) & (kpts0[..., 0] < original_size0[:, [1]]) & (kpts0[..., 1] > 0) & (kpts0[..., 1] < original_size0[:, [0]])  # N * L
    return valid_mask, kpts_warpped


def warp_points_torch(kpts: torch.Tensor, homo: torch.Tensor, *, inverse: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Apply a homography to 2D points.

    Args:
        kpts: Keypoints in pixel coordinates, shaped (B, N, 2) or (N, 2).
        homo: Homography matrix shaped (B, 3, 3) or (3, 3).
        inverse: If True, applies the inverse homography.
        eps: Small constant for numerical stability in homogeneous division.

    Returns:
        Warped keypoints with the same shape as `kpts`.
    """
    squeeze_batch = False
    if kpts.dim() == 2:
        kpts = kpts.unsqueeze(0)
        squeeze_batch = True
    if homo.dim() == 2:
        homo = homo.unsqueeze(0)

    if inverse:
        homo = torch.inverse(homo)

    if homo.shape[0] != kpts.shape[0]:
        if homo.shape[0] == 1:
            homo = homo.expand(kpts.shape[0], -1, -1)
        else:
            raise ValueError(f"Batch mismatch: kpts batch={kpts.shape[0]}, homo batch={homo.shape[0]}")

    ones = torch.ones_like(kpts[..., :1])
    kpts_h = torch.cat([kpts, ones], dim=-1)  # (B, N, 3)
    warped_h = torch.bmm(homo, kpts_h.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)
    warped = warped_h[..., :2] / (warped_h[..., 2:3] + eps)
    return warped.squeeze(0) if squeeze_batch else warped


@torch.no_grad()
def homo_warp_kpts_glue(kpts0, homo, original_size0=None, original_size1=None):
    """
    original_size1: N * 2, (h, w)
    """
    kpts_warpped = warp_points_torch(kpts0, homo, inverse=False)

    valid_mask = (kpts_warpped[..., 0] > 0) & (kpts_warpped[..., 0] < original_size1[:, [1]]) & (kpts_warpped[..., 1] > 0) & (kpts_warpped[..., 1] < original_size1[:, [0]])  # N * L
    if original_size0 is not None:
        valid_mask *= (kpts0[..., 0] > 0) & (kpts0[..., 0] < original_size0[:, [1]]) & (kpts0[..., 1] > 0) & (kpts0[..., 1] < original_size0[:, [0]])  # N * L
    return valid_mask, kpts_warpped

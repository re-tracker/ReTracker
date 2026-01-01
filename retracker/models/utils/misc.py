"""Misc tensor utilities used by the ReTracker model.

Keep this module lightweight: it is imported by the model at runtime.
If a helper is only needed for training, place it under `retracker.training`.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def get_pos_enc(y: torch.Tensor, pos_conv_func, freq: int = 8) -> torch.Tensor:
    """Compute a learned Fourier-style positional encoding for a feature map.

    Args:
        y: Feature map tensor of shape (B, C, H, W). Only H/W/device are used.
        pos_conv_func: A callable (typically `nn.Conv2d(2, C, 1)`) applied to a
            2-channel coordinate grid.
        freq: Frequency multiplier applied before cosine.

    Returns:
        Tensor of shape (B, C_out, H, W).
    """
    b, _c, h, w = y.shape
    coarse_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
        ),
        indexing="ij",
    )
    coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[None].expand(
        b, h, w, 2
    )
    coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
    coarse_embedded_coords = torch.cos(freq * math.pi * pos_conv_func(coarse_coords))
    return coarse_embedded_coords


def normalize_keypoints(
    keypoints: torch.FloatTensor, H: int, W: int, range: tuple[float, float] = (-1, 1)
) -> torch.FloatTensor:
    """Normalize keypoints from pixel coordinates to a target range (default: [-1, 1]).

    Args:
        keypoints: Tensor of shape [..., 2] in (x, y) pixel coordinates.
        H: Image height.
        W: Image width.
        range: Target numeric range.

    Returns:
        Tensor of shape [..., 2] in the target range.
    """
    scale = range[1] - range[0]
    keypoints_norm = keypoints.clone()
    keypoints_norm[..., 0] = scale * (keypoints_norm[..., 0] / (W - 1)) + range[0]
    keypoints_norm[..., 1] = scale * (keypoints_norm[..., 1] / (H - 1)) + range[0]
    return keypoints_norm


def extract_interpolated_features(
    pos: torch.Tensor,
    feat: torch.Tensor,
    size_hw: torch.LongTensor,
    num_levels: int = 1,
) -> torch.Tensor:
    """Sample interpolated features from a feature map at normalized positions.

    Args:
        pos: Normalized coordinates of shape (B, N, 2) in (x, y), within [-1, 1].
        feat: Feature tensor of shape (B, H*W, C) or (B, C, H, W).
        size_hw: LongTensor (H, W) for reshaping when `feat` is flattened.
        num_levels: If >1, additionally samples from progressively pooled maps.

    Returns:
        If num_levels == 1: Tensor of shape (B, N, C).
        Else: Tensor of shape (B, N, num_levels, C).
    """
    grid = pos[:, None, ...]  # [B, N, 2] -> [B, 1, N, 2]
    if feat.dim() == 3:
        feat = rearrange(feat, "b (h w) c -> b c h w", h=size_hw[0], w=size_hw[1])

    if num_levels == 1:
        feat_interp = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)
        return rearrange(feat_interp, "b c 1 n -> b n c")

    feat_interp_n_lvls: list[torch.Tensor] = []
    for _ in range(num_levels):
        feat_interp = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)
        feat_interp_n_lvls.append(rearrange(feat_interp, "b c 1 n -> b n c"))
        feat = F.avg_pool2d(feat, 2, stride=2)

    return torch.stack(feat_interp_n_lvls, dim=-2)


def bilinear_sampler(
    input: torch.Tensor,
    coords: torch.Tensor,
    align_corners: bool = True,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    r"""Sample a tensor using bilinear interpolation.

    This is a thin wrapper around `torch.nn.functional.grid_sample` but uses a
    different coordinate convention.

    Args:
        input: Tensor of shape (B, C, H, W) or (B, C, T, H, W).
        coords: Tensor of shape (B, H_o, W_o, 2) for 2D, or (..., 3) for 3D.
        align_corners: Passed through to `grid_sample`.
        padding_mode: Passed through to `grid_sample`.

    Returns:
        Sampled tensor.
    """
    sizes = input.shape[2:]
    if len(sizes) not in (2, 3):
        raise ValueError(f"Expected input to be 4D/5D, got shape {tuple(input.shape)}")

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    coords -= 1
    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def clip_kpts(kpts: torch.Tensor, hw: torch.Tensor) -> torch.Tensor:
    """Clamp keypoints to valid pixel indices for a given (H, W)."""
    torch.clamp_(kpts[..., 0], min=0, max=hw[1] - 1)
    torch.clamp_(kpts[..., 1], min=0, max=hw[0] - 1)
    return kpts


def queries_to_coarse_ids(
    queries: torch.Tensor, hw0_c: torch.Tensor, scale_c: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pixel queries to flattened indices on a coarse grid."""
    B, n = queries.shape[0], queries.shape[1]
    b_ids = repeat(torch.arange(B, device=queries.device), "b -> (b n)", n=n)
    queries_coarse = (queries + scale_c / 2) // scale_c
    queries_coarse = clip_kpts(queries_coarse, hw0_c).long()
    i_ids = queries_coarse[..., 0] + queries_coarse[..., 1] * hw0_c[1]
    i_ids = i_ids.long().reshape(-1)
    return b_ids, i_ids


def _robust_cat(tensors: list[torch.Tensor | None], dim: int = 0) -> torch.Tensor | None:
    """Concatenate tensors while allowing `None` entries.

    `None` entries are replaced with a zero tensor (matching the first valid tensor).
    If all entries are `None`, returns `None`.
    """
    first_valid_tensor = next((t for t in tensors if t is not None), None)
    if first_valid_tensor is None:
        return None

    processed = [t if t is not None else torch.zeros_like(first_valid_tensor) for t in tensors]
    return torch.cat(processed, dim=dim)


def random_erase_patch(patch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Apply random erasing to a (B, WW, C) patch token tensor."""
    B, WW, _C = patch.shape
    W = 7
    device = patch.device

    mask = torch.ones((B, WW, 1), device=device)
    rand_vals = torch.rand(B, device=device)

    # Strategy A: visual dropout (all zeros)
    prob_blind = p * 0.3
    prob_block = p

    blind_indices = rand_vals < prob_blind
    if blind_indices.any():
        mask[blind_indices] = 0.0

    # Strategy B: block masking
    block_indices = (rand_vals >= prob_blind) & (rand_vals < prob_block)
    num_block = int(block_indices.sum().item())
    if num_block > 0:
        block_masks = _generate_7x7_block_masks(num_block, W, device)
        mask[block_indices] = block_masks

    return patch * mask


def _generate_7x7_block_masks(B: int, W: int, device: torch.device) -> torch.Tensor:
    """Generate (B, 49, 1) masks with a random 2~5 block erased on a 7x7 grid."""
    min_size, max_size = 2, 5
    block_w = torch.randint(min_size, max_size + 1, (B, 1), device=device)
    block_h = torch.randint(min_size, max_size + 1, (B, 1), device=device)

    max_x = W - block_w
    max_y = W - block_h

    start_x = (torch.rand(B, 1, device=device) * (max_x + 1)).floor().long()
    start_y = (torch.rand(B, 1, device=device) * (max_y + 1)).floor().long()

    grid = torch.arange(W, device=device).unsqueeze(0)

    mask_x_1d = (grid >= start_x) & (grid < (start_x + block_w))
    mask_y_1d = (grid >= start_y) & (grid < (start_y + block_h))

    mask_2d = mask_x_1d.unsqueeze(1) & mask_y_1d.unsqueeze(2)
    return (~mask_2d).float().flatten(1).unsqueeze(-1)

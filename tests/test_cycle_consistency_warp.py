import math

import torch
import torch.nn.functional as F


# ==============================================================================
# Provided Functions (with minor corrections for robustness)
# ==============================================================================


@torch.no_grad()
def cls_to_flow_refine(cls, is_A_to_B=True):
    """
    Calculates a flow field from a classification tensor.
    If is_A_to_B is False, it computes the backward flow.
    """
    B, C, H, W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))

    # Create the grid of possible flow vectors (lookup table)
    G_coords = torch.meshgrid(
        *[torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device) for _ in range(2)],
        indexing="ij",
    )
    G = torch.stack([G_coords[1], G_coords[0]], dim=-1).reshape(C, 2)

    # Convert scores to probabilities
    if device.type == "mps":
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)

    mode = cls.max(dim=1).indices

    # Get indices of the 5 neighbors (center, up, down, left, right)
    index = (
        torch.stack(
            (
                mode - res,  # Up
                mode - 1,  # Left
                mode,  # Center
                mode + 1,  # Right
                mode + res,  # Down
            ),
            dim=1,
        )
        .clamp(0, C - 1)
        .long()
    )

    # Gather probabilities of the 5 neighbors
    neighbours_probs = torch.gather(cls, dim=1, index=index)

    # Normalize probabilities to sum to 1 across the 5 neighbors
    # This prevents division by zero if total prob is 0
    total_prob_sum = neighbours_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    normalized_probs = neighbours_probs / total_prob_sum

    # Get the corresponding flow vectors for the 5 neighbors
    flow_vectors = G[index]  # Shape: [B, 5, H, W, 2]

    # Calculate the weighted average of the flow vectors
    # Note: The original implementation was correct but this is more concise
    flow = (normalized_probs.unsqueeze(-1) * flow_vectors).sum(dim=1)  # Shape: [B, H, W, 2]

    if not is_A_to_B:
        flow = get_backward_flow(flow)

    return flow


@torch.no_grad()
def get_backward_flow(flow):
    """
    Approximates the backward flow (B to A) from a given forward flow (A to B).

    Args:
        flow (torch.Tensor): The forward flow field, shape (B, H, W, 2).
                             Flow values are assumed to be in the normalized [-1, 1] range.

    Returns:
        torch.Tensor: The backward flow field, shape (B, H, W, 2).
    """
    # The `warp` function expects pixel-based flow, so we must scale it first.
    B, H, W, _ = flow.shape
    device = flow.device

    # Create scaling factor for normalized flow to pixel flow
    # [W-1]/2 for x-dim, [H-1]/2 for y-dim
    scaling_factor = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0], device=device).view(1, 1, 1, 2)

    flow_pixels = flow * scaling_factor

    # To use PyTorch functions, we need channels-first format (B, 2, H, W).
    flow_ab_ch_first = flow_pixels.permute(0, 3, 1, 2)

    # Create the field we want to sample from: the negated forward flow.
    neg_flow_ab = -flow_ab_ch_first

    # Warp the negated forward flow using the forward flow.
    # This approximates the backward flow in pixel units.
    flow_ba_pixels = warp(neg_flow_ab, flow_ab_ch_first)

    # Convert the resulting backward flow back to the normalized [-1, 1] range
    flow_ba_normalized = flow_ba_pixels / scaling_factor.permute(0, 3, 1, 2)

    # Permute back to the original (B, H, W, 2) format and return.
    return flow_ba_normalized.permute(0, 2, 3, 1)


def warp(feature_map: torch.Tensor, flow: torch.Tensor):
    """
    Warp a feature map using a pixel-based flow field.

    Args:
        feature_map (torch.Tensor): The map to be warped, shape (B, C, H, W).
        flow (torch.Tensor): The flow field in PIXEL units, shape (B, 2, H, W).

    Returns:
        torch.Tensor: The warped feature map.
    """
    B, _, H, W = flow.shape
    device = flow.device

    # Create an identity grid of pixel coordinates
    xx = torch.arange(0, W, device=device)
    yy = torch.arange(0, H, device=device)
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing="xy")
    grid = torch.stack([grid_x, grid_y], dim=0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)

    # Add the pixel displacement flow to the identity grid
    vgrid = grid + flow

    # Normalize the grid to the [-1, 1] range expected by grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # grid_sample expects the grid in (B, H, W, 2) format
    vgrid = vgrid.permute(0, 2, 3, 1)

    warped_map = F.grid_sample(
        feature_map, vgrid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return warped_map


# ==============================================================================
# Test Function for Cycle Consistency
# ==============================================================================


def test_cycle_consistency_warp():
    """
    Tests the forward and backward warp for cycle consistency.
    1. Creates a test image `image_a`.
    2. Generates a forward flow `flow_ab`.
    3. Warps `image_a` to get `image_b`.
    4. Generates a backward flow `flow_ba`.
    5. Warps `image_b` back to get `image_a_reconstructed`.
    6. Compares `image_a` with `image_a_reconstructed`.
    """
    print("--- Running Cycle Consistency Test ---")

    # Test parameters
    B, H, W = 1, 20, 20
    res = 9  # 9x9 grid
    C = res * res  # 81 classes
    device = "cpu"

    # 1. Create a simple test image (a white square on a black background)
    image_a = torch.zeros(B, 1, H, W, device=device)
    image_a[:, :, 5:15, 5:15] = 1.0
    print(f"Created a test image of size {H}x{W}.")

    # 2. Create a mock classification tensor to generate a simple flow
    # We'll make a flow that moves everything 2 pixels right and 1 pixel down
    # This corresponds to a normalized flow of:
    # x = 2 / ((W-1)/2) = 4 / 19 = 0.21
    # y = 1 / ((H-1)/2) = 2 / 19 = 0.105
    # We find the class in our grid `G` closest to [0.21, 0.105] and set its score high.
    # For a 9x9 grid, class 49 corresponds to [0.25, 0.125], which is close enough.
    cls = torch.randn(B, C, H, W, device=device) * 0.1
    cls[:, 42, :, :] = 10.0  # High score for the desired flow class

    # 3. Calculate the forward flow (A -> B)
    flow_ab = cls_to_flow_refine(cls, is_A_to_B=True)
    print("Generated forward flow (A -> B).")

    # 4. Warp image_a to image_b
    # IMPORTANT: The warp function needs flow in PIXEL units, not normalized [-1, 1] units.
    scaling_factor = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0], device=device).view(1, 1, 1, 2)
    flow_ab_pixels = (flow_ab * scaling_factor).permute(0, 3, 1, 2)

    image_b = warp(image_a, flow_ab_pixels)
    print("Warped image_a to image_b.")

    # 5. Calculate the backward flow (B -> A)
    # Note: get_backward_flow now handles the pixel/normalized conversion internally
    flow_ba = get_backward_flow(flow_ab)
    print("Generated backward flow (B -> A).")

    # 6. Warp image_b back to image_a_reconstructed
    flow_ba_pixels = (flow_ba * scaling_factor).permute(0, 3, 1, 2)
    image_a_reconstructed = warp(image_b, flow_ba_pixels)
    print("Warped image_b back to image_a_reconstructed.")

    # 7. Compare the original and reconstructed images
    mse = F.mse_loss(image_a, image_a_reconstructed)
    print("\n--- Test Result ---")
    print(f"Mean Squared Error (MSE) between original and reconstructed image: {mse.item():.6f}")

    if mse.item() < 0.01:
        print("SUCCESS: The reconstructed image is very similar to the original.")
    else:
        print("FAILURE: The reconstructed image differs significantly from the original.")
    print("---------------------\n")


if __name__ == "__main__":
    test_cycle_consistency_warp()

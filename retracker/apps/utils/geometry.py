"""Geometry utilities for point generation and filtering."""

from typing import Optional, Tuple
import torch


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = None,
    margin_ratio: float = 1.0 / 64,
) -> torch.Tensor:
    """
    Get a grid of points covering a rectangular region.
    
    Args:
        size: Grid size (creates size x size points)
        extent: (H, W) specifying the height and width of the rectangle
        center: Optional (c_y, c_x) specifying the center coordinates
        device: Target device for tensor
        margin_ratio: Margin ratio from border (default: 1/64)
        
    Returns:
        Tensor of shape (1, size*size, 2) with points in (x, y) format
    """
    if device is None:
        device = torch.device("cpu")
    
    # Handle single point case
    if size == 1:
        return torch.tensor(
            [extent[1] / 2, extent[0] / 2], 
            device=device
        )[None, None]
    
    # Set default center
    if center is None:
        center = [extent[0] / 2, extent[1] / 2]
    
    # Calculate ranges with margin
    margin = extent[1] * margin_ratio
    range_y = (
        margin - extent[0] / 2 + center[0], 
        extent[0] / 2 + center[0] - margin
    )
    range_x = (
        margin - extent[1] / 2 + center[1], 
        extent[1] / 2 + center[1] - margin
    )
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    
    # Stack and reshape to (1, N, 2) with (x, y) order
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def filter_queries_by_mask(
    queries: torch.Tensor, 
    seg_mask: torch.Tensor
) -> torch.Tensor:
    """
    Filter query points by segmentation mask.
    
    Args:
        queries: Tensor of shape (N, 3) in (t, x, y) order or (B, N, 3)
        seg_mask: Tensor of shape (H, W) or (1, 1, H, W)
        
    Returns:
        Boolean mask indicating valid queries
    """
    # Handle batch dimension
    if queries.dim() == 3:
        queries = queries[0]  # Remove batch dim for filtering
    
    # Ensure mask is 2D
    if seg_mask.dim() == 4:
        seg_mask = seg_mask[0, 0]
    elif seg_mask.dim() == 3:
        seg_mask = seg_mask[0]
    
    # Convert to long for indexing
    queries_long = queries.long()
    seg_mask = seg_mask.to(queries.device)
    
    # Filter by mask (queries are in t,x,y format, so index with y,x)
    valid_mask = seg_mask[queries_long[:, 2], queries_long[:, 1]] > 0
    
    return valid_mask

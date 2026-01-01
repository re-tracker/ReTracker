"""Query point generation module with multiple strategies."""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore

from ..config import QueryConfig
from ..utils import get_points_on_a_grid, filter_queries_by_mask
from retracker.utils.rich_utils import CONSOLE


class BaseQueryGenerator(ABC):
    """Abstract base class for query point generation."""
    
    def __init__(self, config: QueryConfig):
        self.config = config
    
    @abstractmethod
    def generate(
        self, 
        video: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate query points.
        
        Args:
            video: Video tensor of shape (B, T, C, H, W)
            seg_mask: Optional segmentation mask (B, 1, H, W) or (H, W)
            **kwargs: Additional arguments
            
        Returns:
            Query points tensor of shape (B, N, 3) in (t, x, y) format
        """
        pass


class GridQueryGenerator(BaseQueryGenerator):
    """Grid-based query point generation."""
    
    def generate(
        self, 
        video: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate uniform grid points."""
        B, T, C, H, W = video.shape
        device = video.device
        
        # Generate grid points (1, N, 2) in (x, y) format
        grid_pts = get_points_on_a_grid(
            size=self.config.grid_size,
            extent=(H, W),
            device=device,
            margin_ratio=self.config.margin_ratio,
        )
        
        # Add time dimension (initial frame)
        # Convert to (t, x, y) format
        queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * self.config.initial_frame,
            grid_pts
        ], dim=2)
        
        # Filter by segmentation mask if provided
        if seg_mask is not None and self.config.use_mask_filter:
            queries = self._filter_by_mask(queries, seg_mask)
        
        return queries
    
    def _filter_by_mask(
        self, 
        queries: torch.Tensor, 
        seg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Filter queries by segmentation mask."""
        # Get queries on the initial frame
        frame_queries = queries[queries[..., 0] == self.config.initial_frame]
        
        # Get valid mask
        valid_mask = filter_queries_by_mask(frame_queries, seg_mask)
        
        # Filter queries
        if queries.shape[0] == 1:  # Batch size 1
            queries = queries[:, valid_mask]
        else:
            queries = queries[valid_mask]
        
        return queries


class SegmentationQueryGenerator(BaseQueryGenerator):
    """Generate queries from segmentation mask."""
    
    def generate(
        self, 
        video: torch.Tensor,
        seg_mask: torch.Tensor,
        num_points: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample points from segmentation mask."""
        B, T, C, H, W = video.shape
        device = video.device
        
        # Ensure mask is 2D
        if seg_mask.dim() == 4:
            seg_mask = seg_mask[0, 0]
        elif seg_mask.dim() == 3:
            seg_mask = seg_mask[0]
        
        # Get valid pixel coordinates
        valid_coords = torch.nonzero(seg_mask > 0, as_tuple=False)  # (N, 2) in (y, x)
        
        if valid_coords.shape[0] == 0:
            raise ValueError("No valid points in segmentation mask")
        
        # Sample points if num_points specified
        if num_points is not None and valid_coords.shape[0] > num_points:
            indices = torch.randperm(valid_coords.shape[0])[:num_points]
            valid_coords = valid_coords[indices]
        
        # Convert to (t, x, y) format
        queries = torch.cat([
            torch.ones(valid_coords.shape[0], 1, device=device) * self.config.initial_frame,
            valid_coords[:, 1:2],  # x
            valid_coords[:, 0:1],  # y
        ], dim=1)
        
        # Add batch dimension
        queries = queries.unsqueeze(0)
        
        return queries


class CustomQueryGenerator(BaseQueryGenerator):
    """Custom query generator from file or coordinates."""
    
    def generate(
        self, 
        video: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
        query_path: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """Load custom query points."""
        device = video.device
        
        if query_coords is not None:
            queries = query_coords
        elif query_path is not None:
            # Load from file (implement based on format)
            queries = self._load_queries_from_file(query_path)
        else:
            raise ValueError("Either query_coords or query_path must be provided")
        
        # Ensure proper shape and device
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
        
        return queries.to(device)
    
    def _load_queries_from_file(self, query_path: str) -> torch.Tensor:
        """Load queries from file (NPZ, TXT, etc.)."""
        import numpy as np
        
        if query_path.endswith('.npz'):
            data = np.load(query_path)
            queries = torch.from_numpy(data['queries'])
        elif query_path.endswith('.txt'):
            queries = torch.from_numpy(np.loadtxt(query_path))
        else:
            raise ValueError(f"Unsupported query file format: {query_path}")
        
        return queries


class SIFTQueryGenerator(BaseQueryGenerator):
    """SIFT keypoint-based query point generation."""

    def __init__(self, config: QueryConfig):
        super().__init__(config)
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for SIFT query generation. Ensure `opencv-python` is installed and compatible."
            )
        # Create SIFT detector with config parameters
        self.sift = cv2.SIFT_create(
            nfeatures=config.sift_n_features,
            nOctaveLayers=config.sift_n_octave_layers,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=config.sift_sigma,
        )

    def generate(
        self,
        video: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        max_points: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate query points using SIFT keypoint detection.

        Args:
            video: Video tensor of shape (B, T, C, H, W)
            seg_mask: Optional segmentation mask (B, 1, H, W) or (H, W)
            max_points: Maximum number of points to return (sample if exceeded)

        Returns:
            Query points tensor of shape (B, N, 3) in (t, x, y) format
        """
        B, T, C, H, W = video.shape
        device = video.device

        # Get the initial frame
        initial_frame_idx = self.config.initial_frame
        frame = video[0, initial_frame_idx]  # (C, H, W)

        # Convert to numpy for SIFT
        frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

        # Convert to uint8 grayscale for SIFT
        if frame_np.max() > 1.0:
            frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = (frame_np * 255).astype(np.uint8)

        if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_np

        # Detect SIFT keypoints
        keypoints = self.sift.detect(gray, None)

        if len(keypoints) == 0:
            CONSOLE.print("[yellow]Warning: No SIFT keypoints detected, falling back to grid[/yellow]")
            # Fallback to grid
            from ..utils import get_points_on_a_grid
            grid_pts = get_points_on_a_grid(
                size=self.config.grid_size,
                extent=(H, W),
                device=device,
                margin_ratio=self.config.margin_ratio,
            )
            queries = torch.cat([
                torch.ones_like(grid_pts[:, :, :1]) * initial_frame_idx,
                grid_pts
            ], dim=2)
            return queries

        # Extract keypoint coordinates
        coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])  # (N, 2) in (x, y)

        # Sort by response (strength) in descending order
        responses = np.array([kp.response for kp in keypoints])
        sorted_indices = np.argsort(responses)[::-1]
        coords = coords[sorted_indices]

        # Sample if too many points
        if max_points is not None and len(coords) > max_points:
            # Keep top max_points by response (already sorted)
            coords = coords[:max_points]

        # Convert to tensor
        coords_tensor = torch.from_numpy(coords).float().to(device)  # (N, 2)

        # Create queries in (t, x, y) format
        queries = torch.cat([
            torch.ones(coords_tensor.shape[0], 1, device=device) * initial_frame_idx,
            coords_tensor,
        ], dim=1)

        # Add batch dimension
        queries = queries.unsqueeze(0)  # (1, N, 3)

        # Filter by segmentation mask if provided
        if seg_mask is not None and self.config.use_mask_filter:
            queries = self._filter_by_mask(queries, seg_mask)

        return queries

    def _filter_by_mask(
        self,
        queries: torch.Tensor,
        seg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Filter queries by segmentation mask."""
        # Ensure mask is 2D
        if seg_mask.dim() == 4:
            seg_mask = seg_mask[0, 0]
        elif seg_mask.dim() == 3:
            seg_mask = seg_mask[0]

        # Get query positions
        query_xy = queries[0, :, 1:3]  # (N, 2) in (x, y)

        # Check which points are within mask
        valid_mask = []
        H, W = seg_mask.shape
        for i in range(query_xy.shape[0]):
            x, y = int(query_xy[i, 0]), int(query_xy[i, 1])
            if 0 <= x < W and 0 <= y < H:
                valid_mask.append(seg_mask[y, x] > 0)
            else:
                valid_mask.append(False)

        valid_mask = torch.tensor(valid_mask, device=queries.device)
        queries = queries[:, valid_mask]

        return queries


class QueryGeneratorFactory:
    """Factory for creating query generators."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, generator_class):
        """Register a custom query generator."""
        cls._registry[name] = generator_class
    
    @classmethod
    def create(cls, config: QueryConfig) -> BaseQueryGenerator:
        """
        Create query generator based on config.
        
        Args:
            config: QueryConfig instance
            
        Returns:
            Query generator instance
        """
        generators = {
            'grid': GridQueryGenerator,
            'segmentation': SegmentationQueryGenerator,
            'custom': CustomQueryGenerator,
            'sift': SIFTQueryGenerator,
            **cls._registry,
        }
        
        generator_class = generators.get(config.strategy)
        if generator_class is None:
            raise ValueError(f"Unknown query strategy: {config.strategy}")
        
        return generator_class(config)

"""
Detection module for automatic object detection.

Future implementation: YOLO, DINO, etc. for query generation
"""

import torch
from typing import List, Dict, Optional


class AutoDetectionModule:
    """
    Automatic detection module (placeholder for future).

    Future work: Integration with YOLO, DINO, or similar detection models
    is planned for a future release.
    """
    
    def __init__(self, model_type: str = 'yolo'):
        self.model_type = model_type
        self.model = None  # Load model here
    
    def detect(
        self, 
        frame: torch.Tensor,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Frame tensor (C, H, W)
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections with boxes and classes
        """
        raise NotImplementedError("Auto-detection not yet implemented")

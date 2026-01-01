"""Compatibility shim for the DINOv2 vision transformer implementation.

The canonical implementation lives in:
`retracker.models.backbone.models.vision_transformer`

Historically, some code imported these symbols from `retracker.models.backbone.dinov2`.
To avoid breaking those call sites, keep this module as a thin re-export layer.
"""

from __future__ import annotations

from .models import vision_transformer as _vits


# Re-export public helpers/classes from the canonical module.
named_apply = _vits.named_apply
BlockChunk = _vits.BlockChunk
DinoVisionTransformer = _vits.DinoVisionTransformer
init_weights_vit_timm = _vits.init_weights_vit_timm
vit_small = _vits.vit_small
vit_base = _vits.vit_base
vit_large = _vits.vit_large
vit_giant2 = _vits.vit_giant2

__all__ = [
    "BlockChunk",
    "DinoVisionTransformer",
    "init_weights_vit_timm",
    "named_apply",
    "vit_base",
    "vit_giant2",
    "vit_large",
    "vit_small",
]

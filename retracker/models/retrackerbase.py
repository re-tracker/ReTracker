import torch
import torch.nn as nn

from .backbone import build_backbone, build_dino_backbone
from .modules.coarse_matching import CoarseMatching_CLS_selected
from .modules.pips_refinement import MatchesRefinement
from .modules.temporal_fusion import TemporalFusion
from .modules.transformer import TransformerDecoder
from .utils.dino_encoder import DinoEncoder
from .utils.kernel_functions import CosKernel
from .utils.mem_manager.memorymanager import MemoryManager
from .utils.position_encoding import PositionEncoding1D


class ReTrackerBase(nn.Module):
    def __init__(self, config):
        super().__init__()

        # config
        self.config = config
        self.winsz_f = config["fine_window_size"]
        self.chn_d = config["chn_d"]
        self.chn_c = config["chn_c"]
        self.chn_f = config["chn_f"]
        self.sliding_wz = config["sliding_wz"]
        self.occ_thresh = config["occ_thresh"]
        self.corr_num_levels = config["pips_refinement"]["corr_num_levels"]
        self.corr_radius = config["pips_refinement"]["corr_radius"]

        # Modules
        self.K = CosKernel(T=0.2)
        self.mem_manager = MemoryManager(config["memory_manager"])
        self._build_backbone(config)
        self._build_global_roma(config)
        self.matches_refinement = MatchesRefinement(
            self.mem_manager,
            "pips",
            use_last_mem=False,
            config=config["pips_refinement"],
        )

    def forward(self, data):
        raise NotImplementedError

    def _build_backbone(self, config):
        """Build backbone(s). When using DINOv3, only one backbone is needed."""
        # Auto-configure backbone_type based on dino_version if provided
        dino_version = config.get("dino_version", None)
        if dino_version:
            if dino_version == "v3":
                config["backbone_type"] = "DINOv3"
            elif dino_version == "v2":
                config["backbone_type"] = "ResNetFPN"
            else:
                raise ValueError(f"Unknown dino_version: {dino_version}. Must be 'v2' or 'v3'")

        self.backbone_type = config.get("backbone_type", "ResNetFPN")
        self.use_dinov3 = self.backbone_type == "DINOv3"

        if self.use_dinov3:
            # DINOv3 can output all levels (16x, 8x, 2x), so we only need one backbone
            self.backbone = build_backbone(config)
            self.dino_backbone = None  # Not needed when using DINOv3
        else:
            # Traditional setup: separate backbone for CNN features and DINO features
            self.backbone = build_backbone(config)
            self.dino_backbone = build_dino_backbone(config)

    def _build_global_roma(self, config):
        """build global tracking module"""
        self.dino_encoder = DinoEncoder(config["dino_encoder"])
        self.pos_conv = torch.nn.Conv2d(2, config.dino.d_model, 1, 1)
        self.decoder_coarse_pos_encoding = PositionEncoding1D(config.coarse.d_model)
        self.dino_decoder = TransformerDecoder(config.dino_decoder)

        ### TYPE3 mlp decoder
        self.cls_decoder = nn.Sequential(
            nn.LayerNorm(config.dino_decoder.d_model),
            nn.Linear(config.dino_decoder.d_model, config.dino_decoder.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.dino_decoder.d_model, config.cls_decoder.num_classes + 1),
        )

        for m in self.cls_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        last_layer = self.cls_decoder[-1]
        nn.init.normal_(last_layer.weight, std=0.01)

        self.certainty_decoder = nn.Linear(config.dino_decoder.d_model, 1)
        self.occlusion_decoder = nn.Linear(config.dino_decoder.d_model, 1)
        self.temporal_attn = TemporalFusion(
            config.temporal_fusion_dino, config.chn_d, module_id="dino"
        )
        self.coarse_matching_by_cls = CoarseMatching_CLS_selected(config["coarsematching_cls"])

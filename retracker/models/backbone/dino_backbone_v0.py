import logging
import os
from pathlib import Path

import torch
from torch import nn

from retracker.utils.checkpoint import safe_torch_load

from .hub.backbone import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg


logger = logging.getLogger(__name__)


model_dict = {
    "dinov2_vits14_reg4": {
        "build_func": dinov2_vits14_reg,
        "embedding_dim": 384,
    },
    "dinov2_vitb14_reg4": {
        "build_func": dinov2_vitb14_reg,
        "embedding_dim": 768,
    },
    "dinov2_vitl14_reg4": {
        "build_func": dinov2_vitl14_reg,
        "embedding_dim": 1024,
    },
    "dinov2_vitg14_reg4": {
        "build_func": dinov2_vitg14_reg,
        "embedding_dim": 1536,
    },
}


class DINO_backbone(nn.Module):
    _config = {
        "arch": "dinov2_vitl14_reg4",
        "download_online": True,
        "dino_weights_path": None,
        "hid_dim": 384,
        "d_model": 384,  # out_dim
    }

    def __init__(self, config):
        """"""
        super().__init__()
        config = {**self._config, **config}
        self.dino = None
        self.res_refiner = None
        self.dino_arch = config["arch"]
        assert self.dino_arch in model_dict.keys(), "undefined pretrained dinov2 model"

        self.emb_dim = model_dict[self.dino_arch]["embedding_dim"]
        self.hid_dim = config["hid_dim"]
        self.out_dim = config["d_model"]

        # from emb_dim to out_dim directly
        self.proj = nn.Sequential(
            nn.Conv2d(self.emb_dim, self.out_dim, 1, 1), nn.BatchNorm2d(self.out_dim)
        )

        self.use_trainable_residual = config["use_trainable_residual"]
        if self.use_trainable_residual:
            self.res_refiner = nn.Sequential(
                nn.Conv2d(self.emb_dim, self.hid_dim, 3, 1, 1),
                nn.BatchNorm2d(self.hid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hid_dim, self.out_dim, 3, 1, 1),
                nn.BatchNorm2d(self.out_dim),
            )

        if config["download_online"]:
            self.dino = model_dict[self.dino_arch]["build_func"](pretrained=True)
        else:
            self.dino = model_dict[self.dino_arch]["build_func"](pretrained=False)
            PROJECT_DIR = str(os.getenv("PROJECT_DIR"))
            if PROJECT_DIR == "None":
                # PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
            dino_weights_dir = (
                os.path.join(PROJECT_DIR, "weights")
                if config["dino_weights_path"] is None
                else config["dino_weights_path"]
            )
            dino_weights_path = os.path.join(dino_weights_dir, self.dino_arch + "_pretrain.pth")
            state_dict = safe_torch_load(dino_weights_path, map_location="cpu", weights_only=True)
            self.dino.load_state_dict(state_dict)

        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
        logger.info(f"[INFO] DINO backbone '{self.dino_arch}' is frozen and set to eval mode.")

    def forward(self, x):
        """
        Args:
            x(torch.Tensor): [B, C, H, W], C=3
        """
        with torch.no_grad():
            # resize input to produce same output
            _, _C, _H, _W = x.shape
            if _C == 1:  # tmp solve for gray images
                x = x.repeat(1, 3, 1, 1)

            x = nn.functional.interpolate(x, size=(448, 448), mode="bilinear", align_corners=True)
            B, C, H, W = x.shape

            x = self.dino.forward_features(x)["x_norm_patchtokens"]
            x = x.permute(0, 2, 1).reshape(B, self.emb_dim, H // 14, W // 14)  # B, 384, 32, 32
            x_emb = x
        x = self.proj(x)
        if self.use_trainable_residual:
            x = x + self.res_refiner(x_emb)
        return x

import os

import torch
from torch import nn

from retracker.utils.checkpoint import safe_torch_load

from .dino_backbone import DINO_backbone as DINO_backbone_adaptor
from .dino_backbone_v0 import DINO_backbone as DINO_backbone
from .dino_backbone_v3 import (
    DINOv3_backbone_gatevitcnn,
    DINOv3_backbone_vitcnn,
    DINOv3_backbone_vitconvnext,
)
from .resnet_fpn import ResNetFPN_2, ResNetFPN_4, ResNetFPN_4_2, ResNetFPN_8_2, ResNetFPN_16_4


def _torchvision_models():
    """Import torchvision lazily.

    Torchvision is only required for legacy CNN backbones (e.g. ResNet/VGG) and
    should not be a hard import-time dependency for DINOv3-only inference.
    """
    try:
        from torchvision import models
    except Exception as exc:
        raise ImportError(
            "torchvision is required for `pretrained_resnet18` / `VGG16_BN` backbones. "
            "Install torchvision or switch to the DINOv3 backbone."
        ) from exc
    return models


class Pretrained_ResNet18(nn.Module):
    def __init__(self, config):
        super().__init__()
        models = _torchvision_models()
        self.resnet18 = models.resnet18(pretrained=config["use_pretrained"])
        if not config["use_pretrained"]:
            state_dict = safe_torch_load(config["weights_path"], map_location="cpu", weights_only=True)
            self.resnet18.load_state_dict(state_dict, strict=True)

        self.resnet18.eval()
        self.feature_maps = {}
        self.register_feature_hook()

    def forward(self, x):
        self.resnet18(x)
        return self.feature_maps["layer2"], self.feature_maps["layer1"]

    def register_feature_hook(self):
        feature_maps = self.feature_maps

        def get_features(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()

            return hook

        layer_names = [
            "layer1",  # 4x downsampling
            "layer2",  # 8x downsampling
            "layer3",  # 16x downsampling
            "layer4",  # 32x downsampling
        ]

        for name in layer_names:
            self.resnet18._modules.get(name).register_forward_hook(get_features(name))


class Pretrained_VGG16(nn.Module):
    def __init__(self, config):
        super().__init__()
        models = _torchvision_models()
        self.vgg16 = models.vgg16_bn(pretrained=config["download_pretrained_online"])
        if not config["download_pretrained_online"]:
            # Prefer an explicit full path, otherwise interpret relative paths from
            # PROJECT_DIR (if set) or the current working directory.
            weights_path = config["weights_path"]
            if os.path.isabs(weights_path):
                full_path = weights_path
            else:
                base_path = os.getenv("PROJECT_DIR") or os.getcwd()
                full_path = os.path.join(base_path, weights_path)

            state_dict = safe_torch_load(full_path, map_location="cpu", weights_only=True)
            self.vgg16.load_state_dict(state_dict, strict=True)

        self.vgg16.eval()
        self.feature_maps = {}
        self.register_feature_pre_hook()

    def forward(self, x):
        self.vgg16(x)
        return self.feature_maps["22"], self.feature_maps["8"]

    def register_feature_pre_hook(self):
        feature_maps = self.feature_maps

        def get_features(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()

            return hook

        idx_list = [3, 8, 15, 22, 29]
        for idx, layer in enumerate(self.vgg16.features):
            if idx in idx_list:
                layer.register_forward_hook(get_features(str(idx)))


def build_backbone(config):
    backbone = None
    if config["backbone_type"] == "ResNetFPN":
        if config["resolution"] == (4, 2):
            backbone = ResNetFPN_4_2(config["resnetfpn"])
        if config["resolution"] == (8, 2):
            backbone = ResNetFPN_8_2(config["resnetfpn"])
        elif config["resolution"] == (16, 4):
            backbone = ResNetFPN_16_4(config["resnetfpn"])
        elif config["resolution"] == 4:
            backbone = ResNetFPN_4(config["resnetfpn"])
        elif config["resolution"] == 2:
            backbone = ResNetFPN_2(config["resnetfpn"])
    elif config["backbone_type"] == "DINOv3":
        # Use DINOv3 to produce multi-scale outputs (16x, 8x, 2x)
        dino_cfg = config.get("dino", {})
        dinov3_cfg = {
            "repo_dir": dino_cfg.get("repo_dir", "retracker/src/core/backbone/dinov3"),
            "github_repo": dino_cfg.get("github_repo", "facebookresearch/dinov3"),
            "model_name": dino_cfg.get("model_name", "dinov3_vitl16"),
            "weights_path": dino_cfg.get(
                "weights_path", "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            ),
            "source": dino_cfg.get("source", "auto"),
            "download_online": dino_cfg.get("download_online", True),
            "patch_size": dino_cfg.get("patch_size", 16),
            "d_model_16": config.get("chn_d", 384),
            "d_model_8": config.get("chn_c", 256),
            "d_model_2": config.get("chn_f", 128),
            "return_layers": dino_cfg.get("return_layers", 1),
            "use_norm": dino_cfg.get("use_norm", True),
            "return_type": "multi",  # Return all three scales (16x, 8x, 2x)
        }
        # Optional CNN branch controls (used by hybrid DINOv3 backbones).
        # We only forward keys when explicitly provided so we don't override
        # each backbone's internal defaults.
        if "use_cnn_for_fine_features" in dino_cfg:
            dinov3_cfg["use_cnn_for_fine_features"] = dino_cfg.get("use_cnn_for_fine_features")
        if "use_fusion" in dino_cfg:
            dinov3_cfg["use_fusion"] = dino_cfg.get("use_fusion")
        if "cnn_model_name" in dino_cfg and dino_cfg.get("cnn_model_name") is not None:
            dinov3_cfg["cnn_model_name"] = dino_cfg.get("cnn_model_name")
        if "cnn_weights_path" in dino_cfg and dino_cfg.get("cnn_weights_path") is not None:
            dinov3_cfg["cnn_weights_path"] = dino_cfg.get("cnn_weights_path")
        if "gate_hidden_dim" in dino_cfg and dino_cfg.get("gate_hidden_dim") is not None:
            dinov3_cfg["gate_hidden_dim"] = dino_cfg.get("gate_hidden_dim")
        if config["cnn_fusion_type"] == "vitcnn":
            backbone = DINOv3_backbone_vitcnn(dinov3_cfg)
        elif config["cnn_fusion_type"] == "gatevitcnn":
            backbone = DINOv3_backbone_gatevitcnn(dinov3_cfg)
        elif config["cnn_fusion_type"] == "vitconvnext":
            backbone = DINOv3_backbone_vitconvnext(dinov3_cfg)
        else:
            raise ValueError(f"Fusion type {config['cnn_fusion_type']} not supported.")
    elif config["backbone_type"] == "pretrained_resnet18":
        backbone = Pretrained_ResNet18(config["resnet18"])
        backbone.eval()
    elif config["backbone_type"] == "VGG16_BN":
        backbone = Pretrained_VGG16(config["vgg16_bn"])
    else:
        raise ValueError(f"BACKBONE_TYPE {config['backbone_type']} not supported.")
    return backbone


def build_dino_backbone(config):
    """Return Dino backbone
    1. Resize images to image / 8 * 14
    2. forward & get (B, C, 32, 32)

    When using DINOv3 (backbone_type == 'DINOv3'), returns None because
    all features (16x, 8x, 2x) are extracted from self.backbone.
    """
    # If using DINOv3 as the main backbone, we don't need a separate dino_backbone
    if config.get("backbone_type") == "DINOv3":
        return None

    dino_cfg = config["dino"]
    use_transformers = dino_cfg.get("use_transformers", False) or ("model_name" in dino_cfg)
    if use_transformers:
        # Use DINOv3 for dino features only (when backbone_type != 'DINOv3')
        dinov3_cfg = {
            "repo_dir": dino_cfg.get("repo_dir", "retracker/src/core/backbone/dinov3"),
            "github_repo": dino_cfg.get("github_repo", "facebookresearch/dinov3"),
            "model_name": dino_cfg.get("model_name", "dinov3_vitl16"),
            "weights_path": dino_cfg.get(
                "weights_path", "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            ),
            "source": dino_cfg.get("source", "auto"),
            "download_online": dino_cfg.get("download_online", True),
            "patch_size": dino_cfg.get("patch_size", 16),
            "d_model_16": config.get("chn_d", 384),
            "d_model_8": config.get("chn_c", 256),
            "d_model_2": config.get("chn_f", 128),
            "return_layers": dino_cfg.get("return_layers", 1),
            "use_norm": dino_cfg.get("use_norm", True),
            "return_type": "dino",  # Only return 16x features for dino backbone
        }
        # Optional CNN branch controls (used by hybrid DINOv3 backbones).
        if "use_cnn_for_fine_features" in dino_cfg:
            dinov3_cfg["use_cnn_for_fine_features"] = dino_cfg.get("use_cnn_for_fine_features")
        if "use_fusion" in dino_cfg:
            dinov3_cfg["use_fusion"] = dino_cfg.get("use_fusion")
        if "cnn_model_name" in dino_cfg and dino_cfg.get("cnn_model_name") is not None:
            dinov3_cfg["cnn_model_name"] = dino_cfg.get("cnn_model_name")
        if "cnn_weights_path" in dino_cfg and dino_cfg.get("cnn_weights_path") is not None:
            dinov3_cfg["cnn_weights_path"] = dino_cfg.get("cnn_weights_path")
        if "gate_hidden_dim" in dino_cfg and dino_cfg.get("gate_hidden_dim") is not None:
            dinov3_cfg["gate_hidden_dim"] = dino_cfg.get("gate_hidden_dim")
        if config["cnn_fusion_type"] == "vitcnn":
            backbone = DINOv3_backbone_vitcnn(dinov3_cfg)
        elif config["cnn_fusion_type"] == "gatevitcnn":
            backbone = DINOv3_backbone_gatevitcnn(dinov3_cfg)
        elif config["cnn_fusion_type"] == "vitconvnext":
            backbone = DINOv3_backbone_vitconvnext(dinov3_cfg)
        else:
            raise ValueError(f"Fusion type {config['cnn_fusion_type']} not supported.")
    else:
        # Traditional DINOv2 setup
        if config["dino"]["use_adaptor"]:
            backbone = DINO_backbone_adaptor(config["dino"])
        else:
            backbone = DINO_backbone(config["dino"])
    return backbone

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from retracker.utils.checkpoint import safe_torch_load


logger = logging.getLogger(__name__)


def _torchvision_models():
    """Import torchvision lazily.

    Torchvision is only required when the optional CNN branch is enabled
    (`use_cnn_for_fine_features=True`).
    """
    try:
        import torchvision.models as models
    except Exception as exc:
        raise ImportError(
            "torchvision is required for the CNN branch of the DINOv3 backbone. "
            "Install torchvision or set `use_cnn_for_fine_features=False` in the config."
        ) from exc
    return models


class DINOv3_backbone_vitcnn(nn.Module):
    """
    Hybrid Backbone with optional Feature Fusion.
    - DINOv3 branch provides 16x semantic features.
    - CNN branch provides 8x and 2x detailed features.
    - If `use_fusion` is True, the 16x ViT feature is upsampled and fused with
      the CNN features at the 8x and 2x levels, enriching them with global context.
    """

    _default_config = {
        "repo_dir": "retracker/models/backbone/dinov3",
        "github_repo": "facebookresearch/dinov3",
        "model_name": "dinov3_vitl16",
        "weights_path": "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "source": "local",
        "download_online": True,
        "patch_size": 16,
        "d_model_16": 384,
        "d_model_8": 256,
        "d_model_2": 128,
        "return_layers": 1,
        "use_norm": True,
        "return_type": "multi",
        "img_mean": (0.485, 0.456, 0.406),
        "img_std": (0.229, 0.224, 0.225),
        "use_cnn_for_fine_features": True,
        "cnn_model_name": "resnet18",
        ### MODIFICATION ###
        # New flag to enable the fusion of ViT and CNN features.
        "use_fusion": True,
        "cnn_weights_path": "weights/resnet18-f37072fd.pth",
    }

    def __init__(self, config: dict):
        super().__init__()
        cfg = {**self._default_config, **(config or {})}

        # --- (Existing config parsing) ---
        self.repo_dir: str | None = cfg["repo_dir"] or os.getenv("DINOV3_LOCATION", None)
        self.github_repo: str = cfg["github_repo"]
        self.model_name: str = cfg["model_name"]
        self.weights_path: str | None = cfg["weights_path"]
        # When False, build the backbone architecture without loading hub weights.
        # Useful for fast-start inference when a full ReTracker checkpoint is loaded next.
        self.pretrained: bool = bool(cfg.get("pretrained", True))
        self.source: str = cfg["source"]
        self.download_online: bool = cfg["download_online"]
        self.patch_size: int = int(cfg["patch_size"])
        self.out16: int = int(cfg["d_model_16"])
        self.out8: int = int(cfg["d_model_8"])
        self.out2: int = int(cfg["d_model_2"])
        self.return_layers: int | list[int] = cfg["return_layers"]
        self.use_norm: bool = bool(cfg["use_norm"])
        self.return_type: str = cfg.get("return_type", cfg.get("mode", "multi"))
        self.img_mean = torch.tensor(cfg["img_mean"]).view(1, 3, 1, 1)
        self.img_std = torch.tensor(cfg["img_std"]).view(1, 3, 1, 1)
        self.use_cnn_for_fine_features = cfg["use_cnn_for_fine_features"]
        self.use_fusion = cfg["use_fusion"]
        self.cnn_weights_path = self._resolve_weights_path(cfg["cnn_weights_path"])

        self.model = self._build_model()
        self.embed_dim = getattr(self.model, "embed_dim", 1024)
        self.proj16 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out16, 1, bias=False),
            nn.BatchNorm2d(self.out16),
        )
        # Projections used when the optional CNN branch is disabled.
        self.vit_proj8 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out8, 1, bias=False),
            nn.BatchNorm2d(self.out8),
        )
        self.vit_proj2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out2, 1, bias=False),
            nn.BatchNorm2d(self.out2),
        )

        # --- Conditional setup for fine feature extraction ---
        if self.use_cnn_for_fine_features:
            models = _torchvision_models()
            logger.info("Building hybrid backbone with CNN for fine features.")

            ### MODIFICATION ###
            # Change how the CNN model is loaded to support offline weights.
            if cfg["cnn_model_name"] == "resnet18":
                cnn_out_channels_8x, cnn_out_channels_2x = 128, 64
                # 1. Instantiate the model architecture WITHOUT pre-trained weights
                cnn_base = models.resnet18(weights=None)

                # 2. Check if a local weights path is provided and exists
                if self.cnn_weights_path and os.path.exists(self.cnn_weights_path):
                    logger.info(f"Loading local CNN weights from: {self.cnn_weights_path}")
                    # 3. Load the state dictionary from the local file
                    state_dict = safe_torch_load(self.cnn_weights_path, map_location="cpu", weights_only=True)
                    cnn_base.load_state_dict(state_dict)
                else:
                    # Fallback to online downloading if the local file is not found
                    logger.warning("Local CNN weights not found. Attempting to download online.")
                    cnn_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:  # Example for ResNet34
                # (You can apply the same logic here for other models)
                cnn_out_channels_8x, cnn_out_channels_2x = 128, 64
                cnn_base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

            self.cnn_conv1, self.cnn_bn1, self.cnn_relu = (
                cnn_base.conv1,
                cnn_base.bn1,
                cnn_base.relu,
            )
            self.cnn_maxpool, self.cnn_layer1, self.cnn_layer2 = (
                cnn_base.maxpool,
                cnn_base.layer1,
                cnn_base.layer2,
            )

            ### MODIFICATION ###
            if self.use_fusion:
                logger.info("Feature fusion between ViT and CNN is enabled.")
                # Fusion layers to process concatenated features (CNN + ViT)
                # For 8x features:
                self.fusion_conv8 = nn.Sequential(
                    nn.Conv2d(
                        cnn_out_channels_8x + self.embed_dim,
                        self.out8,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out8),
                    nn.ReLU(inplace=True),
                )
                # For 2x features:
                self.fusion_conv2 = nn.Sequential(
                    nn.Conv2d(
                        cnn_out_channels_2x + self.embed_dim,
                        self.out2,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out2),
                    nn.ReLU(inplace=True),
                )
            else:
                logger.info("Feature fusion is disabled.")
                # Original projection layers (only process CNN features)
                self.proj8 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_8x, self.out8, 1, bias=False),
                    nn.BatchNorm2d(self.out8),
                )
                self.proj2 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_2x, self.out2, 1, bias=False),
                    nn.BatchNorm2d(self.out2),
                )

            # Freeze CNN layers
            for p in cnn_base.parameters():
                p.requires_grad = False

        else:  # Original interpolation logic
            logger.info("Building backbone with interpolation for fine features.")
            self.proj8 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out8, 1, bias=False), nn.BatchNorm2d(self.out8)
            )
            self.proj2 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out2, 1, bias=False), nn.BatchNorm2d(self.out2)
            )

        # Freeze DINOv3 model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # --- (Helper methods _resolve_source, etc. remain unchanged) ---
    def _resolve_weights_path(self, weights_path: str | None) -> str | None:
        """Resolve weights path to absolute path if it's relative."""
        if not weights_path:
            return None
        if os.path.isabs(weights_path):
            return weights_path
        # Try to find project root (where the weights/ symlink should be)
        current_file = os.path.abspath(__file__)
        project_root = current_file
        for _ in range(10):  # Search up to 10 levels
            project_root = os.path.dirname(project_root)
            weights_candidate = os.path.join(project_root, weights_path)
            if os.path.exists(weights_candidate):
                return weights_candidate
        # Fallback: return original relative path
        return weights_path

    def _resolve_source(self):  # ... (no change) ...
        if self.source in ("local", "github"):
            return self.source
        if self.repo_dir is not None and os.path.isdir(self.repo_dir):
            return "local"
        return "github"

    def _build_model(self):  # ... (no change) ...
        repo = self.github_repo
        src = self._resolve_source()
        if src == "local":
            repo = self.repo_dir
        # NOTE: Some hub implementations will still load weights if `weights=...` is passed,
        # even when `pretrained=False`.
        kwargs = {"pretrained": self.pretrained}
        if self.pretrained and self.weights_path:
            kwargs["weights"] = self.weights_path
        try:
            model = torch.hub.load(repo, self.model_name, source=src, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load hub model '{self.model_name}' from '{repo}' (source={src}): {e}"
            ) from e
        return model

    def _check_size(self, x: torch.Tensor):  # ... (no change) ...
        h, w = x.shape[-2], x.shape[-1]
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            pass

    def _normalize(self, x: torch.Tensor):  # ... (no change) ...
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        device = x.device
        mean = self.img_mean.to(device)
        std = self.img_std.to(device)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._normalize(x)

        if self.use_cnn_for_fine_features:
            # --- HYBRID FORWARD ---
            with torch.no_grad():
                # 1. Get raw 16x ViT feature (for global context)
                self._check_size(x)
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat16_raw = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list

                # 2. Get raw CNN features (for local details)
                c1 = self.cnn_relu(self.cnn_bn1(self.cnn_conv1(x_in)))  # 2x feature
                c4 = self.cnn_layer2(self.cnn_layer1(self.cnn_maxpool(c1)))  # 8x feature

            # 3. Project 16x feature
            out16 = self.proj16(feat16_raw)

            if self.return_type == "dino":
                return out16

            ### MODIFICATION ###
            if self.use_fusion:
                # --- Fusion Logic ---
                # Fuse for 8x level
                vit_feat_for_8x = F.interpolate(
                    feat16_raw, size=c4.shape[-2:], mode="bilinear", align_corners=False
                )
                fused_8x = torch.cat([c4, vit_feat_for_8x], dim=1)
                out8 = self.fusion_conv8(fused_8x)

                # Fuse for 2x level
                vit_feat_for_2x = F.interpolate(
                    feat16_raw, size=c1.shape[-2:], mode="bilinear", align_corners=False
                )
                fused_2x = torch.cat([c1, vit_feat_for_2x], dim=1)
                out2 = self.fusion_conv2(fused_2x)
            else:
                # --- Non-Fusion Logic (original hybrid) ---
                out8 = self.proj8(c4)
                out2 = self.proj2(c1)

            return out16, out8, out2

        else:
            # --- ORIGINAL INTERPOLATION FORWARD (no changes here) ---
            # ... (your original interpolation logic) ...
            self._check_size(x)
            with torch.no_grad():
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list
            B, C, Hp, Wp = feat.shape
            Hin, Win = x.shape[-2:]
            H16, W16, H8, W8, H2, W2 = Hin // 16, Win // 16, Hin // 8, Win // 8, Hin // 2, Win // 2
            feat_16 = (
                F.interpolate(feat, (H16, W16), mode="bilinear", align_corners=True)
                if (Hp, Wp) != (H16, W16)
                else feat
            )
            out16 = self.proj16(feat_16)
            if self.return_type == "dino":
                return out16
            feat_8 = F.interpolate(feat, (H8, W8), mode="bilinear", align_corners=True)
            feat_2 = F.interpolate(feat, (H2, W2), mode="bilinear", align_corners=True)
            out8, out2 = self.vit_proj8(feat_8), self.vit_proj2(feat_2)
            return out16, out8, out2


class DINOv3_backbone_gatevitcnn(nn.Module):
    """
    Hybrid Backbone with optional Feature Fusion.
    - DINOv3 branch provides 16x semantic features.
    - CNN branch provides 8x and 2x detailed features.
    - If `use_fusion` is True, a dynamic gating mechanism adaptively fuses
      ViT and CNN features based on input content.
    """

    _default_config = {
        "repo_dir": "retracker/models/backbone/dinov3",
        "github_repo": "facebookresearch/dinov3",
        "model_name": "dinov3_vitl16",
        "weights_path": "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "source": "auto",
        "download_online": True,
        "patch_size": 16,
        "d_model_16": 384,
        "d_model_8": 256,
        "d_model_2": 128,
        "return_layers": 1,
        "use_norm": True,
        "return_type": "multi",
        "img_mean": (0.485, 0.456, 0.406),
        "img_std": (0.229, 0.224, 0.225),
        "use_cnn_for_fine_features": True,
        "cnn_model_name": "resnet18",
        "use_fusion": True,
        "cnn_weights_path": "weights/resnet18-f37072fd.pth",
        ### NEW ###
        # Hidden dimension for the gating network. A smaller value is more efficient.
        "gate_hidden_dim": 64,
    }

    def __init__(self, config: dict):
        super().__init__()
        cfg = {**self._default_config, **(config or {})}

        # --- (Existing config parsing, no changes) ---
        self.repo_dir: str | None = cfg["repo_dir"] or os.getenv("DINOV3_LOCATION", None)
        self.github_repo: str = cfg["github_repo"]
        self.model_name: str = cfg["model_name"]
        self.weights_path: str | None = cfg["weights_path"]
        # When False, build the backbone architecture without loading hub weights.
        # Useful for fast-start inference when a full ReTracker checkpoint is loaded next.
        self.pretrained: bool = bool(cfg.get("pretrained", True))
        self.source: str = cfg["source"]
        self.download_online: bool = cfg["download_online"]
        self.patch_size: int = int(cfg["patch_size"])
        self.out16: int = int(cfg["d_model_16"])
        self.out8: int = int(cfg["d_model_8"])
        self.out2: int = int(cfg["d_model_2"])
        self.return_layers: int | list[int] = cfg["return_layers"]
        self.use_norm: bool = bool(cfg["use_norm"])
        self.return_type: str = cfg.get("return_type", cfg.get("mode", "multi"))
        self.img_mean = torch.tensor(cfg["img_mean"]).view(1, 3, 1, 1)
        self.img_std = torch.tensor(cfg["img_std"]).view(1, 3, 1, 1)
        self.use_cnn_for_fine_features = cfg["use_cnn_for_fine_features"]
        self.use_fusion = cfg["use_fusion"]
        self.cnn_weights_path = self._resolve_weights_path(cfg["cnn_weights_path"])

        self.model = self._build_model()
        self.embed_dim = getattr(self.model, "embed_dim", 1024)
        self.proj16 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out16, 1, bias=False),
            nn.BatchNorm2d(self.out16),
        )
        # Projections used when the optional CNN branch is disabled.
        self.vit_proj8 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out8, 1, bias=False),
            nn.BatchNorm2d(self.out8),
        )
        self.vit_proj2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out2, 1, bias=False),
            nn.BatchNorm2d(self.out2),
        )

        # --- Conditional setup for fine feature extraction ---
        if self.use_cnn_for_fine_features:
            models = _torchvision_models()
            logger.info("Building hybrid backbone with CNN for fine features.")

            # --- (CNN loading logic, no changes) ---
            if cfg["cnn_model_name"] == "resnet18":
                cnn_out_channels_8x, cnn_out_channels_2x = 128, 64
                cnn_base = models.resnet18(weights=None)
                if self.cnn_weights_path and os.path.exists(self.cnn_weights_path):
                    logger.info(f"Loading local CNN weights from: {self.cnn_weights_path}")
                    state_dict = safe_torch_load(self.cnn_weights_path, map_location="cpu", weights_only=True)
                    cnn_base.load_state_dict(state_dict)
                else:
                    logger.warning("Local CNN weights not found. Attempting to download online.")
                    cnn_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                cnn_out_channels_8x, cnn_out_channels_2x = 128, 64
                cnn_base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

            self.cnn_conv1, self.cnn_bn1, self.cnn_relu = (
                cnn_base.conv1,
                cnn_base.bn1,
                cnn_base.relu,
            )
            self.cnn_maxpool, self.cnn_layer1, self.cnn_layer2 = (
                cnn_base.maxpool,
                cnn_base.layer1,
                cnn_base.layer2,
            )

            ### MODIFIED ###
            # Replaced the simple concatenation fusion with Dynamic Expert Gating.
            if self.use_fusion:
                logger.info("Dynamic expert gating fusion is enabled.")

                # 1. DINOv3 projectors to match CNN feature dimensions
                self.dino_proj8 = nn.Conv2d(self.embed_dim, cnn_out_channels_8x, kernel_size=1)
                self.dino_proj2 = nn.Conv2d(self.embed_dim, cnn_out_channels_2x, kernel_size=1)

                # 2. Gating modules to generate the adaptive weight 'alpha'
                gate_hidden_dim = cfg["gate_hidden_dim"]
                self.gate_generator8 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_8x + self.embed_dim, gate_hidden_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(gate_hidden_dim, 1, 1),
                    nn.Sigmoid(),
                )
                self.gate_generator2 = nn.Sequential(
                    nn.Conv2d(
                        cnn_out_channels_2x + self.embed_dim, gate_hidden_dim // 2, 3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(gate_hidden_dim // 2, 1, 1),
                    nn.Sigmoid(),
                )

                # 3. Final output projectors (now process the fused features)
                self.proj8 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_8x, self.out8, 1, bias=False),
                    nn.BatchNorm2d(self.out8),
                )
                self.proj2 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_2x, self.out2, 1, bias=False),
                    nn.BatchNorm2d(self.out2),
                )

            else:
                logger.info("Feature fusion is disabled.")
                # Original projection layers (only process CNN features)
                self.proj8 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_8x, self.out8, 1, bias=False),
                    nn.BatchNorm2d(self.out8),
                )
                self.proj2 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_2x, self.out2, 1, bias=False),
                    nn.BatchNorm2d(self.out2),
                )

            # Freeze CNN layers
            for p in cnn_base.parameters():
                p.requires_grad = False

        else:  # Original interpolation logic
            # --- (No changes here) ---
            logger.info("Building backbone with interpolation for fine features.")
            self.proj8 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out8, 1, bias=False), nn.BatchNorm2d(self.out8)
            )
            self.proj2 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out2, 1, bias=False), nn.BatchNorm2d(self.out2)
            )

        # Freeze DINOv3 model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # --- (Helper methods _resolve_source, etc. remain unchanged) ---
    def _resolve_weights_path(self, weights_path: str | None) -> str | None:
        """Resolve weights path to absolute path if it's relative."""
        if not weights_path:
            return None
        if os.path.isabs(weights_path):
            return weights_path
        # Try to find project root (where the weights/ symlink should be)
        current_file = os.path.abspath(__file__)
        project_root = current_file
        for _ in range(10):  # Search up to 10 levels
            project_root = os.path.dirname(project_root)
            weights_candidate = os.path.join(project_root, weights_path)
            if os.path.exists(weights_candidate):
                return weights_candidate
        # Fallback: return original relative path
        return weights_path

    def _resolve_source(self):
        if self.source in ("local", "github"):
            return self.source
        if self.repo_dir is not None and os.path.isdir(self.repo_dir):
            return "local"
        return "github"

    def _build_model(self):
        repo = self.github_repo
        src = self._resolve_source()
        if src == "local":
            repo = self.repo_dir
        # NOTE: Some hub implementations will still load weights if `weights=...` is passed,
        # even when `pretrained=False`.
        kwargs = {"pretrained": self.pretrained}
        if self.pretrained and self.weights_path:
            kwargs["weights"] = self.weights_path
        try:
            model = torch.hub.load(repo, self.model_name, source=src, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load hub model '{self.model_name}' from '{repo}' (source={src}): {e}"
            ) from e
        return model

    def _check_size(self, x: torch.Tensor):
        h, w = x.shape[-2], x.shape[-1]
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            pass

    def _normalize(self, x: torch.Tensor):
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        device = x.device
        mean = self.img_mean.to(device)
        std = self.img_std.to(device)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._normalize(x)

        if self.use_cnn_for_fine_features:
            # --- HYBRID FORWARD ---
            with torch.no_grad():
                # --- (Feature extraction, no changes) ---
                self._check_size(x)
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat16_raw = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list
                c1 = self.cnn_relu(self.cnn_bn1(self.cnn_conv1(x_in)))
                c4 = self.cnn_layer2(self.cnn_layer1(self.cnn_maxpool(c1)))

            # --- (out16 projection, no changes) ---
            out16 = self.proj16(feat16_raw)
            if self.return_type == "dino":
                return out16

            ### MODIFIED ###
            # Replaced the simple concatenation fusion with Dynamic Expert Gating.
            if self.use_fusion:
                # --- Dynamic Gating Fusion Logic ---
                # 1. Prepare DINOv3 features for gating and fusion
                vit_feat_for_8x = F.interpolate(
                    feat16_raw, size=c4.shape[-2:], mode="bilinear", align_corners=False
                )
                vit_feat_for_2x = F.interpolate(
                    feat16_raw, size=c1.shape[-2:], mode="bilinear", align_corners=False
                )

                # 2. Generate the dynamic weight 'alpha' for each scale
                gate_input_8x = torch.cat([c4.detach(), vit_feat_for_8x], dim=1)
                alpha_8x = self.gate_generator8(gate_input_8x)

                gate_input_2x = torch.cat([c1.detach(), vit_feat_for_2x], dim=1)
                alpha_2x = self.gate_generator2(gate_input_2x)

                # 3. Project DINOv3 features to match CNN dimensions for the weighted sum
                vit_feat_8x_proj = self.dino_proj8(vit_feat_for_8x)
                vit_feat_2x_proj = self.dino_proj2(vit_feat_for_2x)

                # 4. Perform the adaptive weighted sum
                fused_8x = (alpha_8x * c4) + ((1 - alpha_8x) * vit_feat_8x_proj)
                fused_2x = (alpha_2x * c1) + ((1 - alpha_2x) * vit_feat_2x_proj)

                # 5. Project the fused features to their final output dimensions
                out8 = self.proj8(fused_8x)
                out2 = self.proj2(fused_2x)
            else:
                # --- (Non-Fusion Logic, no changes) ---
                out8 = self.proj8(c4)
                out2 = self.proj2(c1)

            return out16, out8, out2

        else:
            # --- (ORIGINAL INTERPOLATION FORWARD, no changes) ---
            self._check_size(x)
            with torch.no_grad():
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list
            B, C, Hp, Wp = feat.shape
            Hin, Win = x.shape[-2:]
            H16, W16, H8, W8, H2, W2 = Hin // 16, Win // 16, Hin // 8, Win // 8, Hin // 2, Win // 2
            feat_16 = (
                F.interpolate(feat, (H16, W16), mode="bilinear", align_corners=True)
                if (Hp, Wp) != (H16, W16)
                else feat
            )
            out16 = self.proj16(feat_16)
            if self.return_type == "dino":
                return out16
            feat_8 = F.interpolate(feat, (H8, W8), mode="bilinear", align_corners=True)
            feat_2 = F.interpolate(feat, (H2, W2), mode="bilinear", align_corners=True)
            out8, out2 = self.vit_proj8(feat_8), self.vit_proj2(feat_2)
            return out16, out8, out2


class DINOv3_backbone_vitconvnext(nn.Module):
    """
    Hybrid Backbone with optional Feature Fusion.
    - DINOv3 (ViT) branch provides 16x semantic features.
    ### MODIFICATION ###
    - ConvNeXt-Tiny branch provides 8x and 4x features, with 2x features derived via upsampling.
    - If `use_fusion` is True, the 16x ViT feature is upsampled and fused with
      the CNN features at the 8x and 2x levels.
    """

    _default_config = {
        "repo_dir": "retracker/models/backbone/dinov3",
        "github_repo": "facebookresearch/dinov3",
        "model_name": "dinov3_vitl16",
        "weights_path": "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "cnn_weights_path": "weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "source": "auto",
        "download_online": True,
        "patch_size": 16,
        "d_model_16": 384,
        "d_model_8": 256,
        "d_model_2": 128,
        "return_layers": 1,
        "use_norm": True,
        "return_type": "multi",
        "img_mean": (0.485, 0.456, 0.406),
        "img_std": (0.229, 0.224, 0.225),
        "use_cnn_for_fine_features": True,
        ### MODIFICATION ###
        "cnn_model_name": "convnext_tiny",  # Changed from 'convnext_large'
        "use_fusion": True,
    }

    def __init__(self, config: dict):
        super().__init__()
        cfg = {**self._default_config, **(config or {})}

        # --- (ViT branch config parsing remains the same) ---
        self.repo_dir: str | None = cfg["repo_dir"] or os.getenv("DINOV3_LOCATION", None)
        self.github_repo: str = cfg["github_repo"]
        self.model_name: str = cfg["model_name"]
        self.weights_path: str | None = cfg["weights_path"]
        # When False, build the backbone architecture without loading hub weights.
        # Useful for fast-start inference when a full ReTracker checkpoint is loaded next.
        self.pretrained: bool = bool(cfg.get("pretrained", True))
        self.source: str = cfg["source"]
        self.download_online: bool = cfg["download_online"]
        self.patch_size: int = int(cfg["patch_size"])
        self.out16: int = int(cfg["d_model_16"])
        self.out8: int = int(cfg["d_model_8"])
        self.out2: int = int(cfg["d_model_2"])
        self.return_layers: int | list[int] = cfg["return_layers"]
        self.use_norm: bool = bool(cfg["use_norm"])
        self.return_type: str = cfg.get("return_type", cfg.get("mode", "multi"))
        self.img_mean = torch.tensor(cfg["img_mean"]).view(1, 3, 1, 1)
        self.img_std = torch.tensor(cfg["img_std"]).view(1, 3, 1, 1)
        self.use_cnn_for_fine_features = cfg["use_cnn_for_fine_features"]
        self.use_fusion = cfg["use_fusion"]
        self.cnn_weights_path = self._resolve_weights_path(cfg["cnn_weights_path"])
        self.cnn_model_name = cfg["cnn_model_name"]
        self.cnn_pretrained: bool = bool(cfg.get("cnn_pretrained", True))

        self.model = self._build_model()
        self.embed_dim = getattr(self.model, "embed_dim", 1024)
        self.proj16 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out16, 1, bias=False),
            nn.BatchNorm2d(self.out16),
        )
        # Projections used when the optional CNN branch is disabled.
        self.vit_proj8 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out8, 1, bias=False),
            nn.BatchNorm2d(self.out8),
        )
        self.vit_proj2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out2, 1, bias=False),
            nn.BatchNorm2d(self.out2),
        )

        if self.use_cnn_for_fine_features:
            logger.info("Building hybrid backbone with ConvNeXt-Tiny for fine features.")

            ### MODIFICATION ###
            if cfg["cnn_model_name"] == "convnext_tiny":
                logger.info(
                    "Building ConvNeXt-Tiny backbone for fine features "
                    f"(pretrained={self.cnn_pretrained})."
                )
                # Channel numbers specific to ConvNeXt-Tiny
                cnn_out_channels_8x = 192
                cnn_out_channels_4x = 96

                # try:
                # Load the DINOv3 pretrained ConvNeXt-Tiny model
                cnn_base = self._build_cnn_model()

                # except Exception as e:
                #     raise RuntimeError(f"Failed to load hub model 'dinov3_convnextt' from '{self.github_repo}': {e}")
            else:
                raise ValueError(
                    f"CNN model '{cfg['cnn_model_name']}' is not supported. Please use 'convnext_tiny'."
                )

            # Extract necessary layers from the ConvNeXt architecture
            self.cnn_stage1 = nn.Sequential(
                cnn_base.downsample_layers[0], cnn_base.stages[0]
            )  # 4x features
            self.cnn_stage2 = nn.Sequential(
                cnn_base.downsample_layers[1], cnn_base.stages[1]
            )  # 8x features

            if self.use_fusion:
                logger.info("Feature fusion between ViT and ConvNeXt is enabled.")
                # Fusion layer for 8x features, updated with Tiny's channel count
                self.fusion_conv8 = nn.Sequential(
                    nn.Conv2d(
                        cnn_out_channels_8x + self.embed_dim,
                        self.out8,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out8),
                    nn.ReLU(inplace=True),
                )
                # Fusion layer for 2x features, updated with Tiny's 4x channel count (as it's upsampled)
                self.fusion_conv2 = nn.Sequential(
                    nn.Conv2d(
                        cnn_out_channels_4x + self.embed_dim,
                        self.out2,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out2),
                    nn.ReLU(inplace=True),
                )
            else:
                logger.info("Feature fusion is disabled.")
                # Projection layers updated with Tiny's channel counts
                self.proj8 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_8x, self.out8, 1, bias=False),
                    nn.BatchNorm2d(self.out8),
                )
                self.proj2 = nn.Sequential(
                    nn.Conv2d(cnn_out_channels_4x, self.out2, 1, bias=False),
                    nn.BatchNorm2d(self.out2),
                )

            # Freeze CNN layers
            for p in cnn_base.parameters():
                p.requires_grad = False

        else:  # Original interpolation logic (if not using CNN branch)
            logger.info("Building backbone with interpolation for fine features.")
            self.proj8 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out8, 1, bias=False), nn.BatchNorm2d(self.out8)
            )
            self.proj2 = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.out2, 1, bias=False), nn.BatchNorm2d(self.out2)
            )

        # Freeze DINOv3 ViT model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _resolve_weights_path(self, weights_path: str | None) -> str | None:
        """Resolve weights path to absolute path if it's relative."""
        if not weights_path:
            return None
        if os.path.isabs(weights_path):
            return weights_path
        # Try to find project root (where the weights/ symlink should be)
        current_file = os.path.abspath(__file__)
        project_root = current_file
        for _ in range(10):  # Search up to 10 levels
            project_root = os.path.dirname(project_root)
            weights_candidate = os.path.join(project_root, weights_path)
            if os.path.exists(weights_candidate):
                return weights_candidate
        # Fallback: return original relative path
        return weights_path

    def _resolve_source(self):  # ... (no change) ...
        if self.source in ("local", "github"):
            return self.source
        if self.repo_dir is not None and os.path.isdir(self.repo_dir):
            return "local"
        return "github"

    def _build_model(self):  # ... (no change) ...
        repo = self.github_repo
        src = self._resolve_source()
        if src == "local":
            repo = self.repo_dir
        # NOTE: Some hub implementations will still load weights if `weights=...` is passed,
        # even when `pretrained=False`.
        kwargs = {"pretrained": self.pretrained}
        if self.pretrained and self.weights_path:
            kwargs["weights"] = self.weights_path
        try:
            model = torch.hub.load(repo, self.model_name, source=src, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load hub model '{self.model_name}' from '{repo}' (source={src}): {e}"
            ) from e
        return model

    def _build_cnn_model(self):
        repo = self.github_repo
        src = self._resolve_source()
        if src == "local":
            repo = self.repo_dir
        kwargs = {"pretrained": self.cnn_pretrained}
        if self.cnn_pretrained and self.cnn_weights_path:
            kwargs["weights"] = self.cnn_weights_path
        model = torch.hub.load(repo, "dinov3_" + self.cnn_model_name, source=src, **kwargs)
        return model

    def _check_size(self, x: torch.Tensor):  # ... (no change) ...
        h, w = x.shape[-2], x.shape[-1]
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            pass

    def _normalize(self, x: torch.Tensor):  # ... (no change) ...
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        device = x.device
        mean = self.img_mean.to(device)
        std = self.img_std.to(device)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._normalize(x)
        H, W = x_in.shape[-2:]

        if self.use_cnn_for_fine_features:
            with torch.no_grad():
                # 1. Get raw 16x ViT feature
                self._check_size(x)
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat16_raw = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list

                # 2. Get raw ConvNeXt features
                cnn_feat_4x = self.cnn_stage1(x_in)
                cnn_feat_8x = self.cnn_stage2(cnn_feat_4x)

            # 3. Project 16x feature
            out16 = self.proj16(feat16_raw)

            if self.return_type == "dino":
                return out16

            if self.use_fusion:
                # --- Fusion Logic ---
                # Fuse for 8x level
                vit_feat_for_8x = F.interpolate(
                    feat16_raw, size=cnn_feat_8x.shape[-2:], mode="bilinear", align_corners=False
                )
                fused_8x = torch.cat([cnn_feat_8x, vit_feat_for_8x], dim=1)
                out8 = self.fusion_conv8(fused_8x)

                # Fuse for 2x level
                target_size_2x = (H // 2, W // 2)
                cnn_feat_2x_upsampled = F.interpolate(
                    cnn_feat_4x, size=target_size_2x, mode="bilinear", align_corners=False
                )
                vit_feat_for_2x = F.interpolate(
                    feat16_raw, size=target_size_2x, mode="bilinear", align_corners=False
                )
                fused_2x = torch.cat([cnn_feat_2x_upsampled, vit_feat_for_2x], dim=1)
                out2 = self.fusion_conv2(fused_2x)
            else:
                # --- Non-Fusion Logic (original hybrid) ---
                out8 = self.proj8(cnn_feat_8x)

                target_size_2x = (H // 2, W // 2)
                cnn_feat_2x_upsampled = F.interpolate(
                    cnn_feat_4x, size=target_size_2x, mode="bilinear", align_corners=False
                )
                out2 = self.proj2(cnn_feat_2x_upsampled)

            return out16, out8, out2

        else:
            # --- Original interpolation forward logic ---
            self._check_size(x)
            with torch.no_grad():
                feats_list = self.model.get_intermediate_layers(
                    x_in, n=self.return_layers, reshape=True, norm=self.use_norm
                )
                feat = feats_list[-1] if isinstance(feats_list, (list, tuple)) else feats_list
            H16, W16, H8, W8, H2, W2 = H // 16, W // 16, H // 8, W // 8, H // 2, W // 2
            feat_16 = F.interpolate(feat, (H16, W16), mode="bilinear", align_corners=False)
            out16 = self.proj16(feat_16)
            if self.return_type == "dino":
                return out16
            feat_8 = F.interpolate(feat, (H8, W8), mode="bilinear", align_corners=False)
            feat_2 = F.interpolate(feat, (H2, W2), mode="bilinear", align_corners=False)
            out8, out2 = self.vit_proj8(feat_8), self.vit_proj2(feat_2)
            return out16, out8, out2

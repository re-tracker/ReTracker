import os

from torch import nn

from .hub.backbone import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg


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
        self.dino_arch = config["arch"]
        assert self.dino_arch in model_dict.keys(), "undefined pretrained dinov2 model"

        self.emb_dim = model_dict[self.dino_arch]["embedding_dim"]
        self.hid_dim = config["hid_dim"]
        self.out_dim = config["d_model"]

        # from emb_dim to out_dim directly
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.emb_dim, self.out_dim, 1, 1), nn.BatchNorm2d(self.out_dim)
                )
                for i in range(4)
            ]
        )

        self.dino_w_adaptor = self.build_dino_with_vitadaptor(config)

    def forward(self, x):
        """
        Args:
            x(torch.Tensor): [B, C, H, W], C=3
        """
        # resize input to produce same output
        _, _C, _H, _W = x.shape
        if _C == 1:  # tmp solve for gray images
            x = x.repeat(1, 3, 1, 1)
        B, C, H, W = x.shape
        out = self.dino_w_adaptor(x)  # 128, 64, 32, 16
        # out = map(self.projs,
        # feat_4x = self.projs[0](out[0])
        feat_4x, feat_8x, feat_16x, feat_32x = (
            self.projs[0](out[0]),
            self.projs[1](out[1]),
            self.projs[2](out[2]),
            self.projs[3](out[3]),
        )
        # feat_2x = F.interpolate(feat_4x, scale_factor=2., mode='bilinear', align_corners=True)
        return feat_4x, feat_8x, feat_16x, feat_32x

    def build_dino_with_vitadaptor(self, config):
        # Import ViTAdapter lazily so inference-only users don't pay the import cost
        # (or hit missing `mmcv`/`timm`) unless they explicitly enable the adaptor.
        try:
            from retracker.models.backbone.vit_adaptor.vit_adaptor import ViTAdapter
        except ImportError as exc:
            raise ImportError(
                "ViTAdapter is not installed, but the adaptor backbone was requested. "
                "Install the optional dependencies (mmcv/timm) or set `dino.use_adaptor=false`."
            ) from exc

        PROJECT_DIR = str(os.getenv("PROJECT_DIR"))
        if PROJECT_DIR == "None":
            PROJECT_DIR = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        dino_weights_dir = (
            os.path.join(PROJECT_DIR, "weights")
            if config["dino_weights_path"] is None
            else config["dino_weights_path"]
        )
        dino_weights_path = os.path.join(dino_weights_dir, self.dino_arch + "_pretrain.pth")
        vit_kwargs = {}
        vit_arch_name = ""
        if "vitl" in config["arch"]:
            vit_kwargs = {
                "img_size": 518,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "patch_size": 14,
                "mlp_ratio": 4,
                "init_values": 1.0,
                "ffn_layer": "mlp",
                "block_chunks": 0,
                "num_register_tokens": 4,
                "interpolate_antialias": True,
                "interpolate_offset": 0.0,
            }
            vit_arch_name = "vit_large"
        elif "vits" in config["arch"]:
            vit_kwargs = {
                "img_size": 518,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "patch_size": 14,
                "mlp_ratio": 4,
                "init_values": 1.0,
                "ffn_layer": "mlp",
                "block_chunks": 0,
                "num_register_tokens": 4,
                "interpolate_antialias": True,
                "interpolate_offset": 0.0,
            }
            vit_arch_name = "vit_small"
        else:
            pass

        model = ViTAdapter(
            pretrain_size=512,
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
            # interaction_indexes=[[0, 5],[12, 17]],
            deform_num_heads=4,
            conv_inplane=8,
            # deform_num_heads=16,
            deform_ratio=0.5,
            vit_arch_name=vit_arch_name,
            # pretrained dinov2_reg4 checkpoint settings
            vit_kwargs=vit_kwargs,
            pretrained=True,
            dino_weights_path=dino_weights_path,
            with_cp=False,
        )

        return model
        # state_dict = torch.load(dino_weights_path)
        # self.dino.load_sta    te_dict(state_dict)

# Copyright (c) Shanghai AI Lab. All rights reserved.
import itertools
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from retracker.utils.checkpoint import safe_torch_load

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from ..dinov2 import DinoVisionTransformer
from .adapter_modules import InteractionBlock, SpatialPriorModule, deform_inputs


_logger = logging.getLogger(__name__)

__all__ = ["ViTAdapter"]


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


class ViTAdapter(DinoVisionTransformer):
    def __init__(
        self,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
        drop_path_rate=0.4,
        vit_arch_name="vit_base",
        vit_kwargs=None,
        pretrained=True,
        download_online=False,
        dino_weights_path="",
    ):
        if vit_kwargs is None:
            vit_kwargs = {}
        super().__init__(**vit_kwargs)

        patch_size = vit_kwargs.get("patch_size")
        if patch_size is None:
            raise ValueError("`vit_kwargs` must contain `patch_size`.")
        num_register_tokens = int(vit_kwargs.get("num_register_tokens", 0))
        model_name = _make_dinov2_model_name(vit_arch_name, patch_size, num_register_tokens)
        if pretrained:
            if download_online:
                url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            else:
                path = dino_weights_path
                state_dict = safe_torch_load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=True)

        # Freeze vit
        for param in self.parameters():
            param.requires_grad = False

        # self.num_classes = 80
        self.drop_path_rate = 0.4
        self.mask_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=True)
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    #  norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, 518 // 14, 518 // 14, -1).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)  # here 18x18

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # here 16 32 64 ...
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        _, _, h, w = x.shape

        h_vit, w_vit = h // 16 * 14, w // 16 * 14
        # Patch Embedding forward
        x = F.interpolate(x, (h_vit, w_vit), mode="bilinear", align_corners=False)
        x = self.patch_embed(x)
        H, W = h_vit // self.patch_size, w_vit // self.patch_size
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = x + pos_embed

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            c1, c2, c3 = c1 + x1, c2 + x2, c3 + x3

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def _forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)  # here 18x18

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # here 16 32 64 ...
        # c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        # c = torch.cat([c2, c3, c4], dim=1)
        c2 = c2 + self.level_embed[0]
        c = torch.cat([c2], dim=1)

        # Patch Embedding forward
        _, _, h, w = x.shape

        h_vit, w_vit = h // 16 * 14, w // 16 * 14
        # Patch Embedding forward
        x = F.interpolate(x, (h_vit, w_vit), mode="bilinear", align_corners=False)
        x = self.patch_embed(x)
        H, W = h_vit // self.patch_size, w_vit // self.patch_size
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = x + pos_embed

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        # c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        # c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        # c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        # c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
            # x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            # x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            # c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            # c1, c2, c3= c1 + x1, c2 + x2, c3 + x3
            c1 = c1 + x1

        # Final Norm
        f1 = self.norm1(c1)
        # f2 = self.norm2(c2)
        # f3 = self.norm3(c3)
        # f4 = self.norm4(c4)
        # return [f1, f2, f3, f4]
        return [f1]

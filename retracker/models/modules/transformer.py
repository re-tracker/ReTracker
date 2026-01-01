"""Core Transformer components for feature matching and attention.

This module contains the essential transformer building blocks used throughout
the ReTracker model, including the LoFTR-style encoder layers and feature
transformers for temporal and spatial attention.

For ROMA decoder components, see roma_decoder.py.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from retracker.models.backbone.layers.attention import MemEffAttention
from retracker.models.backbone.layers.block import Block

from .linear_attention import FullAttention, LinearAttention, RoPELinearAttention, XAttention


if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False


class LoFTREncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        attention="linear",
    ):
        super().__init__()
        xformer = FLASH_AVAILABLE

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "rope_linear":
            self.attention = RoPELinearAttention()
        elif xformer:
            self.attention = XAttention()
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        source,
        x_mask=None,
        source_mask=None,
        QK_preprocessor=None,
        QK_encoder=None,
        QK_postprocessor=None,
        size_info=None,
    ):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        query, key = (
            QK_preprocessor(query, key, size_info) if QK_preprocessor is not None else (query, key)
        )
        query, key = (map(QK_encoder, [query, key])) if QK_encoder is not None else (query, key)
        query, key = (
            QK_postprocessor(query, key, size_info)
            if QK_postprocessor is not None
            else (query, key)
        )
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(message.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        encoder_layer = LoFTREncoderLayer(config["d_model"], config["nhead"], config["attention"])
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), (
            "the feature number of src and transformer must be equal"
        )

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class Mem_Queries_FeatureTransformer(nn.Module):
    """Temporal Feature Transformer"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_num = config["layer_num"]
        encoder_layer = LoFTREncoderLayer(config["d_model"], config["nhead"], config["attention"])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.layer_num)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, QK_encoder=None):
        """
        Args:
            original feat0 (torch.Tensor): [B, N, C]
            original feat1 (torch.Tensor): [B, N, K, C]
            ======== CHANGE =======
            feat0 (torch.Tensor): [B*N, 1, C]
            feat1 (torch.Tensor): [B*N, K, C]

            mask0 (torch.Tensor): [B*N, 1] (optional)
            mask1 (torch.Tensor): [B*N, K] (optional)
        """
        B, N, C = feat0.shape
        feat0 = rearrange(feat0, "B N C -> (B N) 1 C")
        feat1 = rearrange(feat1, "B N K C -> (B N) K C")

        if mask0 is not None:
            mask0 = rearrange(mask0, "B N 1 -> (B N) 1")
        if mask1 is not None:
            mask1 = rearrange(mask1, "B N K -> (B N) K")

        for layer in self.layers:
            feat0 = layer(feat0, feat1, mask0, mask1, QK_encoder=QK_encoder)

        feat0 = rearrange(feat0, "(B N) 1 C -> B N C", N=N)
        feat1 = rearrange(feat1, "(B N) K C -> B N K C", N=N)
        return feat0


class TransformerDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        Block(config["d_model"], config["nhead"], attn_class=MemEffAttention)
                        for _ in range(config["layer_num"])
                    ]
                )
            ]
        )

    def forward(self, encoder_out, tokens):
        """
        Input:
            feat_d0: [n, hw, c]
            feat_d1: [n, hw, c]
            learnable tokens:
        output:
          cls_onehot: n, hw, cls
          conf: n, hw, 1
        """
        z = torch.cat([tokens, encoder_out], dim=2)
        for block in self.blocks:
            z = block(z)
        return z

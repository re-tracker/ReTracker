"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import warnings

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Module


if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False
    warnings.warn("flash attention is not available", stacklevel=2)
from einops.einops import rearrange

from ..utils.position_encoding import RoPEPositionEncodingSine
from .torch_compat import custom_fwd_compatible


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            # QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))
            values = values * kv_mask[:, :, None, None]

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class RoPELinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.RoPE = RoPEPositionEncodingSine(256, max_shape=(256, 256))

    @custom_fwd_compatible
    def forward(self, queries, keys, values, q_mask=None, kv_mask=None, H=None, W=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        nhead, d = Q.size(2), Q.size(3)
        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow

        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        Q = rearrange(Q, "n (h w) nhead d -> n h w (nhead d)", h=H, w=W)
        K = rearrange(K, "n (h w) nhead d -> n h w (nhead d)", h=H, w=W)
        Q, K = self.RoPE(Q), self.RoPE(K)
        Q = rearrange(Q, "n h w (nhead d) -> n (h w) nhead d", nhead=nhead, d=d)
        K = rearrange(K, "n h w (nhead d) -> n (h w) nhead d", nhead=nhead, d=d)
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        del K, values
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class XAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # NOTE: q_mask/kv_mask are currently unused in this implementation.
        if FLASH_AVAILABLE:
            args = [x.contiguous() for x in [queries, keys, values]]
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                out = F.scaled_dot_product_attention(*args)
        else:
            # if flash_attn_func_ok:
            #     out = flash_attn_func(queries, keys, values)
            # else:
            QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)

            # Compute the attention and the weighted average
            softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, values)

        return out

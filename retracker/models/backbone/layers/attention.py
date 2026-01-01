# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# modified version; add PE attention;

import logging
import math
import os
import warnings
from typing import cast

import torch
from torch import Tensor, nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        warnings.warn("xFormers is disabled (Attention)", stacklevel=2)
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)", stacklevel=2)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, kv=None, cache=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q = qkv[0]

        if kv is not None:
            # Cross-Attention
            k, v = kv
        elif cache is not None:
            # Self-Attention with KV Cache
            k_cache, v_cache = cache
            k = torch.cat([k_cache, qkv[1]], dim=2)
            v = torch.cat([v_cache, qkv[2]], dim=2)
            cache = (k, v)
        else:
            # Standard Self-Attention
            k, v = qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, cache


class Attention_with_QK_RoPE(nn.Module):
    """Basic attention block with RoPE"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RotaryPositionEmbedding(head_dim=dim // num_heads, is_eff_attention_order=False)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        q = self.rope(q, seq_len=N)
        k = self.rope(k, seq_len=N)

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # xFormers' memory_efficient_attention is CUDA-only in many builds. If we're on
        # CPU (or xFormers isn't available), fall back to the vanilla attention.
        if (not XFORMERS_AVAILABLE) or (x.device.type != "cuda"):
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        try:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        except NotImplementedError:
            # Some xFormers builds don't support certain head dims/dtypes on a given device.
            # Prefer correctness over performance by falling back when possible.
            if attn_bias is not None:
                raise
            return super().forward(x)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 48,
        base: float = 10000.0,
        is_eff_attention_order=True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_eff_attention_order = is_eff_attention_order

        # Precompute inverse frequencies for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Apply rotary position embeddings to the input tensor.
        Args:
            x: Tensor with shape
                (batch_size, seq_len, num_heads, head_dim) if use efficient attention;
                (batch_size, num_heads, seq_len, head_dim) if use basic attention;
            seq_len: Sequence length
        """
        inv_freq = cast(torch.Tensor, self.inv_freq)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # Shape: (seq_len, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # Shape: (seq_len, head_dim)

        if self.is_eff_attention_order:
            cos = emb.cos().unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim)
            sin = emb.sin().unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim)
        else:
            cos = emb.cos().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated_x * sin)


class MemEffAttention_with_QK_RoPE(Attention_with_QK_RoPE):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.rope = RotaryPositionEmbedding(head_dim=dim // num_heads, is_eff_attention_order=True)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        use_xformers = XFORMERS_AVAILABLE and (x.device.type == "cuda")
        if attn_bias is not None and not use_xformers:
            raise AssertionError("xFormers is required for using nested tensors")

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)  # B N heads C

        q = self.rope(q, seq_len=N)
        k = self.rope(k, seq_len=N)

        if use_xformers:
            try:
                x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            except NotImplementedError:
                # Some xFormers builds don't support certain head dims/dtypes; fall back.
                use_xformers = False

        if not use_xformers:
            # Vanilla scaled dot-product attention (CPU-safe).
            q_t = (q * self.scale).permute(0, 2, 1, 3)  # (B, H, N, D)
            k_t = k.permute(0, 2, 1, 3)  # (B, H, N, D)
            v_t = v.permute(0, 2, 1, 3)  # (B, H, N, D)
            attn = q_t @ k_t.transpose(-2, -1)  # (B, H, N, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v_t).transpose(1, 2)  # (B, N, H, D)

        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiT3RotaryPositionEmbedding(RotaryPositionEmbedding):
    """
    Rotary Embedding with DiT3-style dynamic scaling so it can extrapolate
    beyond the maximum sequence length seen during training.
    """

    def __init__(
        self,
        head_dim: int,
        max_train_seq_len: int = 512,
        base: float = 10000.0,
        scaling: str = "logn",
        is_eff_attention_order: bool = True,
    ):
        super().__init__(
            head_dim=head_dim,
            max_position_embeddings=max_train_seq_len,
            base=base,
            is_eff_attention_order=is_eff_attention_order,
        )
        self.max_train_seq_len = max_train_seq_len
        self.scaling = scaling
        base_inv_freq = cast(torch.Tensor, self.inv_freq).clone()
        self.register_buffer("base_inv_freq", base_inv_freq, persistent=False)

    def _scaled_inv_freq(self, seq_len: int) -> torch.Tensor:
        base_inv_freq = cast(torch.Tensor, self.base_inv_freq)
        if seq_len <= self.max_train_seq_len:
            return base_inv_freq

        scale = seq_len / float(self.max_train_seq_len)

        if self.scaling == "linear":
            factor = 1.0 / scale
        elif self.scaling == "ntk":
            factor = self.max_train_seq_len / float(seq_len)
        elif self.scaling == "logn":
            # Taken from recent autoregressive DiT variants to alleviate extrap gap
            factor = math.log(scale * (math.e - 1) + 1.0) / scale
        else:
            raise ValueError(f"Unsupported DiT3 RoPE scaling mode: {self.scaling}")

        return base_inv_freq * factor

    def forward(self, x: torch.Tensor, seq_len: int):
        inv_freq = self._scaled_inv_freq(seq_len)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        if self.is_eff_attention_order:
            cos = emb.cos().unsqueeze(0).unsqueeze(2)
            sin = emb.sin().unsqueeze(0).unsqueeze(2)
        else:
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated_x * sin)


class DiT3RotaryPositionEmbedding2D(DiT3RotaryPositionEmbedding):
    """
    2D Axial RoPE that handles extra tokens (CLS, Context, Memory).
    It applies 2D rotation ONLY to the spatial square grid and leaves extra tokens untouched.
    """

    def __init__(
        self,
        head_dim: int,
        max_train_seq_len: int = 512,  # 这里应该传入 W*W (图像部分的长度)
        base: float = 10000.0,
        scaling: str = "logn",
        is_eff_attention_order: bool = True,
        extra_token_pos: str = "none",  # 选项: 'none', 'first', 'last'
    ):
        super().__init__(
            head_dim=head_dim,
            max_train_seq_len=max_train_seq_len,
            base=base,
            scaling=scaling,
            is_eff_attention_order=is_eff_attention_order,
        )
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        self.spatial_seq_len = max_train_seq_len  # 记录标准的图像长度 W*W
        self.extra_token_pos = extra_token_pos

    def forward(self, x: torch.Tensor, seq_len: int):
        # seq_len 是当前输入的总长度 (可能包含 extra tokens)
        # self.spatial_seq_len 是预设的图像部分长度 (W*W)

        # 1. 判断是否存在 extra tokens 以及如何分割
        is_pure_spatial = seq_len == self.spatial_seq_len

        if is_pure_spatial:
            x_spatial = x
            x_extra = None
        else:
            # 计算 extra tokens 的数量
            # 注意：这里假设推理时的图像分辨率和训练时一致，或者你需要动态计算最大的平方数
            # 为了简单起见，这里假设图像部分始终等于 self.spatial_seq_len (或者通过开方判断)

            # 更鲁棒的做法：尝试推断空间大小
            # 假设 extra token 数量很少，绝大多数是 spatial
            # 这里我们依据构造时的设定

            if self.extra_token_pos == "last":
                # Block 1 case: [Image, Memory]
                # 假设前部分是空间，最后剩余的是 extra
                # 如果支持变长图像，这里需要传入 H, W，但在 Transformer 中通常假定切分逻辑
                # 既然 seq_len > spatial_seq_len，且我们知道 W*W 是主要的

                # 简单策略：如果 seq_len 不是完全平方数，或者指定了有 extra
                split_idx = self.spatial_seq_len
                # 如果推理时图像变大，这里可能会有问题。
                # 更通用的做法是：side = int(sqrt(seq_len - num_extra))
                # 但为了不把问题搞得太复杂，我们假设 spatial_seq_len 就是当前图像的大小

                x_spatial = x[:, :split_idx, ...]
                x_extra = x[:, split_idx:, ...]

            elif self.extra_token_pos == "first":
                # Block 2 case: [Context, CLS, Image]
                # 图像在最后
                split_idx = seq_len - self.spatial_seq_len
                x_extra = x[:, :split_idx, ...]
                x_spatial = x[:, split_idx:, ...]

            else:
                # 'none' 但长度不匹配，尝试作为纯图像处理(可能会报错)
                x_spatial = x
                x_extra = None

        # 2. 对 x_spatial 生成 2D Grid 并旋转
        # 动态计算边长
        spatial_len = x_spatial.shape[1]
        side_len = int(math.sqrt(spatial_len))
        assert side_len * side_len == spatial_len, (
            f"Spatial part length {spatial_len} is not a square. Check extra_token_pos setting."
        )

        # --- 以下是原有的 2D RoPE 计算逻辑 (针对 x_spatial) ---
        max_train_side = int(math.sqrt(self.max_train_seq_len))
        base_inv_freq = cast(torch.Tensor, self.base_inv_freq)
        base_inv_freq_half = base_inv_freq[: self.head_dim // 4]

        if side_len <= max_train_side:
            inv_freq = base_inv_freq_half
        else:
            scale = side_len / float(max_train_side)
            # ... (scaling logic same as before) ...
            if self.scaling == "linear":
                factor = 1.0 / scale
            elif self.scaling == "logn":
                factor = math.log(scale * (math.e - 1) + 1.0) / scale
            else:
                factor = 1.0
            inv_freq = base_inv_freq_half * factor

        t = torch.arange(side_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb_y = (
            freqs.view(side_len, 1, -1)
            .expand(side_len, side_len, -1)
            .reshape(-1, self.head_dim // 4)
        )
        emb_x = (
            freqs.view(1, side_len, -1)
            .expand(side_len, side_len, -1)
            .reshape(-1, self.head_dim // 4)
        )
        emb = torch.cat([emb_y, emb_x], dim=-1)
        emb = torch.cat((emb, emb), dim=-1)  # [Spatial_N, dim]

        if self.is_eff_attention_order:
            cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, Spatial_N, 1, D]
            sin = emb.sin().unsqueeze(0).unsqueeze(2)
        else:
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)

        x1, x2 = x_spatial[..., : self.head_dim // 2], x_spatial[..., self.head_dim // 2 :]
        rotated_x2 = torch.cat((-x2, x1), dim=-1)
        x_spatial_out = (x_spatial * cos) + (rotated_x2 * sin)
        # ---------------------------------------------------

        # 3. 重新拼接
        if x_extra is None:
            return x_spatial_out

        if self.extra_token_pos == "last":
            return torch.cat([x_spatial_out, x_extra], dim=1)
        elif self.extra_token_pos == "first":
            return torch.cat([x_extra, x_spatial_out], dim=1)

        return x_spatial_out


class MemEffAttention_with_DiT3RoPE(MemEffAttention_with_QK_RoPE):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        max_train_seq_len: int = 512,
        scaling_strategy: str = "logn",
        base_theta: float = 10000.0,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.rope = DiT3RotaryPositionEmbedding(
            head_dim=dim // num_heads,
            max_train_seq_len=max_train_seq_len,
            base=base_theta,
            scaling=scaling_strategy,
            is_eff_attention_order=True,
        )

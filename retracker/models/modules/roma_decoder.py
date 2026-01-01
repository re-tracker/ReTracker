"""ROMA Transformer Decoder and related components.

This module contains the ROMATransformerDecoder3 and its supporting classes
for point tracking refinement.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    from timm.models.layers import DropPath
except ImportError:
    # Fallback if timm is not installed
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
            super().__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor.div_(keep_prob)
            return x * random_tensor


from retracker.models.backbone.layers import Mlp
from retracker.models.backbone.layers.attention import (
    DiT3RotaryPositionEmbedding2D,
    MemEffAttention,
    MemEffAttention_with_DiT3RoPE,
    MemEffAttention_with_QK_RoPE,
)
from retracker.models.backbone.layers.block import AttentionWithBias, Block, GroupBlock


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_seq_pos_cfg(seq_pos_cfg):
    if seq_pos_cfg is None:
        return {
            "type": "dit3_rope",
            "max_train_len": 64,
            "scaling": "logn",
            "base_theta": 10000.0,
        }
    if isinstance(seq_pos_cfg, str):
        return {"type": seq_pos_cfg}
    return seq_pos_cfg


def _build_seq_pos_attn_class(seq_pos_cfg):
    seq_pos_cfg = _normalize_seq_pos_cfg(seq_pos_cfg)
    seq_type = seq_pos_cfg.get("type", "none").lower()

    if seq_type == "none":
        return MemEffAttention

    if seq_type == "rope":
        return MemEffAttention_with_QK_RoPE

    max_train_len = int(seq_pos_cfg.get("max_train_len", 64))
    scaling = seq_pos_cfg.get("scaling", "logn")
    base_theta = float(seq_pos_cfg.get("base_theta", 10000.0))

    if seq_type == "dit3_rope":

        class _DiT3Attention(MemEffAttention_with_DiT3RoPE):
            def __init__(
                self,
                dim,
                num_heads,
                qkv_bias=False,
                proj_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
            ):
                super().__init__(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    max_train_seq_len=max_train_len,
                    scaling_strategy=scaling,
                    base_theta=base_theta,
                )

        return _DiT3Attention

    if seq_type == "2d_rope":
        extra_token_pos = seq_pos_cfg.get("extra_token_pos", "first")
        max_train_len_2d = int(seq_pos_cfg.get("max_train_len_2d", 7 * 7))

        class _DiT3Attention2D(MemEffAttention_with_DiT3RoPE):
            def __init__(
                self,
                dim,
                num_heads,
                qkv_bias=False,
                proj_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
            ):
                super().__init__(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    max_train_seq_len=max_train_len_2d,
                    scaling_strategy=scaling,
                    base_theta=base_theta,
                )

                head_dim = dim // num_heads

                self.rope = DiT3RotaryPositionEmbedding2D(
                    head_dim=head_dim,
                    max_train_seq_len=max_train_len_2d,
                    base=base_theta,
                    scaling=scaling,
                    extra_token_pos=extra_token_pos,
                )

        return _DiT3Attention2D

    raise ValueError(f"Unknown sequence position encoding type: {seq_type}")


# =============================================================================
# Position Encoding Classes
# =============================================================================


class FourierEmbedder(nn.Module):
    def __init__(self, input_dim=2, freq_num=10, max_freq=10.0, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.input_dim = input_dim

        scales = torch.pow(2.0, torch.linspace(0.0, max_freq, steps=freq_num))
        self.register_buffer("scales", scales)

        self.out_dim = input_dim * freq_num * 2
        if include_input:
            self.out_dim += input_dim

    def forward(self, x):
        # x: [..., input_dim]
        b_shape = x.shape[:-1]

        x_scaled = x.unsqueeze(-1) * self.scales * torch.pi
        x_scaled = x_scaled.reshape(*b_shape, -1)

        x_sin = torch.sin(x_scaled)
        x_cos = torch.cos(x_scaled)

        embed = torch.cat([x_sin, x_cos], dim=-1)
        if self.include_input:
            embed = torch.cat([x, embed], dim=-1)

        return embed


class LogFourierCPB(nn.Module):
    """Log-Space Continuous Position Bias + Fourier Encoding"""

    def __init__(self, in_dim=2, embed_dim=64, freq_num=4):
        super().__init__()

        self.fourier = FourierEmbedder(
            input_dim=in_dim, freq_num=freq_num, max_freq=4.0, include_input=True
        )

        fourier_dim = self.fourier.out_dim

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, p1, p2):
        # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
        diff = p1.unsqueeze(2) - p2.unsqueeze(1)

        log_diff = torch.sign(diff) * torch.log(1 + torch.abs(diff))

        fourier_feat = self.fourier(log_diff)

        return self.mlp(fourier_feat)


class PositionMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Attention / Aggregation Classes
# =============================================================================


class LightweightTemporalAttention(nn.Module):
    """
    Performs a lightweight, dynamic, content-driven weighted average over the
    temporal dimension (F). It replaces a simple .mean(dim=1) with a learned
    attention mechanism.

    Includes a `temperature` parameter to control the sharpness of the attention
    distribution, which acts as a regularizer to improve generalization and
    training stability.
    """

    def __init__(
        self,
        d_model: int,
        temporal_attn_hid_dim: int = 32,
        temperature: float = 1.0,
        is_learnable_temp: bool = False,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(d_model, temporal_attn_hid_dim),
            nn.GELU(),
            nn.Linear(temporal_attn_hid_dim, 1),
        )

        if is_learnable_temp:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

        self.is_learnable_temp = is_learnable_temp

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Input tensor of shape [BN, F, WW, C].

        Returns:
            torch.Tensor: Aggregated tensor of shape [BN, WW, C].
        """
        BN, F, WW, C = tokens.shape

        representative_tokens = tokens[:, :, WW // 2, :]

        scores = self.scorer(representative_tokens)

        if self.is_learnable_temp:
            temperature = torch.exp(self.log_temperature)
        else:
            temperature = self.temperature

        weights = torch.softmax(scores / temperature, dim=1)

        weights = weights.unsqueeze(-1)

        aggregated_tokens = torch.sum(weights * tokens, dim=1)

        return aggregated_tokens


class SelfGatedModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Self-Gated module forward pass
        :param x: Input tensor of shape BN x F x W x C
        :return: Output tensor of shape BN x F x W x C
        """
        W = x.shape[2]
        x = rearrange(x, "BN F W C -> (BN W) C F")
        gate_signal = self.sigmoid(self.gate(x))
        x = x * gate_signal
        x = rearrange(x, "(BN W) C F -> BN F W C", W=W)

        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self, dim, num_heads, attn_class, bottleneck_ratio=4, mlp_ratio=2.0, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.internal_dim = dim // bottleneck_ratio

        self.down_proj = nn.Linear(dim, self.internal_dim)
        self.norm_down = nn.LayerNorm(self.internal_dim)

        self.num_heads = max(1, num_heads // bottleneck_ratio)

        self.attn = attn_class(
            dim=self.internal_dim, num_heads=self.num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm_attn = nn.LayerNorm(self.internal_dim)

        hidden_dim = int(self.internal_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.internal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, self.internal_dim),
            nn.Dropout(drop),
        )
        self.norm_mlp = nn.LayerNorm(self.internal_dim)

        self.up_proj = nn.Linear(self.internal_dim, dim)

        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        shortcut = x

        x = self.down_proj(x)

        res = x
        x = self.norm_down(x)
        x = self.attn(x)
        x = res + x

        x = x + self.mlp(self.norm_mlp(x))

        x = self.up_proj(x)

        return shortcut + self.gamma * x


# =============================================================================
# Core Decoder Classes
# =============================================================================


class GeometricGroupRefinement(nn.Module):
    def __init__(self, dim, num_heads=8, depth=2, drop_path_rate=0.1, global_module_drop_rate=0.2):
        super().__init__()

        self.fourier_enc = FourierEmbedder(input_dim=2, freq_num=8)
        self.motion_mlp = nn.Sequential(
            nn.Linear(34, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.log_fourier_cpb = LogFourierCPB(in_dim=2, embed_dim=64)

        self.bias_generator = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, num_heads)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                GroupBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    init_values=1e-5,
                    drop_path=dpr[i],
                    attn_class=AttentionWithBias,
                )
                for i in range(depth)
            ]
        )

        self.gamma = nn.Parameter(torch.zeros(dim))

        self.confidence_gate = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim), nn.Sigmoid()
        )
        self.module_drop = (
            DropPath(global_module_drop_rate) if global_module_drop_rate > 0 else nn.Identity()
        )

        nn.init.constant_(self.confidence_gate[-2].bias, -5.0)

    def forward(self, visual_tokens, coords_init, coords_curr, coords_prev=None):
        B, N, C = visual_tokens.shape

        x = visual_tokens
        if coords_prev is not None:
            velocity = coords_curr - coords_prev
            vel_embed = self.fourier_enc(velocity)
            motion_feat = self.motion_mlp(vel_embed)
            x = x + motion_feat

        emb_query = self.log_fourier_cpb(coords_init, coords_init)
        emb_curr = self.log_fourier_cpb(coords_curr, coords_curr)

        edge_feat = torch.cat([emb_query, emb_curr], dim=-1)
        bias = self.bias_generator(edge_feat)

        attn_bias = bias.permute(0, 3, 1, 2).contiguous()

        for block in self.blocks:
            x = block(x, attn_bias=attn_bias)

        gate = self.confidence_gate(visual_tokens)
        correction = gate * x
        out = visual_tokens + self.module_drop(correction)
        return out


class HybridDenseDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.shared_kv = config.get("shared_kv", True)
        self.W = int(config["patch_size"] ** 0.5)

        self.heatmap_head = nn.Linear(self.d_model, 1)

        self.target_query = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.pos_proj = nn.Linear(2, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)

        if not self.shared_kv:
            self.value_proj = nn.Linear(self.d_model, self.d_model)
            nn.init.eye_(self.value_proj.weight)
            if self.value_proj.bias is not None:
                nn.init.zeros_(self.value_proj.bias)

        self.residual_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, 2)
        )

        self.global_refine_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, 2)
        )
        nn.init.zeros_(self.global_refine_head[-1].weight)
        nn.init.zeros_(self.global_refine_head[-1].bias)

        self.context_residual_head = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 2),
        )
        nn.init.zeros_(self.context_residual_head[-1].weight)
        nn.init.zeros_(self.context_residual_head[-1].bias)

        self.dense_bias_head = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 2),
        )
        nn.init.zeros_(self.dense_bias_head[-1].weight)
        nn.init.zeros_(self.dense_bias_head[-1].bias)

        self.aux_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, 2)
        )

        self.anchor_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.pos_proj_refine = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model)
        )

        self.query_refine_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=4, batch_first=True, dropout=0.1
        )
        self.state_norm = nn.LayerNorm(self.d_model)
        self.temperature = nn.Parameter(torch.ones(1) * 0.05)

        coords = torch.linspace(-(self.W // 2), self.W // 2, self.W)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        local_grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("local_grid", local_grid)

    def soft_argmax(self, heatmap_probs: torch.Tensor) -> torch.Tensor:
        flat_grid = self.local_grid.view(1, -1, 2)
        coords = (heatmap_probs * flat_grid).sum(dim=1)
        return coords

    def forward(self, final_tokens: torch.Tensor, prev_state: torch.Tensor = None):
        """
        final_tokens: [BN, WW, C] Current frame features
        prev_state:   [BN, C]     Previous frame output target feature (Optional)
        """
        BN, WW, C = final_tokens.shape
        flat_grid = self.local_grid.view(1, -1, 2)

        static_q = self.target_query.expand(BN, -1, -1)

        if prev_state is not None:
            memory = self.state_norm(prev_state).unsqueeze(1)

            delta_q, _ = self.query_refine_attn(query=static_q, key=memory, value=memory)
            dynamic_q = static_q + delta_q
        else:
            dynamic_q = static_q

        pos_emb_linear = self.pos_proj(flat_grid)

        pos_emb_refined = self.pos_proj_refine(pos_emb_linear).repeat(BN, 1, 1)
        visual_tokens_norm = self.norm(final_tokens)

        k_visual = self.key_proj(visual_tokens_norm)
        if not self.shared_kv:
            v_visual = self.value_proj(visual_tokens_norm)
        else:
            v_visual = visual_tokens_norm
        patch_keys = k_visual + pos_emb_refined

        anchor_key = self.anchor_token.expand(BN, 1, -1)
        all_keys = torch.cat([patch_keys, anchor_key], dim=1)

        attn_logits = torch.bmm(dynamic_q, all_keys.transpose(1, 2)).squeeze(1) / (
            self.d_model**0.5
        )
        if hasattr(self, "spatial_bias"):
            attn_logits[:, :WW] = attn_logits[:, :WW] + self.spatial_bias

        probs = F.softmax(attn_logits, dim=-1)

        patch_probs = probs[:, :WW].unsqueeze(-1)
        anchor_prob = probs[:, WW:].unsqueeze(-1)

        local_feat = v_visual + pos_emb_refined

        global_ctx = (patch_probs * v_visual).sum(dim=1, keepdim=True)
        residual_input = torch.cat([local_feat, global_ctx.expand(-1, WW, -1)], dim=-1)

        pixel_wise_votes = torch.tanh(self.context_residual_head(residual_input)) * float(self.W)

        global_offset_grid = (patch_probs * pixel_wise_votes).sum(dim=1)

        center_pred = global_offset_grid

        center_anchor = global_offset_grid.detach().unsqueeze(1)
        aux_input = residual_input.detach()
        local_bias = torch.tanh(self.dense_bias_head(aux_input)) * 2.0
        dense_flow_pred = center_anchor + local_bias
        dense_flow_pred[:, WW // 2] = center_pred

        current_state_visual = (patch_probs * final_tokens).sum(dim=1)
        current_state_anchor = anchor_prob.squeeze(1) * self.anchor_token
        next_state = current_state_visual + current_state_anchor

        aux_output = self.aux_head(next_state).reshape(-1, 2)

        is_occ_logit_dense = aux_output[:, 0:1].unsqueeze(1).repeat(1, WW, 1)
        conf_logit_dense = aux_output[:, 1:2].unsqueeze(1).repeat(1, WW, 1)

        final_output = torch.cat([dense_flow_pred, is_occ_logit_dense, conf_logit_dense], dim=-1)

        return final_output, next_state


class ROMATransformerDecoder3(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = config["d_model"]
        n_head = config["nhead"]
        seq_pos_cfg = config.get("seq_pos_encoding", "dit3_rope")
        self.seq_pos_encoding = _normalize_seq_pos_cfg(seq_pos_cfg)
        attn_class_with_rope = _build_seq_pos_attn_class(self.seq_pos_encoding)
        attn_class_with_2d_rope = _build_seq_pos_attn_class({"type": "2d_rope"})
        block_type = config.get("block_type", "bottleneck")

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.walking_memory_token = nn.Parameter(torch.randn(1, 1, d_model))

        if block_type == "bottleneck":
            self.blocks = nn.ModuleList(
                [
                    Block(d_model, n_head, attn_class=attn_class_with_2d_rope)
                    if idx % 2 == 0
                    else BottleneckBlock(
                        d_model, n_head, attn_class=attn_class_with_rope, bottleneck_ratio=4
                    )
                    for idx in range(config["layer_num"])
                ]
            )
            self.use_factorized = False
        else:
            self.blocks = nn.ModuleList(
                [
                    copy.deepcopy(
                        Block(
                            d_model,
                            n_head,
                            attn_class=attn_class_with_2d_rope
                            if idx % 2 == 0
                            else attn_class_with_rope,
                        )
                    )
                    for idx, _ in enumerate(range(config["layer_num"]))
                ]
            )
            self.use_factorized = False
        self.blocks2 = nn.ModuleList(
            [
                copy.deepcopy(Block(d_model, n_head, attn_class=attn_class_with_2d_rope))
                for _ in range(2)
            ]
        )

        self.gated = SelfGatedModule(d_model)
        self.temporal_aggregator = LightweightTemporalAttention(
            d_model=config["d_model"],
            temporal_attn_hid_dim=32,
            temperature=5.0,
            is_learnable_temp=False,
        )

        self.linear = nn.Linear(config["d_model"] * config["patch_size"], config["d_model"])
        self.mlp = Mlp(
            in_features=config["d_model"], hidden_features=config["d_model"], out_features=4
        )

        self.patch_head = HybridDenseDecoder(config)
        self.memory_adapter = nn.Linear(d_model, d_model)
        self.pos_mlp = PositionMLP(in_dim=2, out_dim=d_model)

        self.context_fusion = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Dropout(0.1)
        )

        self.group_refinement = GeometricGroupRefinement(
            dim=d_model,
            num_heads=n_head,
            depth=config.get("group_refine_layers", 2),
            drop_path_rate=0.1,
        )

        self.memory_channel_dropout_rate = config.get("memory_channel_dropout_rate", 0.1)
        self.memory_path_dropout_rate = config.get("memory_path_dropout_rate", 0.1)

    def forward(
        self,
        tokens: torch.Tensor,
        coords_init: torch.Tensor = None,
        coords_curr: torch.Tensor = None,
        coords_prev: torch.Tensor = None,
        context_tokens: list = None,
        last_walking_context: torch.Tensor = None,
        use_walking_memory: bool = True,
        walking_memory_drop_out_rate: float = 0.5,
    ):
        BN, F, WW, C = tokens.shape

        init_memory = self.walking_memory_token.unsqueeze(0).expand(BN, F, -1, -1)

        if use_walking_memory:
            if last_walking_context is None:
                current_memory = init_memory
            else:
                history_memory = last_walking_context.unsqueeze(1).expand(-1, F, -1, -1)

                if self.training:
                    p_reset = self.memory_path_dropout_rate
                    keep_prob = 1.0 - p_reset
                    keep_mask = torch.bernoulli(
                        torch.full((BN, 1, 1, 1), keep_prob, device=tokens.device)
                    )
                    current_memory = keep_mask * history_memory + (1 - keep_mask) * init_memory
                    current_memory = torch.nn.functional.dropout(
                        current_memory, p=self.memory_channel_dropout_rate, training=True
                    )
                else:
                    current_memory = history_memory

            tokens_spatiotemporal = torch.cat([tokens, current_memory], dim=2)
        else:
            tokens_spatiotemporal = tokens

        tokens_spatiotemporal = rearrange(tokens_spatiotemporal, "BN F W C -> (BN F) W C")
        for idx, block in enumerate(self.blocks):
            if idx % 2 == 0:
                tokens_spatiotemporal = block(tokens_spatiotemporal)
                tokens_spatiotemporal = rearrange(
                    tokens_spatiotemporal, "(BN F) W C -> (BN W) F C", F=F
                )
            else:
                tokens_spatiotemporal = block(tokens_spatiotemporal)
                tokens_spatiotemporal = rearrange(
                    tokens_spatiotemporal, "(BN W) F C -> (BN F) W C", BN=BN
                )

        tokens_spatiotemporal = rearrange(tokens_spatiotemporal, "(BN F) W C -> BN F W C", F=F)

        if use_walking_memory:
            image_tokens_over_time = tokens_spatiotemporal[:, :, :-1, :]
            updated_memory_tokens = tokens_spatiotemporal[:, :, -1, :]
            next_walking_context = updated_memory_tokens[:, -1, :].unsqueeze(1)
        else:
            next_walking_context = None
            image_tokens_over_time = tokens_spatiotemporal

        gated_tokens = self.gated(image_tokens_over_time)
        aggregated_image_tokens = self.temporal_aggregator(gated_tokens)
        cls_tokens = self.cls_token.expand(BN, -1, -1)
        prefix_tokens = []
        if context_tokens:
            prefix_tokens.extend(context_tokens)
        prefix_tokens.append(cls_tokens)
        clean_memory_token = current_memory[:, 0, :, :]
        prefix_tokens.append(clean_memory_token)
        full_sequence = torch.cat(prefix_tokens + [aggregated_image_tokens], dim=1)
        num_prefix_tokens = len(prefix_tokens)
        processed_sequence = full_sequence
        for block in self.blocks2:
            processed_sequence = block(processed_sequence)

        new_context_token_for_next_iter = processed_sequence[:, num_prefix_tokens - 1, :].unsqueeze(
            1
        )
        processed_image_tokens = processed_sequence[:, num_prefix_tokens:, :]

        if use_walking_memory:
            decoder_prev_state = clean_memory_token.view(BN, C)
        else:
            decoder_prev_state = None

        patch_predictions, next_state_from_decoder = self.patch_head(
            processed_image_tokens, prev_state=decoder_prev_state
        )
        final_predictions = patch_predictions.unsqueeze(1)
        next_state_from_decoder = next_state_from_decoder[0]

        next_walking_context = next_state_from_decoder.unsqueeze(1)
        return final_predictions, new_context_token_for_next_iter, next_walking_context


class F0RefinementBlock(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_k = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 49, dim))
        self._init_pos_embed()

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _init_pos_embed(self):
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, curr_patch, f0_patch):
        """
        curr_patch: [BN, 1, WW, C] (current frame, as Query)
        f0_patch:   [BN, 1, WW, C] (first frame, as Key/Value)
        """
        assert curr_patch.shape[1] == 1
        curr_patch = curr_patch[:, 0]
        f0_patch = f0_patch[:, 0]

        B, N, C = curr_patch.shape

        q_input = curr_patch + self.pos_embed
        k_input = f0_patch + self.pos_embed

        q = self.to_q(self.norm1_q(q_input))
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.to_kv(self.norm1_k(k_input))
        k, v = (
            kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        curr_patch = curr_patch + x

        curr_patch = curr_patch + self.mlp(self.norm2(curr_patch))

        return curr_patch[:, None]

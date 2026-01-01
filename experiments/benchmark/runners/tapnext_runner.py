#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np

from common import (
    ensure_repo_root_on_syspath,
    load_queries_txt,
    load_video_rgb,
    save_result_npz,
    shift_queries_for_clip,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TapNext runner (JAX/Flax) (writes standardized result.npz).")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--queries", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--resized-h", type=int, default=256)
    p.add_argument("--resized-w", type=int, default=256)
    p.add_argument("--infer-h", type=int, default=256)
    p.add_argument("--infer-w", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--ckpt", type=str, required=True, help="TapNext checkpoint (.npz), e.g. bootstapnext_ckpt.npz")
    p.add_argument("--use-certainty", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def npload(fname: str) -> dict:
    """TapNext checkpoint loader from the official colab.

    Only supports local filesystem paths.
    """

    if os.path.exists(fname):
        loaded = np.load(fname, allow_pickle=False)
    else:
        # Minimal URL support (matches colab behavior).
        with open(fname, "rb") as f:
            data = f.read()
        loaded = np.load(io.BytesIO(data), allow_pickle=False)

    if isinstance(loaded, np.ndarray):
        raise TypeError(f"Unexpected np.load return type for {fname}: numpy.ndarray")
    return dict(loaded)


def recover_tree(flat_dict: dict) -> dict:
    """Recover nested param tree from a flattened 'a/b/c' key dict (colab helper)."""

    tree: dict = {}
    for k, v in flat_dict.items():
        parts = str(k).split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return tree


def _resize_frames_u8(frames_thwc_u8: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    in_h, in_w = int(frames_thwc_u8.shape[1]), int(frames_thwc_u8.shape[2])
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if (in_h, in_w) == (out_h, out_w):
        return frames_thwc_u8

    import cv2  # local import (opencv is optional for NPZ inputs)

    out = []
    for frame in frames_thwc_u8:
        out.append(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0).astype(np.uint8, copy=False)


def _preprocess_frames(frames_thwc_u8: np.ndarray) -> np.ndarray:
    # TapNext expects [-1, 1] float, channels-last.
    frames = frames_thwc_u8.astype(np.float32, copy=False)
    return frames / 255.0 * 2.0 - 1.0


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()

    infer_hw = (int(args.infer_h), int(args.infer_w))
    if infer_hw != (256, 256):
        # The official TapNext colab hard-codes 256x256 positional embeddings.
        raise ValueError(f"TapNext infer size must be 256x256, got {infer_hw[1]}x{infer_hw[0]}")

    # Import JAX deps after parsing args (keeps error messages cleaner).
    import einops
    import flax.linen as nn
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    # ---------------------------
    # Official TapNext model code
    # ---------------------------

    class MlpBlock(nn.Module):
        @nn.compact
        def __call__(self, x):
            d = x.shape[-1]
            x = nn.gelu(nn.Dense(4 * d)(x))
            return nn.Dense(d)(x)

    class ViTBlock(nn.Module):
        num_heads: int = 12

        @nn.compact
        def __call__(self, x):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = MlpBlock()(y)
            x = x + y
            return x

    class Einsum(nn.Module):
        width: int = 768

        def setup(self):
            self.w = self.param("w", nn.initializers.zeros_init(), (2, self.width, self.width * 4))
            self.b = self.param("b", nn.initializers.zeros_init(), (2, 1, 1, self.width * 4))[:, 0]

        def __call__(self, x):
            return jnp.einsum("...d,cdD->c...D", x, self.w) + self.b

    class RMSNorm(nn.Module):
        width: int = 768

        def setup(self):
            self.scale = self.param("scale", nn.initializers.zeros_init(), (self.width))

        def __call__(self, x):
            var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            normed_x = x * jax.lax.rsqrt(var + 1e-6)
            scale = jnp.expand_dims(self.scale, axis=range(len(x.shape) - 1))
            return normed_x * (scale + 1)

    class Conv1D(nn.Module):
        width: int = 768
        kernel_size: int = 4

        def setup(self):
            self.w = self.param("w", nn.initializers.zeros_init(), (self.kernel_size, self.width))
            self.b = self.param("b", nn.initializers.zeros_init(), (self.width))

        def __call__(self, x, state):
            if state is None:
                state = jnp.zeros((x.shape[0], self.kernel_size - 1, x.shape[1]), dtype=x.dtype)
            x = jnp.concatenate([state, x[:, None]], axis=1)  # shape: (b, k, c)
            out = (x * self.w[None]).sum(axis=-2) + self.b[None]  # shape: (b, c)
            state = x[:, 1 - self.kernel_size :]  # shape: (b, k - 1, c)
            return out, state

    class BlockDiagonalLinear(nn.Module):
        width: int = 768
        num_heads: int = 12

        def setup(self):
            width = self.width // self.num_heads
            self.w = self.param("w", nn.initializers.zeros_init(), (self.num_heads, width, width))
            self.b = self.param("b", nn.initializers.zeros_init(), (self.num_heads, width))

        def __call__(self, x):
            x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_heads)
            y = jnp.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
            return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_heads)

    class RGLRU(nn.Module):
        width: int = 768
        num_heads: int = 12

        def setup(self):
            self.a_real_param = self.param("a_param", nn.initializers.zeros_init(), (self.width))
            self.input_gate = BlockDiagonalLinear(self.width, self.num_heads, name="input_gate")
            self.a_gate = BlockDiagonalLinear(self.width, self.num_heads, name="a_gate")

        def __call__(self, x, state):
            gate_x = jnn.sigmoid(self.input_gate(x))
            if state is None:
                return gate_x * x  # No memory accumulation, return directly
            gate_a = jnn.sigmoid(self.a_gate(x))
            log_a = -8.0 * gate_a * jnn.softplus(self.a_real_param)
            a = jnp.exp(log_a)
            scale_factor = jnp.sqrt(1 - jnp.exp(2 * log_a))  # Compute decay factor
            return a * state + gate_x * x * scale_factor  # Memory update

    class MLPBlock(nn.Module):
        width: int = 768

        def setup(self):
            self.ffw_up = Einsum(self.width, name="ffw_up")
            self.ffw_down = nn.Dense(self.width, name="ffw_down")

        def __call__(self, x):
            out = self.ffw_up(x)
            return self.ffw_down(nn.gelu(out[0]) * out[1])

    class RecurrentBlock(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4

        def setup(self) -> None:
            self.linear_y = nn.Dense(self.width, name="linear_y")
            self.linear_x = nn.Dense(self.width, name="linear_x")
            self.conv_1d = Conv1D(self.width, self.kernel_size, name="conv_1d")
            self.lru = RGLRU(self.width, self.num_heads, name="rg_lru")
            self.linear_out = nn.Dense(self.width, name="linear_out")

        def __call__(self, x, state):
            y = jax.nn.gelu(self.linear_y(x))
            x = self.linear_x(x)
            x, conv1d_state = self.conv_1d(x, None if state is None else state["conv1d_state"])
            rg_lru_state = self.lru(x, None if state is None else state["rg_lru_state"])
            x = self.linear_out(rg_lru_state * y)
            return x, {"rg_lru_state": rg_lru_state, "conv1d_state": conv1d_state}

    class ResidualBlock(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4

        def setup(self):
            self.temporal_pre_norm = RMSNorm(self.width)
            self.recurrent_block = RecurrentBlock(self.width, self.num_heads, self.kernel_size, name="recurrent_block")
            self.channel_pre_norm = RMSNorm(self.width)
            self.mlp = MLPBlock(self.width, name="mlp_block")

        def __call__(self, x, state):
            y = self.temporal_pre_norm(x)
            y, state = self.recurrent_block(y, state)
            x = x + y
            y = self.mlp(self.channel_pre_norm(x))
            x = x + y
            return x, state

    class ViTSSMBlock(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4

        def setup(self):
            self.ssm_block = ResidualBlock(self.width, self.num_heads, self.kernel_size)
            self.vit_block = ViTBlock(self.num_heads)

        def __call__(self, x, state):
            b = x.shape[0]
            x = einops.rearrange(x, "b n c -> (b n) c")
            x, state = self.ssm_block(x, state)
            x = einops.rearrange(x, "(b n) c -> b n c", b=b)
            x = self.vit_block(x)
            return x, state

    class ViTSSMBackbone(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4
        num_blocks: int = 12

        def setup(self):
            self.blocks = [
                ViTSSMBlock(self.width, self.num_heads, self.kernel_size, name=f"encoderblock_{i}")
                for i in range(self.num_blocks)
            ]
            self.encoder_norm = nn.LayerNorm()

        def __call__(self, x, state):
            new_states = []
            for i in range(self.num_blocks):
                x, new_state = self.blocks[i](x, None if state is None else state[i])
                new_states.append(new_state)
            x = self.encoder_norm(x)
            return x, new_states

    def posemb_sincos_2d(h, w, width):
        """Compute 2D sine-cosine positional embeddings following MoCo v3 logic."""
        y, x = jnp.mgrid[0:h, 0:w]
        freqs = jnp.linspace(0, 1, num=width // 4, endpoint=True)
        inv_freq = 1.0 / (10_000**freqs)
        y = jnp.einsum("h w, d -> h w d", y, inv_freq)
        x = jnp.einsum("h w, d -> h w d", x, inv_freq)
        pos_emb = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1)
        return pos_emb

    class Backbone(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4
        num_blocks: int = 12

        def setup(self):
            self.lin_proj = nn.Conv(self.width, (1, 8, 8), strides=(1, 8, 8), padding="VALID", name="embedding")
            self.mask_token = self.param("mask_token", nn.initializers.zeros_init(), (1, 1, 1, self.width))[:, 0]
            self.unknown_token = self.param("unknown_token", nn.initializers.zeros_init(), (1, 1, self.width))
            self.point_query_token = self.param(
                "point_query_token", nn.initializers.zeros_init(), (1, 1, 1, self.width)
            )[:, 0]
            self.image_pos_emb = self.param(
                "pos_embedding", nn.initializers.zeros_init(), (1, 256 // 8 * 256 // 8, self.width)
            )
            self.encoder = ViTSSMBackbone(self.width, self.num_heads, self.kernel_size, self.num_blocks, name="Transformer")

        def __call__(self, frame, query_points, step, state):
            x = self.lin_proj(frame)  # x: (b, h, w, c)
            b, h, w, c = x.shape
            query_points = jnp.concatenate([query_points[..., :1] - step, query_points[..., 1:]], axis=-1)  # (b, q, 3)
            posemb2d = posemb_sincos_2d(256, 256, self.width)  # (256,256,c)

            def interp(x_, y_):
                return jax.scipy.ndimage.map_coordinates(x_, y_.T - 0.5, order=1, mode="nearest")

            interp_fn = jax.vmap(interp, in_axes=(-1, None), out_axes=-1)
            interp_fn = jax.vmap(interp_fn, in_axes=(None, 0), out_axes=0)
            point_tokens = self.point_query_token + interp_fn(posemb2d, query_points[..., 1:])  # (b, q, c)

            # Query tokens
            query_timesteps = query_points[..., 0:1].astype(jnp.int32)  # (b, q, 1)
            query_tokens = jnp.where(query_timesteps > 0, self.unknown_token, self.mask_token)  # (b,q,c)
            query_tokens = jnp.where(query_timesteps == 0, point_tokens, query_tokens)  # (b,q,c)

            # Image tokens
            image_tokens = jnp.reshape(x, [b, h * w, c]) + self.image_pos_emb  # (b, h*w, c)

            x = jnp.concatenate([image_tokens, query_tokens], axis=-2)  # (b, h*w+q, c)
            x, state = self.encoder(x, state)
            _, q, _ = query_points.shape
            x = x[:, -q:, :]  # (b,q,c)
            return x, state

    def get_window(coord, softmax, radius=6):
        """Note: coord is assumed to be a raster coordinate."""
        start = jnp.maximum(jnp.array(jnp.floor(coord - radius - 0.5), jnp.int32), 0)
        softmax = jax.lax.dynamic_slice(softmax, [start], [radius * 2 + 1])
        coord = start + 0.5 + jnp.arange(radius * 2 + 1)
        return softmax, coord

    def get_uncertainty(coord_yx, track_logits, radius=6):
        """Get uncertainty from coordinate logits for a single point/frame."""
        logits_y, logits_x = jnp.split(track_logits, 2, axis=-1)
        track_softmax_y = jax.nn.softmax(logits_y)
        track_softmax_x = jax.nn.softmax(logits_x)
        sm_y, coord_y = get_window(coord_yx[0], track_softmax_y)
        sm_x, coord_x = get_window(coord_yx[1], track_softmax_x)
        sm = sm_y[:, jnp.newaxis] * sm_x[jnp.newaxis, :]
        grid_x, grid_y = jnp.meshgrid(coord_x, coord_y)
        grid = jnp.stack([grid_y, grid_x], axis=-1)
        in_radius = jnp.sum(jnp.square(grid - coord_yx), axis=-1) <= jnp.square(radius) + 1e-8
        return jnp.sum(sm * in_radius)

    def tracker_uncertainty(tracks, track_logits):
        """Get uncertainty for all points/frames in a batch."""
        vmapped_uncertain_fn = get_uncertainty
        for _ in range(len(tracks.shape) - 1):
            vmapped_uncertain_fn = jax.vmap(vmapped_uncertain_fn)
        uncertainty = vmapped_uncertain_fn(tracks, track_logits)
        return uncertainty[..., jnp.newaxis]

    class TAPNext(nn.Module):
        width: int = 768
        num_heads: int = 12
        kernel_size: int = 4
        num_blocks: int = 12
        use_certainty: bool = True

        def setup(self):
            self.backbone = Backbone(self.width, self.num_heads, self.kernel_size, self.num_blocks)
            self.visible_head = nn.Sequential(
                [
                    nn.Dense(256),
                    nn.LayerNorm(),
                    nn.gelu,
                    nn.Dense(256),
                    nn.LayerNorm(),
                    nn.gelu,
                    nn.Dense(1),
                ]
            )
            self.coordinate_head = nn.Sequential(
                [
                    nn.Dense(256),
                    nn.LayerNorm(),
                    nn.gelu,
                    nn.Dense(256),
                    nn.LayerNorm(),
                    nn.gelu,
                    nn.Dense(512),
                ]
            )

        @nn.compact
        def __call__(self, frame, query_points, step, state):
            feat, state = self.backbone(frame, query_points, step, state)
            track_logits = self.coordinate_head(feat)
            visible_logits = self.visible_head(feat)

            position_x, position_y = jnp.split(track_logits, 2, axis=-1)
            position = jnp.stack([position_x, position_y], axis=-2)
            index = jnp.arange(position.shape[-1])[None, None, None]
            argmax = jnp.argmax(position, axis=-1, keepdims=True)
            mask = jnp.abs(argmax - index) <= 20
            probs = jnn.softmax(position * 0.5, axis=-1) * mask
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
            tracks = jnp.sum(probs * index, axis=-1) + 0.5

            if self.use_certainty:
                certain = tracker_uncertainty(tracks, track_logits)
                visible = ((jax.nn.sigmoid(visible_logits) * certain) > 0.5).astype(jnp.float32)
            else:
                visible = (visible_logits > 0).astype(jnp.float32)
            return tracks, visible, state

    model = TAPNext(use_certainty=bool(args.use_certainty))

    @jax.jit
    def forward(params, frame, query_points, step, state):
        tracks, visible, state = model.apply({"params": params}, frame, query_points, step, state)
        return tracks, visible, state

    # -----
    # Inputs
    # -----

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TapNext checkpoint not found: {ckpt_path}")
    params = recover_tree(npload(str(ckpt_path)))
    params = jax.tree_util.tree_map(jnp.asarray, params)

    resized_hw = (int(args.resized_h), int(args.resized_w))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None
    frames_vis = load_video_rgb(args.video, resized_hw=resized_hw, start=int(args.start), max_frames=max_frames)
    T = int(frames_vis.shape[0])

    frames_infer_u8 = _resize_frames_u8(frames_vis, out_hw=infer_hw)
    frames_infer = _preprocess_frames(frames_infer_u8)

    # Queries are (t, x, y) in vis coords; shift for clip start, scale to infer, then convert to (t, y, x).
    queries = load_queries_txt(args.queries)
    queries = shift_queries_for_clip(queries, start=int(args.start), clip_len=T)
    q_txy = queries.txy

    q_t = np.clip(np.round(q_txy[:, 0]), 0, T - 1).astype(np.float32)
    scale_x = float(infer_hw[1]) / float(resized_hw[1])
    scale_y = float(infer_hw[0]) / float(resized_hw[0])
    q_x_infer = q_txy[:, 1] * scale_x
    q_y_infer = q_txy[:, 2] * scale_y
    q_tyx = np.stack([q_t, q_y_infer, q_x_infer], axis=1).astype(np.float32)  # (N,3)

    query_points = jnp.asarray(q_tyx)[None, ...]  # (1,N,3)  [t, y, x]
    frames_jnp = jnp.asarray(frames_infer)[None, ...]  # (1,T,H,W,3)

    print(f"[Info] JAX backend: {jax.default_backend()}")
    print(f"[Info] JAX devices: {[d.platform + ':' + str(d.id) for d in jax.devices()]}")
    print(f"[Info] video: {args.video} clip_len={T} vis_hw={resized_hw[0]}x{resized_hw[1]} infer_hw=256x256")
    print(f"[Info] queries: {args.queries} N={int(q_txy.shape[0])}")
    print(f"[Info] ckpt: {ckpt_path}")
    print("[Info] Running inference...")

    # Inference loop (recurrent state).
    tracks_list = []
    visibles_list = []
    state = None

    t0 = time.time()
    last_tracks = None
    for t in range(T):
        frame = frames_jnp[:, t]  # (1,H,W,3)
        pred_tracks, pred_visible, state = forward(params, frame, query_points, t, state)
        tracks_list.append(pred_tracks)
        visibles_list.append(pred_visible)
        last_tracks = pred_tracks
    assert last_tracks is not None
    jax.block_until_ready(last_tracks)
    dt = time.time() - t0
    print(f"[Info] Done. Time: {dt:.3f}s")

    # Stack -> (T,1,N,2)/(T,1,N,1), then convert coords + scale back to vis.
    tracks_tbn2 = jnp.stack(tracks_list, axis=0)  # (T,1,N,2) in (y,x) @ infer
    vis_tbn1 = jnp.stack(visibles_list, axis=0)  # (T,1,N,1)

    # device_get can return a read-only view backed by JAX buffers; make it writable.
    tracks_tn2 = np.array(jax.device_get(tracks_tbn2[:, 0, :, ::-1]), dtype=np.float32, copy=True)  # (T,N,2) (x,y)
    visibles_tn = np.asarray(jax.device_get(vis_tbn1[:, 0, :, 0]) > 0.5, dtype=np.bool_)  # (T,N) bool

    # Scale to visualization resolution.
    tracks_tn2[..., 0] *= float(resized_hw[1]) / float(infer_hw[1])
    tracks_tn2[..., 1] *= float(resized_hw[0]) / float(infer_hw[0])

    save_result_npz(
        out_npz=args.out_npz,
        method="tapnext",
        video_path=args.video,
        resized_hw=resized_hw,
        queries=queries,
        tracks_xy_tn2=tracks_tn2,
        visibles_tn=visibles_tn,
        runtime_sec=float(dt),
        meta={
            "ckpt": str(ckpt_path),
            "infer_hw": [infer_hw[0], infer_hw[1]],
            "use_certainty": bool(args.use_certainty),
        },
    )
    print(f"[Done] Wrote: {args.out_npz}")


if __name__ == "__main__":
    main()

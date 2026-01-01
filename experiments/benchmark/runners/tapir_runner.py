#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from common import (
    ensure_repo_root_on_syspath,
    load_queries_txt,
    load_video_rgb,
    save_result_npz,
    shift_queries_for_clip,
)


def _infer_tapir_variant_from_ckpt(ckpt_path: Path) -> str:
    """Infer TAPIR variant from checkpoint filename.

    TapNet PyTorch checkpoints do not currently bundle a config, so we infer the
    most likely variant from common naming patterns.
    """

    name = ckpt_path.name.lower()
    is_bootstapir = "bootstapir" in name
    is_causal = "causal" in name

    if is_bootstapir and is_causal:
        return "bootstapir_online"
    if is_bootstapir:
        return "bootstapir"
    if is_causal:
        return "tapir_online"
    return "tapir"


def _tapir_model_kwargs_for_variant(variant: str) -> dict:
    """Return TapNet torch.TAPIR kwargs for a given variant.

    Values are aligned with TapNet configs/colabs:
    - TAPIR: pyramid_level=0, no extra_convs, softmax_temperature default (20.0)
    - BootsTAPIR: pyramid_level=1, extra_convs, softmax_temperature=10.0
    - "online"/causal variants: use_casual_conv=True
    """

    v = str(variant).strip().lower()
    if v not in {"tapir", "tapir_online", "bootstapir", "bootstapir_online"}:
        raise ValueError(f"Unknown TAPIR variant: {variant}")

    if v.startswith("bootstapir"):
        # See tapnet/configs/tapir_bootstrap_config.py: softmax_temperature=10.0.
        kwargs = {"pyramid_level": 1, "extra_convs": True, "softmax_temperature": 10.0}
    else:
        kwargs = {"pyramid_level": 0, "extra_convs": False, "softmax_temperature": 20.0}

    kwargs["use_casual_conv"] = v.endswith("_online")
    return kwargs


def _add_to_syspath(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _import_tapir_torch_model(tapnet_root: Path):
    """
    Import TapNet's torch_model without importing the top-level `tapnet/__init__.py`.

    Upstream TapNet imports JAX-only deps in `tapnet/__init__.py` (chex/jax),
    but we only need the PyTorch TAPIR implementation under tapnet/torch_model/.
    """

    pkg_dir = tapnet_root / "tapnet"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"tapnet package dir not found: {pkg_dir}")

    if "tapnet" not in sys.modules:
        stub = types.ModuleType("tapnet")
        stub.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        sys.modules["tapnet"] = stub

    # TapNet has had multiple layouts for PyTorch TAPIR over time.
    # Try the known locations.
    last_err: Exception | None = None
    for mod in ["tapnet.torch.tapir_model", "tapnet.torch_model.tapir_model"]:
        try:
            return importlib.import_module(mod)
        except ModuleNotFoundError as e:
            last_err = e
    assert last_err is not None
    raise last_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAPIR runner (writes standardized result.npz).")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--queries", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--resized-h", type=int, default=384)
    p.add_argument("--resized-w", type=int, default=512)
    p.add_argument("--infer-h", type=int, default=256)
    p.add_argument("--infer-w", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--tapnet-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "tapir", "tapir_online", "bootstapir", "bootstapir_online"],
        help="Model variant. 'auto' infers from checkpoint filename.",
    )
    p.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help="Override TAPIR softmax_temperature (default depends on variant).",
    )
    p.add_argument(
        "--use-casual-conv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TAPIR use_casual_conv (default depends on variant).",
    )
    return p.parse_args()


def preprocess_frames(frames_thwc_u8: torch.Tensor) -> torch.Tensor:
    # TAPIR expects [-1, 1] float, channels-last.
    frames = frames_thwc_u8.float()
    frames = frames / 255.0 * 2.0 - 1.0
    return frames


def postprocess_visibles(occlusions: torch.Tensor, expected_dist: torch.Tensor) -> torch.Tensor:
    # Matches tapnet/demo_utils.py behavior.
    return (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()

    if args.infer_h % 8 != 0 or args.infer_w % 8 != 0:
        raise ValueError(f"infer size must be multiple of 8, got {args.infer_w}x{args.infer_h}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")

    tapnet_root = Path(args.tapnet_root).resolve()
    if not tapnet_root.exists():
        raise FileNotFoundError(f"tapnet root not found: {tapnet_root}")
    _add_to_syspath(tapnet_root)
    tapir_model = _import_tapir_torch_model(tapnet_root)

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TAPIR checkpoint not found: {ckpt_path}")

    variant = str(args.variant)
    if variant == "auto":
        variant = _infer_tapir_variant_from_ckpt(ckpt_path)
    model_kwargs = _tapir_model_kwargs_for_variant(variant)
    if args.softmax_temperature is not None:
        model_kwargs["softmax_temperature"] = float(args.softmax_temperature)
    if args.use_casual_conv is not None:
        model_kwargs["use_casual_conv"] = bool(args.use_casual_conv)

    print(
        "[Info] TAPIR config:"
        f" variant={variant}"
        f" pyramid_level={model_kwargs.get('pyramid_level')}"
        f" extra_convs={model_kwargs.get('extra_convs')}"
        f" use_casual_conv={model_kwargs.get('use_casual_conv')}"
        f" softmax_temperature={model_kwargs.get('softmax_temperature')}"
    )

    resized_hw = (int(args.resized_h), int(args.resized_w))
    infer_hw = (int(args.infer_h), int(args.infer_w))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    # Load clip at visualization resolution.
    video_vis = load_video_rgb(args.video, resized_hw=resized_hw, start=int(args.start), max_frames=max_frames)
    T = int(video_vis.shape[0])

    # Optionally resize for inference.
    if infer_hw == resized_hw:
        video_infer = video_vis
    else:
        import cv2  # local import to keep deps minimal

        out = []
        for frame in video_vis:
            out.append(cv2.resize(frame, (infer_hw[1], infer_hw[0]), interpolation=cv2.INTER_LINEAR))
        video_infer = np.stack(out, axis=0).astype(np.uint8, copy=False)

    # Queries (t,x,y) in vis coords; shift for clip start and convert to (t,y,x) in infer coords.
    queries = load_queries_txt(args.queries)
    queries = shift_queries_for_clip(queries, start=int(args.start), clip_len=T)
    q_txy = queries.txy  # (N,3)
    q_t = np.clip(np.round(q_txy[:, 0]), 0, T - 1).astype(np.float32)

    scale_x = float(infer_hw[1]) / float(resized_hw[1])
    scale_y = float(infer_hw[0]) / float(resized_hw[0])
    q_x_infer = q_txy[:, 1] * scale_x
    q_y_infer = q_txy[:, 2] * scale_y
    q_tyx = np.stack([q_t, q_y_infer, q_x_infer], axis=1).astype(np.float32)  # (N,3)
    query_points = torch.from_numpy(q_tyx).to(device).unsqueeze(0)  # (1,N,3)

    # Model
    model = tapir_model.TAPIR(**model_kwargs)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(device).eval()

    print("[Info] Running inference...")
    frames = torch.from_numpy(video_infer).to(device)  # (T,H,W,3) uint8
    video_tensor = preprocess_frames(frames).unsqueeze(0)  # (1,T,H,W,3) float

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        outputs = model(video_tensor, query_points)
        tracks_bnt2 = outputs["tracks"]  # (B,N,T,2) in (x,y) @ infer
        occ_bnt = outputs["occlusion"]  # (B,N,T)
        exp_bnt = outputs["expected_dist"]  # (B,N,T)
        vis_bnt = postprocess_visibles(occ_bnt, exp_bnt)  # (B,N,T) bool
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print(f"[Info] Done. Time: {dt:.3f}s")

    # Convert to (T,N,2) @ infer, then scale back to vis.
    tracks_tn2 = tracks_bnt2[0].permute(1, 0, 2).contiguous()  # (T,N,2)
    vis_tn = vis_bnt[0].permute(1, 0).contiguous()  # (T,N)

    tracks_tn2 = tracks_tn2.clone()
    tracks_tn2[..., 0] *= float(resized_hw[1]) / float(infer_hw[1])
    tracks_tn2[..., 1] *= float(resized_hw[0]) / float(infer_hw[0])

    save_result_npz(
        out_npz=args.out_npz,
        method="tapir",
        video_path=args.video,
        resized_hw=resized_hw,
        queries=queries,
        tracks_xy_tn2=tracks_tn2.detach().cpu().numpy().astype(np.float32, copy=False),
        visibles_tn=vis_tn.detach().cpu().numpy().astype(np.bool_, copy=False),
        runtime_sec=float(dt),
        meta={
            "ckpt": str(ckpt_path),
            "infer_hw": [infer_hw[0], infer_hw[1]],
            "variant": variant,
            "model_kwargs": model_kwargs,
        },
    )
    print(f"[Done] Wrote: {args.out_npz}")


if __name__ == "__main__":
    main()

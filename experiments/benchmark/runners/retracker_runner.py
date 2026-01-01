#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _install_retracker_import_stubs() -> None:
    """
    Some local copies of re-tracker may contain unresolved git conflict markers
    in modules that are not needed for the default checkpoint. If those modules
    are imported unconditionally, *any* import can fail.

    We stub a few known-problematic modules so that DINOv3-based checkpoints can
    still be loaded.
    """
    from torch import nn

    def _install(module_name: str, attrs: dict[str, object]) -> None:
        if module_name in sys.modules:
            return
        stub = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(stub, k, v)
        sys.modules[module_name] = stub

    class _Broken(nn.Module):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Import stub hit for a broken/missing re-tracker module. "
                "Fix the re-tracker checkout if your checkpoint depends on it."
            )

    _install("retracker.models.backbone.dino_backbone_v0", {"DINO_backbone": _Broken})
    _install(
        "retracker.models.modules.deprecated_transformer",
        {
            "TriLocalFeatureTransformer": _Broken,
            "Mem_Featmap_Transformer": _Broken,
            "TransformerEncoder": _Broken,
            "TransformerCrossEncoder": _Broken,
            "TransformerSelfCrossEncoder": _Broken,
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReTracker runner (writes standardized result.npz).")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--queries", type=str, required=True, help="Queries txt (t x y)")
    p.add_argument("--ckpt", type=str, required=True, help="ReTracker checkpoint (.ckpt)")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--resized-h", type=int, default=384)
    p.add_argument("--resized-w", type=int, default=512)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out-npz", type=str, required=True)
    return p.parse_args()


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")

    _install_retracker_import_stubs()
    from retracker.inference.engine import ReTrackerEngine  # noqa: E402

    resized_hw = (int(args.resized_h), int(args.resized_w))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    # Load video clip (RGB uint8) and move to torch in BTCHW format.
    video_rgb = load_video_rgb(args.video, resized_hw=resized_hw, start=int(args.start), max_frames=max_frames)
    T = int(video_rgb.shape[0])
    video_tchw = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).contiguous().float()  # (T,3,H,W)
    video_tchw = video_tchw.to(device)
    video_btchw = video_tchw.unsqueeze(0)  # (1,T,3,H,W)

    # Queries (t,x,y) in vis coords; shift t for clip start.
    queries = load_queries_txt(args.queries)
    queries = shift_queries_for_clip(queries, start=int(args.start), clip_len=T)
    q = torch.from_numpy(queries.txy).to(device).unsqueeze(0)  # (1,N,3)

    # Build engine (demo default uses interp_shape 512x512).
    engine = ReTrackerEngine(ckpt_path=str(Path(args.ckpt).resolve()), interp_shape=(512, 512))
    engine = engine.eval().to(device)
    engine.set_task_mode("tracking")
    engine.set_visibility_thresholds(tracking=0.1, matching=0.5)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]

    print("[Info] Running inference...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda" and amp_dtype != torch.float32)):
        pred_trj, pred_vsb = engine.video_forward(video_btchw, q, use_aug=False)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print(f"[Info] Done. Time: {dt:.3f}s")

    # Standardize shapes: (T,N,2) + (T,N)
    tracks_tn2 = pred_trj[0].detach().cpu().numpy().astype(np.float32, copy=False)
    vis_tn = pred_vsb[0].detach().cpu().numpy().astype(bool, copy=False)

    save_result_npz(
        out_npz=args.out_npz,
        method="retracker",
        video_path=args.video,
        resized_hw=resized_hw,
        queries=queries,
        tracks_xy_tn2=tracks_tn2,
        visibles_tn=vis_tn,
        runtime_sec=float(dt),
        meta={"dtype": args.dtype, "interp_shape": [512, 512]},
    )
    print(f"[Done] Wrote: {args.out_npz}")


if __name__ == "__main__":
    main()

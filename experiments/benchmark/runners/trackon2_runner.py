#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
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


def _add_to_syspath(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track-On2 runner (writes standardized result.npz).")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--queries", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--resized-h", type=int, default=384)
    p.add_argument("--resized-w", type=int, default=512)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--trackon-root", type=str, required=True, help="Path to third_party/track_on")
    p.add_argument("--config", type=str, required=True, help="Track-On2 config yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Track-On2 checkpoint (.pt)")
    p.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional dataset identifier (used to mirror official eval-time overrides, e.g. DAVIS).",
    )
    p.add_argument(
        "--support-grid-size",
        type=int,
        default=20,
        help="Support grid size (S); adds S^2 auxiliary queries at t=0. Official Track-On2 eval uses 20.",
    )
    return p.parse_args()


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")

    trackon_root = Path(args.trackon_root).resolve()
    if not trackon_root.exists():
        raise FileNotFoundError(f"trackon root not found: {trackon_root}")
    _add_to_syspath(trackon_root)

    from model.trackon_predictor import Predictor  # noqa: E402
    from utils.train_utils import load_args_from_yaml  # noqa: E402

    resized_hw = (int(args.resized_h), int(args.resized_w))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    # Video -> (1, T, 3, H, W) float
    video_rgb = load_video_rgb(args.video, resized_hw=resized_hw, start=int(args.start), max_frames=max_frames)
    video_tchw = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).contiguous().float()
    video_btchw = video_tchw.unsqueeze(0).to(device)
    T = int(video_btchw.shape[1])

    # Queries (t,x,y) -> shift t for clip start.
    queries = load_queries_txt(args.queries)
    queries = shift_queries_for_clip(queries, start=int(args.start), clip_len=T)
    q = torch.from_numpy(queries.txy).unsqueeze(0).to(device)

    # Model
    model_args = load_args_from_yaml(str(Path(args.config).resolve()))
    # Mirror official Track-On2 eval overrides:
    # - DAVIS uses a smaller inference-time memory extension (M_i=24).
    if str(args.dataset_name).lower().find("davis") >= 0:
        model_args.M_i = 24
    model = Predictor(
        model_args,
        checkpoint_path=str(Path(args.ckpt).resolve()),
        support_grid_size=int(args.support_grid_size),
    ).to(device).eval()

    print("[Info] Running inference...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        tracks, visibles = model(video_btchw, queries=q)  # (1,T,N,2), (1,T,N)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print(f"[Info] Done. Time: {dt:.3f}s")

    save_result_npz(
        out_npz=args.out_npz,
        method="trackon2",
        video_path=args.video,
        resized_hw=resized_hw,
        queries=queries,
        tracks_xy_tn2=tracks[0].detach().cpu().numpy().astype(np.float32, copy=False),
        visibles_tn=visibles[0].detach().cpu().numpy().astype(bool, copy=False),
        runtime_sec=float(dt),
        meta={"ckpt": str(Path(args.ckpt).resolve()), "config": str(Path(args.config).resolve())},
    )
    print(f"[Done] Wrote: {args.out_npz}")


if __name__ == "__main__":
    main()

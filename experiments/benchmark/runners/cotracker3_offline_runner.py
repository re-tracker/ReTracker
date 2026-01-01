#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
    p = argparse.ArgumentParser(description="CoTracker3 offline runner (writes standardized result.npz).")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--queries", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--resized-h", type=int, default=384)
    p.add_argument("--resized-w", type=int, default=512)
    p.add_argument("--interp-h", type=int, default=384)
    p.add_argument("--interp-w", type=int, default=512)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--cotracker-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--window-len", type=int, default=60, help="Model window length (offline checkpoint is typically trained with 60).")
    p.add_argument("--grid-size", type=int, default=5, help="Support grid size (N); adds N^2 aux points at t=0 (official eval uses 5).")
    p.add_argument("--n-iters", type=int, default=6, help="Number of refinement iterations (official eval uses 6).")
    p.add_argument("--vis-thr", type=float, default=0.6, help="Visibility threshold used for TAP-Vid metrics in official eval.")
    p.add_argument(
        "--query-batch-size",
        type=int,
        default=0,
        help=(
            "If >0, split queries into chunks to reduce peak memory. "
            "If 0, run all queries at once and automatically fall back to chunking on OOM."
        ),
    )
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Autocast dtype for GPU inference (reduces memory).",
    )
    return p.parse_args()

def _autocast_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()

    # Helps reduce fragmentation for long runs / multi-process eval.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")

    cotracker_root = Path(args.cotracker_root).resolve()
    if not cotracker_root.exists():
        raise FileNotFoundError(f"cotracker root not found: {cotracker_root}")
    _add_to_syspath(cotracker_root)
    from cotracker.models.build_cotracker import build_cotracker  # noqa: E402
    from cotracker.models.evaluation_predictor import EvaluationPredictor  # noqa: E402

    resized_hw = (int(args.resized_h), int(args.resized_w))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    video_rgb = load_video_rgb(args.video, resized_hw=resized_hw, start=int(args.start), max_frames=max_frames)
    video_tchw = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).contiguous().float()
    video_btchw = video_tchw.unsqueeze(0).to(device)
    T = int(video_btchw.shape[1])

    queries = load_queries_txt(args.queries)
    queries = shift_queries_for_clip(queries, start=int(args.start), clip_len=T)
    q = torch.from_numpy(queries.txy).unsqueeze(0).to(device)

    ckpt_path = str(Path(args.ckpt).resolve())
    model = build_cotracker(checkpoint=ckpt_path, offline=True, window_len=int(args.window_len), v2=False)
    interp_hw = (int(args.interp_h), int(args.interp_w))
    predictor = EvaluationPredictor(
        model,
        interp_shape=interp_hw,
        grid_size=int(args.grid_size),
        local_grid_size=8,
        sift_size=0,
        single_point=False,
        num_uniformly_sampled_pts=0,
        n_iters=int(args.n_iters),
        local_extent=50,
    ).to(device).eval()

    print("[Info] Running inference...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    amp_dtype = _autocast_dtype(str(args.amp_dtype))

    def _run_one(q_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda" and amp_dtype != torch.float32)):
                return predictor(video_btchw, queries=q_chunk)

    # Try full inference first for best-faithfulness to the original model behavior,
    # then fall back to query chunking if the GPU is busy and we OOM.
    N = int(q.shape[1])
    qbs = int(args.query_batch_size)
    force_chunking = bool(qbs > 0 and N > qbs)
    if not force_chunking and device == "cuda" and qbs <= 0 and N > 0:
        # Heuristic: if the GPU is already busy, skip the expensive OOM attempt and
        # go straight to chunking.
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 12 * 1024**3:
            qbs = min(8, N)
            force_chunking = bool(N > qbs)
    try:
        if force_chunking:
            raise RuntimeError("force chunking")  # handled below
        tracks, vis = _run_one(q)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and str(e) != "force chunking":
            raise
        if device != "cuda":
            raise
        torch.cuda.empty_cache()
        # Default chunk size for crowded GPUs; can be overridden via CLI.
        qbs = qbs if qbs > 0 else min(8, N)
        if qbs <= 0:
            raise
        print(f"[Warn] OOM in full-query mode; retry with --query-batch-size={qbs}")
        # Keep outputs on CPU to minimize peak GPU memory while chunking.
        tracks_chunks = torch.empty((1, T, N, 2), device="cpu", dtype=torch.float32)
        vis_chunks = torch.empty((1, T, N), device="cpu", dtype=torch.float32)
        for s in range(0, N, qbs):
            e = min(N, s + qbs)
            tr, vi = _run_one(q[:, s:e])
            tracks_chunks[:, :, s:e] = tr.to(dtype=torch.float32, device="cpu")
            vis_chunks[:, :, s:e] = vi.to(dtype=torch.float32, device="cpu")
            torch.cuda.empty_cache()
        tracks, vis = tracks_chunks, vis_chunks
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print(f"[Info] Done. Time: {dt:.3f}s")

    if vis.ndim == 4 and vis.shape[-1] == 1:
        vis = vis[..., 0]
    if vis.dtype != torch.bool:
        vis = vis > float(args.vis_thr)

    save_result_npz(
        out_npz=args.out_npz,
        method="cotracker3_offline",
        video_path=args.video,
        resized_hw=resized_hw,
        queries=queries,
        tracks_xy_tn2=tracks[0].detach().cpu().numpy().astype(np.float32, copy=False),
        visibles_tn=vis[0].detach().cpu().numpy().astype(bool, copy=False),
        runtime_sec=float(dt),
        meta={
            "ckpt": ckpt_path,
            "window_len": int(args.window_len),
            "interp_hw": [int(interp_hw[0]), int(interp_hw[1])],
            "grid_size": int(args.grid_size),
            "n_iters": int(args.n_iters),
            "vis_thr": float(args.vis_thr),
            "query_batch_size": int(args.query_batch_size),
            "amp_dtype": str(args.amp_dtype),
            "predictor": "EvaluationPredictor",
        },
    )
    print(f"[Done] Wrote: {args.out_npz}")


if __name__ == "__main__":
    main()

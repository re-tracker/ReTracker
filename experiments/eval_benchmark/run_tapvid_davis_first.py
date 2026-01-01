#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Support running as a script: `python experiments/eval_benchmark/run_tapvid_davis_first.py ...`
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from experiments.eval_benchmark.core.io import (  # noqa: E402
    PredictionCache,
    load_prediction_npz,
    save_prediction_npz,
)
from experiments.eval_benchmark.core.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_tapvid_davis_first_metrics,
)
from experiments.eval_benchmark.datasets.tapvid_davis import (  # noqa: E402
    TapVidDavisSequence,
    load_tapvid_davis_first,
    load_tapvid_davis_pickle,
    resolve_existing_pickle_path,
)
from experiments.eval_benchmark.methods.retracker import ReTrackerMethod  # noqa: E402


def _safe_name(name: str) -> str:
    # DAVIS keys are usually safe already (letters, digits, -, _), but sanitize anyway.
    return name.replace(os.sep, "_")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def evaluate_one_sequence(
    seq: TapVidDavisSequence,
    *,
    method: Any,
    pred_path: Path,
    metrics_path: Path,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, float]:
    """Evaluate a single sequence with optional prediction caching."""

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    pred: PredictionCache | None = None

    if pred_path.exists() and resume and not overwrite:
        pred = load_prediction_npz(pred_path)
    else:
        out = method.predict(frames_uint8=seq.frames_uint8, query_points_tyx=seq.query_points_tyx)
        save_prediction_npz(
            pred_path,
            pred_tracks_xy=out.pred_tracks_xy,
            pred_occluded=out.pred_occluded,
            runtime_sec=out.runtime_sec,
            meta={"method": getattr(method, "name", "unknown")},
        )
        pred = PredictionCache(
            pred_tracks_xy=out.pred_tracks_xy,
            pred_occluded=out.pred_occluded,
            runtime_sec=out.runtime_sec,
            meta={"method": getattr(method, "name", "unknown")},
        )

    assert pred is not None

    metrics = compute_tapvid_davis_first_metrics(
        query_points_tyx=seq.query_points_tyx,
        gt_occluded=seq.gt_occluded,
        gt_tracks_xy=seq.gt_tracks_xy,
        pred_occluded=pred.pred_occluded,
        pred_tracks_xy=pred.pred_tracks_xy,
    )

    # Persist per-video metrics for resume/aggregation.
    metrics_payload: dict[str, Any] = {
        "seq_name": seq.name,
        "runtime_sec": pred.runtime_sec,
        "metrics": metrics,
    }
    _write_json(metrics_path, metrics_payload)

    return metrics


@dataclass(frozen=True)
class WorkerResult:
    evaluated: int
    skipped: int


def _parse_gpus(arg: str) -> list[int]:
    s = arg.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _get_git_sha_short(root: Path) -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        return sha.decode("utf-8").strip()
    except Exception:
        return None


def _worker_main(
    *,
    gpu_id: int | None,
    seq_names: list[str],
    pkl_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    resize_hw: tuple[int, int],
    interp_hw: tuple[int, int],
    resume: bool,
    overwrite: bool,
    queue: Any,
) -> None:
    import torch

    if gpu_id is not None and not torch.cuda.is_available():
        # Keep the script runnable on CPU-only machines even when the user left
        # the default `--gpus 0`.
        print(f"[WARN] CUDA not available; falling back to CPU for gpu_id={gpu_id}", file=sys.stderr)
        gpu_id = None

    device = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    method = ReTrackerMethod(ckpt_path=ckpt_path, interp_shape=interp_hw)
    method.load(device)

    seqs = load_tapvid_davis_first(pkl_path, resize_hw=resize_hw, video_names=seq_names)

    evaluated = 0
    skipped = 0

    for seq in seqs:
        safe = _safe_name(seq.name)
        pred_path = out_dir / "predictions" / f"pred_{safe}.npz"
        metrics_path = out_dir / "metrics" / f"metrics_{safe}.json"

        if resume and not overwrite and pred_path.exists() and metrics_path.exists():
            skipped += 1
            continue

        _ = evaluate_one_sequence(
            seq,
            method=method,
            pred_path=pred_path,
            metrics_path=metrics_path,
            resume=resume,
            overwrite=overwrite,
        )
        evaluated += 1

    queue.put(asdict(WorkerResult(evaluated=evaluated, skipped=skipped)))


def main() -> None:
    root = _repo_root()

    p = argparse.ArgumentParser(description="Evaluate ReTracker on TAP-Vid DAVIS (first protocol) with resume + multi-GPU.")
    p.add_argument("--ckpt", type=str, default=None, help="ReTracker checkpoint path")
    p.add_argument("--dataset-root", type=str, default="", help="Dataset root containing tapvid_davis/tapvid_davis.pkl")
    p.add_argument("--out-dir", type=str, default=str(root / "outputs/eval_benchmark"), help="Output directory")

    p.add_argument("--resize-h", type=int, default=256)
    p.add_argument("--resize-w", type=int, default=256)
    p.add_argument("--interp-h", type=int, default=512)
    p.add_argument("--interp-w", type=int, default=512)

    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2. Empty = CPU")
    p.add_argument("--max-videos", type=int, default=0, help="Limit number of DAVIS videos (0 = all)")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if cached predictions exist")

    args = p.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if dataset_root is None:
        # Best-effort default for open-source checkouts.
        candidates = [
            root / "data" / "tapvid_local",
            root / "data" / "tapvid",
        ]
        dataset_root = next((c for c in candidates if c.exists()), None)

    if dataset_root is None:
        raise ValueError("--dataset-root is required (could not infer a default)")

    pkl_path = resolve_existing_pickle_path(dataset_root / "tapvid_davis" / "tapvid_davis.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"TAP-Vid DAVIS pickle not found: {pkl_path}")

    resize_hw = (int(args.resize_h), int(args.resize_w))
    interp_hw = (int(args.interp_h), int(args.interp_w))

    # Load once to get the ordered list of sequence names, but do not decode or resize
    # any videos here (that work happens inside worker processes).
    raw = load_tapvid_davis_pickle(pkl_path)
    seq_names = sorted(raw.keys())
    # Free the large pickle payload ASAP; workers will reload as needed.
    del raw
    if int(args.max_videos) > 0:
        seq_names = seq_names[: int(args.max_videos)]

    gpus = _parse_gpus(args.gpus)
    if not gpus:
        gpu_ids: list[int | None] = [None]
    else:
        gpu_ids = [int(x) for x in gpus]

    run_root = Path(args.out_dir).expanduser().resolve()
    out_dir = run_root / "tapvid_davis_first" / "retracker"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "tapvid_davis_first",
        "pkl_path": str(pkl_path),
        "ckpt": str(ckpt_path),
        "resize_hw": list(resize_hw),
        "interp_hw": list(interp_hw),
        "gpus": [None if x is None else int(x) for x in gpu_ids],
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "git_sha": _get_git_sha_short(root),
        "cmd": " ".join(sys.argv),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out_dir / "manifest.json", manifest)

    start_wall = time.time()

    # Round-robin shard assignment for better balancing.
    shards: list[list[str]] = [[] for _ in range(len(gpu_ids))]
    for i, name in enumerate(seq_names):
        shards[i % len(gpu_ids)].append(name)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()

    procs = []
    for gpu_id, names in zip(gpu_ids, shards):
        if not names:
            continue
        p_ = ctx.Process(
            target=_worker_main,
            kwargs={
                "gpu_id": gpu_id,
                "seq_names": names,
                "pkl_path": pkl_path,
                "ckpt_path": ckpt_path,
                "out_dir": out_dir,
                "resize_hw": resize_hw,
                "interp_hw": interp_hw,
                "resume": bool(args.resume),
                "overwrite": bool(args.overwrite),
                "queue": queue,
            },
        )
        p_.start()
        procs.append(p_)

    evaluated = 0
    skipped = 0
    remaining = len(procs)
    while remaining > 0:
        try:
            msg = queue.get(timeout=5.0)
        except Exception:
            # Avoid hanging forever if a worker crashes before posting results.
            for p_ in procs:
                if p_.exitcode is not None and p_.exitcode != 0:
                    raise RuntimeError(f"Worker process {p_.pid} crashed with exit code {p_.exitcode}")
            continue
        evaluated += int(msg.get("evaluated", 0))
        skipped += int(msg.get("skipped", 0))
        remaining -= 1

    for p_ in procs:
        p_.join()

    wallclock_sec = float(time.time() - start_wall)

    # Aggregate metrics from per-video JSONs.
    per_video: dict[str, dict[str, float]] = {}
    runtime_per_video: dict[str, float | None] = {}
    sum_runtime_sec = 0.0
    metrics_dir = out_dir / "metrics"
    for name in seq_names:
        safe = _safe_name(name)
        mpth = metrics_dir / f"metrics_{safe}.json"
        if not mpth.exists():
            raise FileNotFoundError(f"Missing metrics file (worker failed?): {mpth}")
        data = json.loads(mpth.read_text(encoding="utf-8"))
        per_video[name] = {k: float(v) for k, v in data["metrics"].items()}
        rt = data.get("runtime_sec", None)
        runtime_per_video[name] = None if rt is None else float(rt)
        if rt is not None:
            sum_runtime_sec += float(rt)

    summary = aggregate_metrics(per_video)
    summary["evaluated"] = evaluated
    summary["skipped"] = skipped
    summary["total"] = len(seq_names)
    summary["wallclock_sec"] = wallclock_sec
    summary["sum_runtime_sec"] = float(sum_runtime_sec)
    summary["runtime_per_video_sec"] = runtime_per_video

    result_path = out_dir / "result_eval_tapvid_davis_first.json"
    _write_json(result_path, summary)

    print("[Done] Wrote:")
    print(f"  {result_path}")


if __name__ == "__main__":
    main()

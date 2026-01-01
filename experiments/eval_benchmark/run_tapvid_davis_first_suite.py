#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Support running as a script: `python experiments/eval_benchmark/run_tapvid_davis_first_suite.py ...`
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from experiments.benchmark.core.types import TrackingResult  # noqa: E402
from experiments.benchmark.trackers import (  # noqa: E402
    BenchmarkJob,
    CoTracker3OfflineTracker,
    CoTracker3OnlineTracker,
    ReTrackerTracker,
    TapirTracker,
    TapNextTracker,
    TrackOn2Tracker,
)
from experiments.benchmark.trackers.cotracker3 import CoTracker3Config, CoTracker3OnlineConfig  # noqa: E402
from experiments.benchmark.trackers.external_runner import ExternalRunnerSpec  # noqa: E402
from experiments.benchmark.trackers.retracker import ReTrackerConfig  # noqa: E402
from experiments.benchmark.trackers.tapir import TapirConfig  # noqa: E402
from experiments.benchmark.trackers.tapnext import TapNextConfig  # noqa: E402
from experiments.benchmark.trackers.trackon2 import TrackOn2Config  # noqa: E402
from experiments.eval_benchmark.core.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_tapvid_davis_first_metrics_from_trackingresult,
)
from experiments.eval_benchmark.datasets.tapvid_davis import resolve_existing_pickle_path  # noqa: E402
from experiments.eval_benchmark.suite.cache import build_tapvid_davis_first_cache  # noqa: E402


def _safe_name(name: str) -> str:
    # Keep filesystem layout stable across datasets.
    return name.replace(os.sep, "_")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_worker_result(path: Path, result: WorkerResult) -> None:
    """
    Persist small worker stats to avoid multiprocessing.Queue/Lock, which can
    crash in restricted environments (no POSIX semaphores).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True)
    tmp.replace(path)


def _parse_csv_ints(s: str) -> list[int]:
    raw = s.strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _shard_round_robin(names: list[str], num_shards: int) -> list[list[str]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for i, n in enumerate(names):
        shards[i % num_shards].append(n)
    return shards


def _should_skip(pred_path: Path, metrics_path: Path, *, resume: bool, overwrite: bool) -> bool:
    if overwrite:
        return False
    if not resume:
        return False
    return pred_path.exists() and metrics_path.exists()


@dataclass(frozen=True)
class SuiteConfig:
    # Dataset/cache
    cache_dir: Path
    resize_hw: tuple[int, int]
    max_videos: int | None

    # Method selection
    methods: list[str]

    # Paths/envs for method runners
    third_party_root: Path
    ckpt_dir: Path
    retracker_ckpt: Path
    dtype: str
    trackon_ckpt: Path
    trackon_config: Path
    cotracker_offline_ckpt: Path
    cotracker_online_ckpt: Path
    cotracker_window_len: int
    tapir_ckpt: Path
    tapir_infer_hw: tuple[int, int]
    tapnext_ckpt: Path
    tapnext_infer_hw: tuple[int, int]

    retracker_env: str
    trackon_env: str
    tapir_env: str
    tapnext_env: str

    # Execution flags
    resume: bool
    overwrite: bool


@dataclass(frozen=True)
class WorkerResult:
    evaluated: int
    skipped: int


def _make_tracker(method: str, *, cfg: SuiteConfig, repo_root: Path, ld_preload: str | None) -> Any:
    runners = (repo_root / "experiments/benchmark/runners").resolve()

    if method == "retracker":
        return ReTrackerTracker(
            spec=ExternalRunnerSpec(env_name=cfg.retracker_env, runner_script=runners / "retracker_runner.py"),
            cfg=ReTrackerConfig(ckpt=cfg.retracker_ckpt, dtype=cfg.dtype),
            repo_root=repo_root,
        )

    if method == "trackon2":
        return TrackOn2Tracker(
            spec=ExternalRunnerSpec(env_name=cfg.trackon_env, runner_script=runners / "trackon2_runner.py", ld_preload=ld_preload),
            cfg=TrackOn2Config(
                trackon_root=cfg.third_party_root / "track_on",
                ckpt=cfg.trackon_ckpt,
                config=cfg.trackon_config,
                dataset_name="tapvid_davis_first",
            ),
            repo_root=repo_root,
        )

    if method == "cotracker3_offline":
        return CoTracker3OfflineTracker(
            spec=ExternalRunnerSpec(
                env_name=cfg.trackon_env, runner_script=runners / "cotracker3_offline_runner.py", ld_preload=ld_preload
            ),
            cfg=CoTracker3Config(cotracker_root=cfg.third_party_root / "co-tracker", ckpt=cfg.cotracker_offline_ckpt),
            repo_root=repo_root,
        )

    if method == "cotracker3_online":
        return CoTracker3OnlineTracker(
            spec=ExternalRunnerSpec(
                env_name=cfg.trackon_env, runner_script=runners / "cotracker3_online_runner.py", ld_preload=ld_preload
            ),
            cfg=CoTracker3OnlineConfig(
                cotracker_root=cfg.third_party_root / "co-tracker",
                ckpt=cfg.cotracker_online_ckpt,
                window_len=int(cfg.cotracker_window_len),
            ),
            repo_root=repo_root,
        )

    if method == "tapir":
        return TapirTracker(
            spec=ExternalRunnerSpec(env_name=cfg.tapir_env, runner_script=runners / "tapir_runner.py", ld_preload=ld_preload),
            cfg=TapirConfig(tapnet_root=cfg.third_party_root / "tapnet", ckpt=cfg.tapir_ckpt, infer_hw=cfg.tapir_infer_hw),
            repo_root=repo_root,
        )

    if method == "tapnext":
        return TapNextTracker(
            spec=ExternalRunnerSpec(env_name=cfg.tapnext_env, runner_script=runners / "tapnext_runner.py"),
            cfg=TapNextConfig(ckpt=cfg.tapnext_ckpt, infer_hw=cfg.tapnext_infer_hw),
            repo_root=repo_root,
        )

    raise ValueError(f"Unknown method: {method}")


def _load_gt_from_seq_npz(seq_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(seq_npz, allow_pickle=False) as z:
        q_tyx = np.asarray(z["query_points_tyx"], dtype=np.float32)
        gt_tracks_xy = np.asarray(z["gt_tracks_xy"], dtype=np.float32)
        gt_occluded = np.asarray(z["gt_occluded"]).astype(bool, copy=False)
    return q_tyx, gt_tracks_xy, gt_occluded


def _worker_main(
    *,
    method: str,
    gpu_id: int | None,
    seq_names: list[str],
    suite_cfg: SuiteConfig,
    out_dir_method: Path,
    worker_result_path: Path,
) -> None:
    # GPU isolation for subprocesses: propagate into conda runner env via os.environ.
    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

    repo_root = _repo_root()
    jit_shim = (repo_root / "experiments/benchmark/scripts/libjitprofiling_stub.so").resolve()
    ld_preload = str(jit_shim) if jit_shim.exists() else None

    tracker = _make_tracker(method, cfg=suite_cfg, repo_root=repo_root, ld_preload=ld_preload)

    evaluated = 0
    skipped = 0

    for seq_name in seq_names:
        safe = _safe_name(seq_name)
        seq_npz = suite_cfg.cache_dir / "sequences" / f"{seq_name}.npz"
        queries_txt = suite_cfg.cache_dir / "queries" / f"{seq_name}.txt"

        pred_path = out_dir_method / "predictions" / f"result_{safe}.npz"
        metrics_path = out_dir_method / "metrics" / f"metrics_{safe}.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if _should_skip(pred_path, metrics_path, resume=suite_cfg.resume, overwrite=suite_cfg.overwrite):
            skipped += 1
            continue

        # Inference (or resume prediction only).
        if pred_path.exists() and suite_cfg.resume and not suite_cfg.overwrite:
            pred = TrackingResult.load_npz(pred_path)
        else:
            job = BenchmarkJob(video=seq_npz, queries=queries_txt, resized_hw=suite_cfg.resize_hw, start=0, max_frames=None)

            # Tracker interface wants an output directory; it always writes `result.npz` inside.
            tmp_dir = out_dir_method / "predictions" / f"_tmp_{safe}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pred = tracker.run(job, tmp_dir)
            pred.save_npz(pred_path)
            evaluated += 1

        # Metrics
        q_tyx, gt_tracks_xy, gt_occluded = _load_gt_from_seq_npz(seq_npz)
        metrics = compute_tapvid_davis_first_metrics_from_trackingresult(
            pred=pred,
            query_points_tyx=q_tyx,
            gt_tracks_xy=gt_tracks_xy,
            gt_occluded=gt_occluded,
        )

        payload: dict[str, Any] = {
            "seq_name": seq_name,
            "method": method,
            "runtime_sec": pred.runtime_sec,
            "metrics": metrics,
        }
        _write_json(metrics_path, payload)

    _write_worker_result(worker_result_path, WorkerResult(evaluated=evaluated, skipped=skipped))


def _load_per_video_metrics(out_dir_method: Path, seq_names: list[str]) -> tuple[dict[str, dict[str, float]], float]:
    per_video: dict[str, dict[str, float]] = {}
    sum_runtime = 0.0
    for seq_name in seq_names:
        safe = _safe_name(seq_name)
        mpth = out_dir_method / "metrics" / f"metrics_{safe}.json"
        data = json.loads(mpth.read_text(encoding="utf-8"))
        per_video[seq_name] = {k: float(v) for k, v in data["metrics"].items()}
        rt = data.get("runtime_sec", None)
        if rt is not None:
            sum_runtime += float(rt)
    return per_video, float(sum_runtime)


def main() -> None:
    repo_root = _repo_root()

    p = argparse.ArgumentParser(description="Multi-method TAP-Vid DAVIS (first) evaluation suite (subprocess conda envs).")
    p.add_argument("--dataset-root", type=str, required=True, help="Dataset root containing tapvid_davis/tapvid_davis.pkl")
    p.add_argument("--out-dir", type=str, default=str(repo_root / "outputs/eval_benchmark"), help="Output directory")

    p.add_argument("--methods", type=str, default="retracker", help="Comma-separated methods to run")
    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids. Empty = CPU")
    p.add_argument("--max-videos", type=int, default=0, help="Limit DAVIS videos (0 = all)")
    p.add_argument("--resize-h", type=int, default=256)
    p.add_argument("--resize-w", type=int, default=256)

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if cached predictions exist")

    # Env names
    p.add_argument("--retracker-env", type=str, default="retracker_env")
    p.add_argument("--trackon-env", type=str, default="trackon2")
    p.add_argument("--tapir-env", type=str, default="tapnext", help="Conda env for TAPIR (PyTorch)")
    p.add_argument("--tapnext-env", type=str, default="tapnet", help="Conda env for TapNext (JAX/Flax)")

    # Paths (third_party + checkpoints)
    p.add_argument("--third-party-root", type=str, default=str(repo_root / "experiments/third_party"))
    p.add_argument("--ckpt-dir", type=str, default=str(repo_root / "experiments/benchmark/checkpoints"))

    p.add_argument("--retracker-ckpt", type=str, default=None, help="ReTracker checkpoint path")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    p.add_argument("--trackon-ckpt", type=str, default="", help="Track-On2 checkpoint (.pt)")
    p.add_argument("--trackon-config", type=str, default="", help="Track-On2 config yaml")

    p.add_argument("--cotracker-offline-ckpt", type=str, default="", help="CoTracker3 offline checkpoint (.pth)")
    p.add_argument("--cotracker-online-ckpt", type=str, default="", help="CoTracker3 online checkpoint (.pth)")
    p.add_argument("--cotracker-window-len", type=int, default=16)

    p.add_argument("--tapir-ckpt", type=str, default="", help="TAPIR checkpoint (.pt)")
    p.add_argument("--tapnext-ckpt", type=str, default="", help="TapNext checkpoint (.npz)")
    p.add_argument("--infer-w", type=int, default=256)
    p.add_argument("--infer-h", type=int, default=256)

    args = p.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pkl_path = resolve_existing_pickle_path(dataset_root / "tapvid_davis" / "tapvid_davis.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"TAP-Vid DAVIS pickle not found: {pkl_path}")

    out_root = Path(args.out_dir).expanduser().resolve() / "tapvid_davis_first"
    out_root.mkdir(parents=True, exist_ok=True)

    resize_hw = (int(args.resize_h), int(args.resize_w))
    max_videos = int(args.max_videos) if int(args.max_videos) > 0 else None

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods must be non-empty")

    # Cache is shared across methods.
    cache_dir = out_root / "cache"
    seq_names = build_tapvid_davis_first_cache(pkl_path=pkl_path, out_dir=cache_dir, resize_hw=resize_hw, max_videos=max_videos)

    third_party_root = Path(args.third_party_root).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()

    suite_cfg = SuiteConfig(
        cache_dir=cache_dir,
        resize_hw=resize_hw,
        max_videos=max_videos,
        methods=methods,
        third_party_root=third_party_root,
        ckpt_dir=ckpt_dir,
        retracker_ckpt=Path(args.retracker_ckpt).expanduser().resolve(),
        dtype=str(args.dtype),
        trackon_ckpt=Path(args.trackon_ckpt).expanduser().resolve() if args.trackon_ckpt else (ckpt_dir / "trackon2_dinov2_checkpoint.pt").resolve(),
        trackon_config=Path(args.trackon_config).expanduser().resolve()
        if args.trackon_config
        else (third_party_root / "track_on" / "config" / "test_dinov2.yaml").resolve(),
        cotracker_offline_ckpt=Path(args.cotracker_offline_ckpt).expanduser().resolve()
        if args.cotracker_offline_ckpt
        else (ckpt_dir / "scaled_offline.pth").resolve(),
        cotracker_online_ckpt=Path(args.cotracker_online_ckpt).expanduser().resolve()
        if args.cotracker_online_ckpt
        else (ckpt_dir / "scaled_online.pth").resolve(),
        cotracker_window_len=int(args.cotracker_window_len),
        tapir_ckpt=Path(args.tapir_ckpt).expanduser().resolve()
        if args.tapir_ckpt
        else (ckpt_dir / "causal_bootstapir_checkpoint.pt").resolve(),
        tapir_infer_hw=(int(args.infer_h), int(args.infer_w)),
        tapnext_ckpt=Path(args.tapnext_ckpt).expanduser().resolve()
        if args.tapnext_ckpt
        else (ckpt_dir / "bootstapnext_ckpt.npz").resolve(),
        tapnext_infer_hw=(256, 256),
        retracker_env=str(args.retracker_env),
        trackon_env=str(args.trackon_env),
        tapir_env=str(args.tapir_env),
        tapnext_env=str(args.tapnext_env),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )

    manifest = {
        "dataset": "tapvid_davis_first",
        "pkl_path": str(pkl_path),
        "cache_dir": str(cache_dir),
        "resize_hw": list(resize_hw),
        "methods": methods,
        "cmd": " ".join(sys.argv),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
    }
    _write_json(out_root / "manifest.json", manifest)

    gpus = _parse_csv_ints(str(args.gpus))
    gpu_ids: list[int | None] = [None] if not gpus else [int(x) for x in gpus]

    compare: dict[str, Any] = {"methods": {}, "dataset": "tapvid_davis_first"}

    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    for method in methods:
        out_dir_method = out_root / method
        out_dir_method.mkdir(parents=True, exist_ok=True)

        start_wall = time.time()

        shards = _shard_round_robin(seq_names, len(gpu_ids))
        worker_results_dir = out_dir_method / "worker_results"
        worker_results_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        worker_paths: list[Path] = []
        for worker_idx, (gpu_id, names) in enumerate(zip(gpu_ids, shards)):
            if not names:
                continue
            worker_result_path = worker_results_dir / f"worker_{worker_idx}.json"
            worker_paths.append(worker_result_path)
            p_ = ctx.Process(
                target=_worker_main,
                kwargs={
                    "method": method,
                    "gpu_id": gpu_id,
                    "seq_names": names,
                    "suite_cfg": suite_cfg,
                    "out_dir_method": out_dir_method,
                    "worker_result_path": worker_result_path,
                },
            )
            p_.start()
            procs.append(p_)

        evaluated = 0
        skipped = 0
        for p_ in procs:
            p_.join()
            if p_.exitcode is not None and p_.exitcode != 0:
                raise RuntimeError(f"Worker process {p_.pid} crashed with exit code {p_.exitcode}")

        for pth in worker_paths:
            if not pth.exists():
                raise FileNotFoundError(f"Missing worker result file: {pth}")
            data = json.loads(pth.read_text(encoding="utf-8"))
            evaluated += int(data.get("evaluated", 0))
            skipped += int(data.get("skipped", 0))

        wallclock_sec = float(time.time() - start_wall)

        per_video, sum_runtime_sec = _load_per_video_metrics(out_dir_method, seq_names)
        summary = aggregate_metrics(per_video)
        summary["evaluated"] = evaluated
        summary["skipped"] = skipped
        summary["total"] = len(seq_names)
        summary["wallclock_sec"] = wallclock_sec
        summary["sum_runtime_sec"] = float(sum_runtime_sec)

        result_path = out_dir_method / "result_eval_tapvid_davis_first.json"
        _write_json(result_path, summary)
        compare["methods"][method] = {"avg": summary.get("avg", {}), "result_path": str(result_path)}

        print(f"[Done] method={method} -> {result_path}")

    _write_json(out_root / "compare_methods.json", compare)
    print(f"[Done] compare -> {out_root / 'compare_methods.json'}")


if __name__ == "__main__":
    main()

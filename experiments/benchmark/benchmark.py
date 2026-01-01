#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Support running as a script: `python experiments/benchmark/benchmark.py ...`
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from experiments.benchmark.core.mosaic import MosaicInput, compose_mosaic_mp4  # noqa: E402
from experiments.benchmark.core.types import Queries, TrackingResult  # noqa: E402
from experiments.benchmark.core.video import load_video_rgb, save_video_rgb  # noqa: E402
from experiments.benchmark.core.vis import VisConfig, render_overlay_rgb  # noqa: E402
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


@dataclass(frozen=True)
class Pair:
    video: Path
    queries: Path


def _parse_pairs_file(path: str) -> list[Pair]:
    pairs: list[Pair] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in pairs file (expected 2 columns): {raw.rstrip()}")
            pairs.append(Pair(video=Path(parts[0]).expanduser(), queries=Path(parts[1]).expanduser()))
    if not pairs:
        raise ValueError(f"No pairs found in: {path}")
    return pairs


def _ensure_file(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def main() -> None:
    root = _repo_root()

    p = argparse.ArgumentParser(description="Run multi-method tracking benchmark and compose a mosaic video.")
    in_g = p.add_mutually_exclusive_group(required=True)
    in_g.add_argument("--pair", action="append", nargs=2, metavar=("VIDEO", "QUERIES"), help="(video, queries) pair")
    in_g.add_argument("--pairs-file", type=str, help="Text file with lines: <video_path> <queries_path>")

    p.add_argument("--out-dir", type=str, default=str(root / "outputs/benchmark"), help="Output directory root")
    p.add_argument("--resized-w", type=int, default=512)
    p.add_argument("--resized-h", type=int, default=384)
    p.add_argument("--start", type=int, default=0, help="Start frame (for smoke tests / quick runs)")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = no limit)")
    p.add_argument("--fps", type=int, default=10, help="FPS for output videos")
    p.add_argument("--point-size", type=int, default=100, help="Matplotlib-like scatter size")
    p.add_argument("--cols", type=int, default=3, help="Columns for mosaic (default: 3)")

    # Conda env names
    p.add_argument("--retracker-env", type=str, default="retracker_env")
    p.add_argument("--trackon-env", type=str, default="trackon2")
    p.add_argument("--tapir-env", type=str, default="tapnext", help="Conda env for TAPIR (PyTorch)")
    p.add_argument("--tapnext-env", type=str, default="tapnet", help="Conda env for TapNext (JAX/Flax)")

    # Paths (third_party + checkpoints)
    p.add_argument("--third-party-root", type=str, default=str(root / "experiments/third_party"))
    p.add_argument("--ckpt-dir", type=str, default=str(root / "experiments/benchmark/checkpoints"))

    p.add_argument("--retracker-ckpt", type=str, default=None, help="Path to ReTracker checkpoint (required)")
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

    if args.pairs_file:
        pairs = _parse_pairs_file(args.pairs_file)
    else:
        assert args.pair is not None
        pairs = [Pair(video=Path(v), queries=Path(q)) for (v, q) in args.pair]

    third_party_root = Path(args.third_party_root).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()

    trackon_root = third_party_root / "track_on"
    cotracker_root = third_party_root / "co-tracker"
    tapnet_root = third_party_root / "tapnet"

    trackon_ckpt = Path(args.trackon_ckpt).resolve() if args.trackon_ckpt else (ckpt_dir / "trackon2_dinov2_checkpoint.pt")
    trackon_config = (
        Path(args.trackon_config).resolve() if args.trackon_config else (trackon_root / "config/test_dinov2.yaml").resolve()
    )
    cot_off_ckpt = (
        Path(args.cotracker_offline_ckpt).resolve()
        if args.cotracker_offline_ckpt
        else (ckpt_dir / "scaled_offline.pth").resolve()
    )
    cot_on_ckpt = (
        Path(args.cotracker_online_ckpt).resolve()
        if args.cotracker_online_ckpt
        else (ckpt_dir / "scaled_online.pth").resolve()
    )
    tapir_ckpt = Path(args.tapir_ckpt).resolve() if args.tapir_ckpt else (ckpt_dir / "causal_bootstapir_checkpoint.pt")
    tapnext_ckpt = (
        Path(args.tapnext_ckpt).resolve() if args.tapnext_ckpt else (ckpt_dir / "bootstapnext_ckpt.npz").resolve()
    )

    # Torch iJIT shim for some conda builds.
    jit_shim = (root / "experiments/benchmark/scripts/libjitprofiling_stub.so").resolve()
    ld_preload = str(jit_shim) if jit_shim.exists() else None

    if ld_preload is None:
        print(
            "[Warn] LD_PRELOAD shim not found. If Track-On2/CoTracker/TAPIR crashes with iJIT symbols, run:\n"
            "  bash experiments/benchmark/scripts/build_jitprofiling_stub.sh",
            file=sys.stderr,
        )

    # Runner scripts (live in this repo).
    runners = (root / "experiments/benchmark/runners").resolve()
    retracker_runner = runners / "retracker_runner.py"
    trackon_runner = runners / "trackon2_runner.py"
    cot_off_runner = runners / "cotracker3_offline_runner.py"
    cot_on_runner = runners / "cotracker3_online_runner.py"
    tapir_runner = runners / "tapir_runner.py"
    tapnext_runner = runners / "tapnext_runner.py"

    for rp in [retracker_runner, trackon_runner, cot_off_runner, cot_on_runner, tapir_runner, tapnext_runner]:
        _ensure_file(rp, "Runner script")

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    trackers = [
        ReTrackerTracker(
            spec=ExternalRunnerSpec(env_name=args.retracker_env, runner_script=retracker_runner),
            cfg=ReTrackerConfig(ckpt=Path(args.retracker_ckpt).resolve(), dtype=args.dtype),
            repo_root=root,
        ),
        TrackOn2Tracker(
            spec=ExternalRunnerSpec(env_name=args.trackon_env, runner_script=trackon_runner, ld_preload=ld_preload),
            cfg=TrackOn2Config(trackon_root=trackon_root, ckpt=trackon_ckpt, config=trackon_config),
            repo_root=root,
        ),
        CoTracker3OfflineTracker(
            spec=ExternalRunnerSpec(env_name=args.trackon_env, runner_script=cot_off_runner, ld_preload=ld_preload),
            cfg=CoTracker3Config(cotracker_root=cotracker_root, ckpt=cot_off_ckpt),
            repo_root=root,
        ),
        CoTracker3OnlineTracker(
            spec=ExternalRunnerSpec(env_name=args.trackon_env, runner_script=cot_on_runner, ld_preload=ld_preload),
            cfg=CoTracker3OnlineConfig(cotracker_root=cotracker_root, ckpt=cot_on_ckpt, window_len=args.cotracker_window_len),
            repo_root=root,
        ),
        TapirTracker(
            spec=ExternalRunnerSpec(env_name=args.tapir_env, runner_script=tapir_runner, ld_preload=ld_preload),
            cfg=TapirConfig(tapnet_root=tapnet_root, ckpt=tapir_ckpt, infer_hw=(args.infer_h, args.infer_w)),
            repo_root=root,
        ),
        TapNextTracker(
            spec=ExternalRunnerSpec(env_name=args.tapnext_env, runner_script=tapnext_runner),
            cfg=TapNextConfig(ckpt=tapnext_ckpt),
            repo_root=root,
        ),
    ]

    label_map = {
        "retracker": "ReTracker",
        "trackon2": "Track-On2",
        "cotracker3_offline": "CoTracker3 (offline)",
        "cotracker3_online": "CoTracker3 (online)",
        "tapir": "TAPIR (PyTorch)",
        "tapnext": "TapNext (JAX)",
    }

    for pair in pairs:
        video = pair.video.resolve()
        queries_path = pair.queries.resolve()
        _ensure_file(video, "Video")
        _ensure_file(queries_path, "Queries file")

        # Load queries once (for visualization colors / sanity checks).
        queries = Queries.load_txt(queries_path)

        stem = video.stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f"[Info] === Benchmark: {stem} ===")
        print(f"[Info] video:   {video}")
        print(f"[Info] queries: {queries_path} (N={queries.n})")
        print(f"[Info] out:     {out_dir}")

        job = BenchmarkJob(
            video=video,
            queries=queries_path,
            resized_hw=(int(args.resized_h), int(args.resized_w)),
            start=int(args.start),
            max_frames=max_frames,
        )

        # Load the same (sub)video for visualization.
        video_data = load_video_rgb(video, resized_hw=job.resized_hw, start=job.start, max_frames=job.max_frames)
        fps_out = float(args.fps)

        mosaic_inputs: list[MosaicInput] = []

        for tracker in trackers:
            method_dir = out_dir / tracker.name
            method_dir.mkdir(parents=True, exist_ok=True)

            # Run tracker and save standardized result.
            result: TrackingResult = tracker.run(job, method_dir)
            result.save_npz(method_dir / "result.npz")

            # Render overlay + save mp4.
            label = label_map.get(tracker.name, tracker.name)
            overlay = render_overlay_rgb(video_data.frames_rgb, result, VisConfig(point_size=args.point_size, label=label))
            overlay_mp4 = method_dir / "overlay.mp4"
            save_video_rgb(overlay, overlay_mp4, fps=fps_out)
            mosaic_inputs.append(MosaicInput(label=label, video_path=overlay_mp4))

            rt = f"{result.runtime_sec:.3f}s" if result.runtime_sec is not None else "n/a"
            print(f"[Info] {tracker.name:18s} runtime={rt} -> {overlay_mp4}")

        mosaic_mp4 = out_dir / f"{stem}_benchmark.mp4"
        compose_mosaic_mp4(mosaic_inputs, out_mp4=mosaic_mp4, cols=int(args.cols), fps=fps_out)
        print(f"[Done] Mosaic: {mosaic_mp4}")


if __name__ == "__main__":
    main()

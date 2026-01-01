from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark.core.types import TrackingResult
from experiments.benchmark.trackers.base import BenchmarkJob, Tracker
from experiments.benchmark.trackers.external_runner import ExternalRunnerMixin, ExternalRunnerSpec


@dataclass(frozen=True)
class TapirConfig:
    tapnet_root: Path
    ckpt: Path
    infer_hw: tuple[int, int] = (256, 256)  # (H, W)


class TapirTracker(ExternalRunnerMixin, Tracker):
    def __init__(self, *, spec: ExternalRunnerSpec, cfg: TapirConfig, repo_root: Path) -> None:
        self.spec = spec
        self.cfg = cfg
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "tapir"

    def run(self, job: BenchmarkJob, out_dir: Path) -> TrackingResult:
        out_npz = out_dir / "result.npz"
        argv = [
            "python",
            str(self.spec.runner_script),
            "--video",
            str(job.video),
            "--queries",
            str(job.queries),
            "--out-npz",
            str(out_npz),
            "--resized-h",
            str(job.resized_hw[0]),
            "--resized-w",
            str(job.resized_hw[1]),
            "--infer-h",
            str(self.cfg.infer_hw[0]),
            "--infer-w",
            str(self.cfg.infer_hw[1]),
            "--start",
            str(job.start),
            "--tapnet-root",
            str(self.cfg.tapnet_root),
            "--ckpt",
            str(self.cfg.ckpt),
        ]
        if job.max_frames is not None:
            argv += ["--max-frames", str(job.max_frames)]
        return self._run_runner(argv=argv, cwd=self.repo_root, out_npz=out_npz)

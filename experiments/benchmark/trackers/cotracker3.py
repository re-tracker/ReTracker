from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark.core.types import TrackingResult
from experiments.benchmark.trackers.base import BenchmarkJob, Tracker
from experiments.benchmark.trackers.external_runner import ExternalRunnerMixin, ExternalRunnerSpec


@dataclass(frozen=True)
class CoTracker3Config:
    cotracker_root: Path
    ckpt: Path
    interp_hw: tuple[int, int] = (384, 512)


@dataclass(frozen=True)
class CoTracker3OnlineConfig(CoTracker3Config):
    window_len: int = 16


class CoTracker3OfflineTracker(ExternalRunnerMixin, Tracker):
    def __init__(self, *, spec: ExternalRunnerSpec, cfg: CoTracker3Config, repo_root: Path) -> None:
        self.spec = spec
        self.cfg = cfg
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "cotracker3_offline"

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
            "--interp-h",
            str(int(self.cfg.interp_hw[0])),
            "--interp-w",
            str(int(self.cfg.interp_hw[1])),
            "--start",
            str(job.start),
            "--cotracker-root",
            str(self.cfg.cotracker_root),
            "--ckpt",
            str(self.cfg.ckpt),
        ]
        if job.max_frames is not None:
            argv += ["--max-frames", str(job.max_frames)]
        return self._run_runner(argv=argv, cwd=self.repo_root, out_npz=out_npz)


class CoTracker3OnlineTracker(ExternalRunnerMixin, Tracker):
    def __init__(self, *, spec: ExternalRunnerSpec, cfg: CoTracker3OnlineConfig, repo_root: Path) -> None:
        self.spec = spec
        self.cfg = cfg
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "cotracker3_online"

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
            "--interp-h",
            str(int(self.cfg.interp_hw[0])),
            "--interp-w",
            str(int(self.cfg.interp_hw[1])),
            "--start",
            str(job.start),
            "--cotracker-root",
            str(self.cfg.cotracker_root),
            "--ckpt",
            str(self.cfg.ckpt),
            "--window-len",
            str(int(self.cfg.window_len)),
        ]
        if job.max_frames is not None:
            argv += ["--max-frames", str(job.max_frames)]
        return self._run_runner(argv=argv, cwd=self.repo_root, out_npz=out_npz)

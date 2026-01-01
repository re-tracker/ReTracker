from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark.core.types import TrackingResult
from experiments.benchmark.trackers.base import BenchmarkJob, Tracker
from experiments.benchmark.trackers.external_runner import ExternalRunnerMixin, ExternalRunnerSpec


@dataclass(frozen=True)
class TrackOn2Config:
    trackon_root: Path
    ckpt: Path
    config: Path
    support_grid_size: int = 20
    dataset_name: str | None = None


class TrackOn2Tracker(ExternalRunnerMixin, Tracker):
    def __init__(self, *, spec: ExternalRunnerSpec, cfg: TrackOn2Config, repo_root: Path) -> None:
        self.spec = spec
        self.cfg = cfg
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "trackon2"

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
            "--start",
            str(job.start),
            "--trackon-root",
            str(self.cfg.trackon_root),
            "--config",
            str(self.cfg.config),
            "--ckpt",
            str(self.cfg.ckpt),
            "--support-grid-size",
            str(int(self.cfg.support_grid_size)),
        ]
        if self.cfg.dataset_name:
            argv += ["--dataset-name", str(self.cfg.dataset_name)]
        if job.max_frames is not None:
            argv += ["--max-frames", str(job.max_frames)]
        return self._run_runner(argv=argv, cwd=self.repo_root, out_npz=out_npz)

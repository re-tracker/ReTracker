from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark.core.types import TrackingResult


@dataclass(frozen=True)
class BenchmarkJob:
    video: Path
    queries: Path
    resized_hw: tuple[int, int]  # (H, W)
    start: int = 0
    max_frames: int | None = None


class Tracker(abc.ABC):
    """
    Benchmark-facing tracker interface.

    Implementations may run in-process or via subprocess/conda envs, but must
    produce a standardized `TrackingResult`.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, job: BenchmarkJob, out_dir: Path) -> TrackingResult:
        raise NotImplementedError


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark.core.conda import CondaRunConfig, run_in_conda_env
from experiments.benchmark.core.types import TrackingResult


@dataclass(frozen=True)
class ExternalRunnerSpec:
    env_name: str
    runner_script: Path
    ld_preload: str | None = None


class ExternalRunnerMixin:
    """
    Helper mixin for trackers implemented as per-env runner scripts.
    """

    spec: ExternalRunnerSpec

    def _run_runner(
        self,
        *,
        argv: list[str],
        cwd: Path,
        out_npz: Path,
        extra_env: dict[str, str] | None = None,
    ) -> TrackingResult:
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        run_in_conda_env(
            argv,
            CondaRunConfig(
                env_name=self.spec.env_name,
                cwd=cwd,
                extra_env=extra_env,
                ld_preload=self.spec.ld_preload,
            ),
        )

        if not out_npz.exists():
            raise FileNotFoundError(f"Runner did not produce expected npz: {out_npz}")
        return TrackingResult.load_npz(out_npz)


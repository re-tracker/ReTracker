from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CondaRunConfig:
    env_name: str
    cwd: Path
    extra_env: dict[str, str] | None = None
    ld_preload: str | None = None


def _conda_base_from_path() -> Path:
    conda = shutil.which("conda")
    if not conda:
        raise FileNotFoundError("conda not found in PATH")
    conda_path = Path(conda).resolve()
    base = conda_path.parents[1]
    conda_sh = base / "etc/profile.d/conda.sh"
    if not conda_sh.exists():
        raise FileNotFoundError(f"conda.sh not found at expected location: {conda_sh}")
    return base


def _quote_cmd(argv: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)


def run_in_conda_env(argv: list[str], cfg: CondaRunConfig) -> None:
    """
    Run a command in a conda env using `conda activate` (not `conda run`).

    This is more robust in restricted environments where `conda run` may crash
    due to OpenMP SHM issues.
    """
    conda_base = _conda_base_from_path()
    conda_sh = conda_base / "etc/profile.d/conda.sh"

    bash_lines = [
        "set -euo pipefail",
        # Avoid conda plugins (CUDA detector can crash due to shared-memory perms).
        "export CONDA_NO_PLUGINS=true",
        'export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-}"',
        # Keep runs headless.
        'export MPLBACKEND="${MPLBACKEND:-Agg}"',
        f"source {shlex.quote(str(conda_sh))}",
        f"conda activate {shlex.quote(cfg.env_name)}",
    ]
    if cfg.ld_preload:
        bash_lines.append(
            f'export LD_PRELOAD={shlex.quote(cfg.ld_preload)}${{LD_PRELOAD:+:${{LD_PRELOAD}}}}'
        )
    bash_lines.append(_quote_cmd(argv))

    cmd = ["bash", "-lc", "\n".join(bash_lines)]
    env = dict(os.environ)
    if cfg.extra_env:
        env.update(cfg.extra_env)

    print(f"[Run:{cfg.env_name}] {_quote_cmd(argv)}")
    subprocess.run(cmd, cwd=str(cfg.cwd), env=env, check=True)


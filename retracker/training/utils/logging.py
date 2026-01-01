from __future__ import annotations

from pathlib import Path

from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def configure_logger(dump_dir: str | Path) -> Path | None:
    """Configure best-effort run logging for training.

    Historically this project used loguru to split metrics/loss to multiple sinks.
    The repo now standardizes on Rich's `CONSOLE.print` and mirrors console output
    into a plaintext log file under the run directory.
    """

    dump_dir = Path(dump_dir)
    log_dir = dump_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "retracker.log"

    enabled = enable_file_logging(log_file=log_path)
    if enabled is not None:
        CONSOLE.print(f"[dim]Logging to: {enabled}[/dim]")
    return enabled


# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Additional rich ui components"""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import re
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class _TeeStream:
    """A minimal text stream that tees Rich console output to a log file.

    - Terminal output keeps ANSI styling.
    - File output is stripped of ANSI escape codes for readability.
    """

    def __init__(self) -> None:
        self._log_file = None

    def set_log_file(self, log_file) -> None:
        self._log_file = log_file

    def write(self, text: str) -> int:
        # Always target the current sys.stdout so pytest's `capsys` captures output.
        stream = sys.stdout
        # Rich writes strings (including ANSI control codes).
        written = stream.write(text)
        stream.flush()

        if self._log_file is not None:
            if getattr(self._log_file, "closed", False):
                self._log_file = None
            else:
                try:
                    clean = _ANSI_ESCAPE_RE.sub("", text)
                    self._log_file.write(clean)
                    self._log_file.flush()
                except (OSError, ValueError):
                    # Best-effort logging: keep console output working even if the log
                    # file handle was closed or became invalid.
                    self._log_file = None
        return written

    def flush(self) -> None:
        sys.stdout.flush()
        if self._log_file is not None:
            if getattr(self._log_file, "closed", False):
                self._log_file = None
            else:
                try:
                    self._log_file.flush()
                except (OSError, ValueError):
                    self._log_file = None

    def isatty(self) -> bool:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(sys.stdout, "encoding", "utf-8")


_TEE_STREAM = _TeeStream()
CONSOLE = Console(width=120, file=_TEE_STREAM)

_LOG_FILE_HANDLE = None
_LOG_FILE_PATH: Path | None = None
_PY_LOGGING_CONFIGURED = False


def _default_log_dir() -> Path:
    # Prefer a local ./logs directory when possible; fall back to the user's cache.
    cwd_logs = Path.cwd() / "logs"
    try:
        cwd_logs.mkdir(parents=True, exist_ok=True)
        test_file = cwd_logs / ".retracker_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return cwd_logs
    except OSError:
        cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        return cache_root / "retracker" / "logs"


def enable_file_logging(*, log_file: str | Path | None = None, log_dir: str | Path | None = None) -> Path | None:
    """Enable mirroring console output to a plaintext log file.

    If `log_file` is not provided, it is resolved from:
      1) `RETRACKER_LOG_FILE`
      2) `RETRACKER_LOG_DIR` + "retracker.log"
      3) `./logs/retracker.log` (or fallback to XDG cache if not writable)
    """

    global _LOG_FILE_HANDLE, _LOG_FILE_PATH

    if log_file is None:
        env_file = os.getenv("RETRACKER_LOG_FILE")
        if env_file:
            log_file = env_file
        else:
            env_dir = os.getenv("RETRACKER_LOG_DIR")
            if env_dir:
                log_dir = env_dir

    if log_file is None:
        base_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
        log_file = base_dir / "retracker.log"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent if called repeatedly with the same file.
    if (
        _LOG_FILE_HANDLE is not None
        and _LOG_FILE_PATH == log_path
        and not getattr(_LOG_FILE_HANDLE, "closed", False)
    ):
        return _LOG_FILE_PATH

    previous_handle = _LOG_FILE_HANDLE

    try:
        new_handle = log_path.open("a", encoding="utf-8")
    except OSError:
        # Keep any existing file logging (best-effort). Callers already treat `None`
        # as "logging to file is unavailable".
        if previous_handle is not None and getattr(previous_handle, "closed", False):
            _TEE_STREAM.set_log_file(None)
            _LOG_FILE_HANDLE = None
            _LOG_FILE_PATH = None
        return None

    _LOG_FILE_HANDLE = new_handle
    _LOG_FILE_PATH = log_path
    _TEE_STREAM.set_log_file(_LOG_FILE_HANDLE)

    if previous_handle is not None and previous_handle is not _LOG_FILE_HANDLE:
        try:
            previous_handle.close()
        except OSError:
            pass

    # If a previous handle existed but was closed by external code, ensure the tee stream
    # doesn't hold onto it.
    if previous_handle is not None and getattr(previous_handle, "closed", False):
        if _LOG_FILE_HANDLE is None:
            _TEE_STREAM.set_log_file(None)
            _LOG_FILE_PATH = None
    return _LOG_FILE_PATH


def disable_file_logging() -> None:
    """Disable mirroring console output to a log file (best-effort)."""
    global _LOG_FILE_HANDLE, _LOG_FILE_PATH

    _TEE_STREAM.set_log_file(None)
    if _LOG_FILE_HANDLE is not None:
        try:
            _LOG_FILE_HANDLE.close()
        except OSError:
            pass
    _LOG_FILE_HANDLE = None
    _LOG_FILE_PATH = None


def configure_python_logging(level: str | int | None = None) -> None:
    """Route standard `logging` output through Rich CONSOLE.

    Benefits:
    - Third-party libs using `logging` show up in the same console output.
    - Since CONSOLE is tee'd to a plaintext log file via `enable_file_logging`,
      these log messages are also persisted automatically.

    This is best-effort and idempotent.
    """

    global _PY_LOGGING_CONFIGURED
    if _PY_LOGGING_CONFIGURED:
        return

    import logging

    if level is None:
        level = os.getenv("RETRACKER_LOG_LEVEL", "INFO").upper()

    # Force reconfiguration so we don't duplicate logs when called multiple times
    # (e.g. CLI wrapper + internal modules).
    logging.basicConfig(
        level=level,
        handlers=[
            RichHandler(
                console=CONSOLE,
                rich_tracebacks=False,
                show_time=True,
                show_level=True,
                show_path=False,
            ),
        ],
        force=True,
    )
    _PY_LOGGING_CONFIGURED = True


class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def get_progress(description: str, suffix: Optional[str] = None):
    """Helper function to return a rich Progress object."""
    progress_list = [TextColumn(description), BarColumn(), TaskProgressColumn(show_speed=True)]
    progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
    progress_list += [TimeRemainingColumn(elapsed_when_finished=True, compact=True)]
    # Always render via our shared CONSOLE so output is mirrored to the log file
    # when `enable_file_logging()` is active.
    progress = Progress(*progress_list, console=CONSOLE, transient=True, refresh_per_second=10)
    return progress


def track(iterable, *, description: str, total: int | None = None):  # noqa: ANN001
    """A small wrapper around `rich.progress.track` using the shared CONSOLE."""
    from rich.progress import track as _track

    return _track(
        iterable,
        description=description,
        total=total,
        console=CONSOLE,
        transient=True,
        refresh_per_second=10,
    )

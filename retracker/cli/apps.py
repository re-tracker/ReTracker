from __future__ import annotations

import sys
from collections.abc import Callable

from retracker.utils.rich_utils import CONSOLE

def _run_script_main(prog: str, fn: Callable[[], object], argv: list[str]) -> int:
    """Run an argparse-style `main()` that reads `sys.argv`."""
    old_argv = sys.argv
    try:
        sys.argv = [prog, *argv]
        fn()
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv


def _missing_apps_deps(exc: ModuleNotFoundError) -> int:
    # Best-effort: if the failure comes from an optional dependency, provide a clear hint.
    missing = (exc.name or "").split(".")[0]
    if missing in {"gradio", "mediapy", "imageio"}:
        CONSOLE.print(
            "Missing optional dependencies for apps.\n"
            "Install with: pip install 'retracker[apps]'",
            markup=False,
        )
        return 1
    raise exc


def _print_help() -> None:
    CONSOLE.print(
        "retracker apps - runnable applications\n"
        "\n"
        "Usage:\n"
        "  retracker apps <app> [args...]\n"
        "\n"
        "Apps:\n"
        "  tracking   offline video tracking (v2)\n"
        "  streaming  streaming/online tracking\n"
        "  multiview  multi-view tracking pipeline\n"
        "\n"
        "Tip: `retracker apps <app> --help` shows the underlying app's options.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for `retracker apps ...`."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    app, rest = argv[0], argv[1:]

    try:
        if app == "tracking":
            from retracker.apps import tracking_video

            return _run_script_main("retracker apps tracking", tracking_video.main, rest)
        if app == "streaming":
            from retracker.apps import tracking_streaming

            return _run_script_main("retracker apps streaming", tracking_streaming.main, rest)
        if app == "multiview":
            from retracker.apps import multiview_tracking

            return _run_script_main("retracker apps multiview", multiview_tracking.main, rest)
    except ModuleNotFoundError as exc:
        return _missing_apps_deps(exc)

    CONSOLE.print(f"[red]Unknown app: {app!r}[/]\n")
    _print_help()
    return 2

from __future__ import annotations

import sys

from retracker.utils.rich_utils import CONSOLE, configure_python_logging, enable_file_logging

def _print_help() -> None:
    CONSOLE.print(
        "retracker - ReTracker command line interface\n"
        "\n"
        "Usage:\n"
        "  retracker <command> [args...]\n"
        "\n"
        "Commands:\n"
        "  apps   runnable applications (tracking/streaming/SLAM/multiview)\n"
        "  train  training (source checkout only; run in repo: pip install -e '.[train]')\n"
        "  eval   evaluation (source checkout only; run in repo: pip install -e '.[eval]')\n"
        "\n"
        "Run `retracker <command> --help` for command-specific help.\n"
        ,
        markup=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the `retracker` console script."""
    if argv is None:
        argv = sys.argv[1:]

    # Mirror console output to a plaintext log (best-effort).
    enable_file_logging()
    configure_python_logging()

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "apps":
        from retracker.cli.apps import main as apps_main

        return apps_main(rest)
    if cmd == "train":
        from retracker.cli.train import main as train_main

        return train_main(rest)
    if cmd == "eval":
        from retracker.cli.eval import main as eval_main

        return eval_main(rest)

    CONSOLE.print(f"[red]Unknown command: {cmd!r}[/]\n")
    _print_help()
    return 2

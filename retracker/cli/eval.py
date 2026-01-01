from __future__ import annotations

import sys

from retracker.utils.rich_utils import CONSOLE


def main(argv: list[str] | None = None) -> int:
    """Entry point for `retracker eval` (source checkout only)."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        from retracker.evaluation import cli as eval_cli
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"hydra", "hydra_core", "matplotlib", "imageio"}:
            CONSOLE.print(
                "Missing optional dependencies for evaluation.\n"
                "Install (from a source checkout) with: pip install -e '.[eval]'",
                markup=False,
            )
            return 1
        raise

    old_argv = sys.argv
    try:
        sys.argv = ["retracker eval", *argv]
        eval_cli.evaluate()
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv

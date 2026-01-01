from __future__ import annotations

import sys

from retracker.utils.rich_utils import CONSOLE


def main(argv: list[str] | None = None) -> int:
    """Entry point for `retracker train` (source checkout only)."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        from retracker.training import cli as train_cli
    except ModuleNotFoundError as exc:
        # Provide a clear action for the most common case: missing optional deps.
        if exc.name and exc.name.split(".")[0] in {
            "albumentations",
            "h5py",
            "hydra",
            "hydra_core",
            "jsonargparse",
            "lightning",
            "matplotlib",
            "omegaconf",
            "pytorch_lightning",
            "tensorboard",
            "wandb",
        }:
            CONSOLE.print(
                "Missing optional dependencies for training.\n"
                "Install (from a source checkout) with: pip install -e '.[train]'",
                markup=False,
            )
            return 1
        raise

    # Delegate to the existing training CLI which reads sys.argv.
    old_argv = sys.argv
    try:
        # Keep a stable argv[0] so help text looks like `retracker train ...`.
        sys.argv = ["retracker train", *argv]
        train_cli.cli_main()
        return 0
    except SystemExit as exc:
        # Preserve exit code semantics from underlying CLI.
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv

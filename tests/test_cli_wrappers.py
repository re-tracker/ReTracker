import importlib


def test_cli_wrapper_imports_without_lightning():
    # Importing the wrapper must not import training deps at import time.
    mod = importlib.import_module("retracker.cli.train")
    assert hasattr(mod, "main")

    mod = importlib.import_module("retracker.cli.main")
    assert hasattr(mod, "main")

    # Allow `python -m retracker ...` without requiring an installation.
    importlib.import_module("retracker.__main__")

import importlib

import pytest


def test_models_modules_imports_cleanly():
    try:
        module = importlib.import_module("retracker.models.modules")
    except Exception as exc:
        pytest.fail(f"Import failed: {exc!r}")

    assert hasattr(module, "TransformerDecoder")

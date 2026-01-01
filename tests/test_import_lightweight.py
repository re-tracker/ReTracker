from __future__ import annotations

import builtins
import importlib
import sys


def test_import_retracker_does_not_import_lightning():
    assert "lightning" not in sys.modules
    import retracker  # noqa: F401
    assert "lightning" not in sys.modules


def test_import_slam_video_does_not_require_cv2(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("blocked cv2 import for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    sys.modules.pop("retracker.apps.slam_video", None)
    importlib.import_module("retracker.apps.slam_video")


def test_import_slam_config_does_not_require_cv2(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("blocked cv2 import for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    sys.modules.pop("retracker.apps.slam", None)
    sys.modules.pop("retracker.apps.slam.config", None)
    importlib.import_module("retracker.apps.slam.config")

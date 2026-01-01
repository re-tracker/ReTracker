from __future__ import annotations

from types import SimpleNamespace

import retracker.inference.engines.offline as offline_mod
import retracker.utils.checkpoint as ckpt_utils


class _DummyModel:
    def __init__(self) -> None:
        self.loaded_state_dict = None
        self.loaded_strict = None

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: ANN001
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def test_load_state_dict_from_pl_checkpoint_uses_cpu_map_location(monkeypatch, tmp_path):
    called = {}

    def _fake_torch_load(path, *args, **kwargs):  # noqa: ANN001
        called["path"] = path
        called["args"] = args
        called["kwargs"] = kwargs
        # Minimal Lightning checkpoint shape.
        return {"state_dict": {"matcher.foo": 1, "loss.bar": 2}}

    # Patch torch.load where it's actually used (safe_torch_load helper).
    monkeypatch.setattr(ckpt_utils.torch, "load", _fake_torch_load)

    ckpt = tmp_path / "ckpt.ckpt"
    ckpt.write_bytes(b"dummy")

    model = _DummyModel()
    from retracker.inference.engine import ReTrackerEngine

    ReTrackerEngine.load_state_dict_from_pl_checkpoint(object(), model, str(ckpt))

    # On CPU-only machines, loading GPU-trained checkpoints must use map_location="cpu".
    assert called["kwargs"].get("map_location") == "cpu"
    assert called["kwargs"].get("weights_only") is True

    # Keys are filtered/normalized before loading.
    assert model.loaded_state_dict == {"foo": 1}
    assert model.loaded_strict is True

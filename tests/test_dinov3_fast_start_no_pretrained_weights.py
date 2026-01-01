from __future__ import annotations

import torch


def test_dinov3_backbone_does_not_pass_weights_when_pretrained_false(monkeypatch) -> None:
    """Regression test for --fast_start.

    When `pretrained=False`, we should *not* pass an explicit `weights=...` to
    `torch.hub.load`, otherwise the hub implementation may still load those
    weights and negate the goal of fast-start (we immediately load a full
    ReTracker checkpoint).
    """

    calls: list[tuple[str, dict]] = []

    class _DummyViT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 16
            self._p = torch.nn.Parameter(torch.zeros(()))

    class _DummyCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.downsample_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.stages = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def _fake_hub_load(repo_or_dir, model, *args, source="github", **kwargs):  # noqa: ANN001
        _ = repo_or_dir, args, source
        calls.append((model, dict(kwargs)))
        if model == "dinov3_vitl16":
            return _DummyViT()
        if model == "dinov3_convnext_tiny":
            return _DummyCNN()
        raise AssertionError(f"Unexpected torch.hub model: {model!r}")

    monkeypatch.setattr(torch.hub, "load", _fake_hub_load)

    from retracker.models.backbone.dino_backbone_v3 import DINOv3_backbone_vitconvnext

    _ = DINOv3_backbone_vitconvnext({"pretrained": False, "cnn_pretrained": False})

    assert len(calls) == 2
    assert calls[0][0] == "dinov3_vitl16"
    assert "weights" not in calls[0][1]
    assert calls[1][0] == "dinov3_convnext_tiny"
    assert "weights" not in calls[1][1]


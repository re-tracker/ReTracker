from __future__ import annotations

import torch


def test_offline_engine_video_forward_cpu_shapes_and_no_cuda_sync(monkeypatch) -> None:
    """Regression test: CPU inference must not call torch.cuda.synchronize()."""

    def _boom():  # noqa: ANN001
        raise AssertionError("torch.cuda.synchronize() should not be called for CPU inference")

    monkeypatch.setattr(torch.cuda, "synchronize", _boom)

    from retracker.inference.engines.offline import ReTrackerEngine

    class _DummyMemManager:
        MAX_MEMORY_SIZE = 0

        def __init__(self) -> None:
            self.sample_method = "foremost"

        def reset_all_memory(self) -> None:
            return None

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))
            self.mem_manager = _DummyMemManager()

        def set_max_queries(self, _max):  # noqa: ANN001
            return None

        def video_forward(self, batch, use_aug=False, return_dense_flow=False):  # noqa: ANN001, FBT002
            images = batch["images"]  # [B, T, C, H, W]
            queries = batch["queries"]  # [B, N, 2]
            n = int(queries.shape[1])
            t = int(images.shape[1])
            mkpts1_f = torch.zeros((n, t, 2), device=images.device)
            pred_visibles = torch.ones((n, t, 1), device=images.device, dtype=torch.bool)
            visibility_scores = torch.ones((n, t, 1), device=images.device, dtype=torch.float32)
            return {
                "mkpts1_f": mkpts1_f,
                "pred_visibles": pred_visibles,
                "visibility_scores": visibility_scores,
            }

    engine = ReTrackerEngine(kttr_model=_DummyModel())
    engine.eval()
    engine.to("cpu")

    video = torch.zeros((1, 2, 3, 4, 4), dtype=torch.uint8)
    queries = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

    traj, vis = engine.video_forward(video, queries, use_aug=False)
    assert traj.shape == (1, 2, 1, 2)
    assert vis.shape == (1, 2, 1)


def test_offline_engine_video_forward_multibatch_shapes() -> None:
    from retracker.inference.engines.offline import ReTrackerEngine

    class _DummyMemManager:
        MAX_MEMORY_SIZE = 0

        def __init__(self) -> None:
            self.sample_method = "foremost"

        def reset_all_memory(self) -> None:
            return None

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))
            self.mem_manager = _DummyMemManager()

        def set_max_queries(self, _max):  # noqa: ANN001
            return None

        def video_forward(self, batch, use_aug=False, return_dense_flow=False):  # noqa: ANN001, FBT002
            images = batch["images"]  # [B, T, C, H, W]
            queries = batch["queries"]  # [B, N, 2]
            b = int(images.shape[0])
            n = int(queries.shape[1])
            t = int(images.shape[1])
            # ReTrackerEngine expects per-(B=1) outputs from the wrapped model.
            assert b == 1
            mkpts1_f = torch.zeros((n, t, 2), device=images.device)
            pred_visibles = torch.ones((n, t, 1), device=images.device, dtype=torch.bool)
            return {"mkpts1_f": mkpts1_f, "pred_visibles": pred_visibles}

    engine = ReTrackerEngine(kttr_model=_DummyModel())
    engine.eval()
    engine.to("cpu")

    video = torch.zeros((2, 2, 3, 4, 4), dtype=torch.uint8)
    queries = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]], dtype=torch.float32)

    traj, vis = engine.video_forward(video, queries, use_aug=False)
    assert traj.shape == (2, 2, 1, 2)
    assert vis.shape == (2, 2, 1)


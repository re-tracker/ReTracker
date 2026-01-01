from __future__ import annotations

from types import SimpleNamespace

import torch


def _minimal_pl_config() -> dict:
    # Keep config tiny; PL_ReTracker will OmegaConf-wrap it.
    return {
        "retracker_config": {"debug_mode": False, "debug_batches": 0},
        "loss_config": {},
        "viz_config": {"enable_plotting": False, "matches_plot_interval": 10**9},
    }


def test_unified_training_step_routes_matching_batches(monkeypatch):
    """Matching batches (T=2) must NOT go through the video pair constructor.

    Regression test for a crash:
      KeyError: 'occs' in retracker/training/utils/videodata.py:construct_pairs
    when a MegaDepth matching batch was incorrectly routed through the video path.
    """
    from retracker.training.lightning_module import PL_ReTracker

    recorded_task_modes: list[str] = []

    def fake_init_matcher_and_loss(self):
        self.matcher = SimpleNamespace(
            model_task_type="image_matching",
            set_task_mode=lambda mode: recorded_task_modes.append(mode),
            mem_manager=SimpleNamespace(reset_all_memory=lambda: None),
        )
        self.loss = SimpleNamespace()

    monkeypatch.setattr(PL_ReTracker, "_initialize_matcher_and_loss", fake_init_matcher_and_loss)

    module = PL_ReTracker(config=_minimal_pl_config(), profiler=None)
    module.trainer = SimpleNamespace(global_rank=0, world_size=1)
    # Avoid Lightning plumbing; we only test routing.
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    calls = {"video": 0, "matching": 0}

    def fake_video(batch, type):
        calls["video"] += 1
        batch["loss_scalars"] = {"loss": torch.tensor(0.0)}
        return {"loss": torch.tensor(0.0)}

    def fake_matching(batch, type):
        calls["matching"] += 1
        batch["loss_scalars"] = {"loss": torch.tensor(0.0)}
        return {"loss": torch.tensor(0.0)}

    monkeypatch.setattr(module, "_causal_trainval_video", fake_video)
    monkeypatch.setattr(module, "_trainval_image_matching", fake_matching, raising=False)

    # Minimal matching-like batch: images with T=2, no occs/trajs keys.
    batch = {
        "images": torch.zeros((1, 2, 1, 8, 8), dtype=torch.float32),
        "dataset_name": ["MegaDepth"],
        "scene_name": ["dummy_scene"],
    }

    module._unified_training_step(batch, batch_idx=0)

    assert calls["matching"] == 1
    assert calls["video"] == 0
    assert recorded_task_modes == ["matching"]


def test_unified_training_step_routes_tracking_batches(monkeypatch):
    """Tracking batches (T>2) should continue using the video path."""
    from retracker.training.lightning_module import PL_ReTracker

    def fake_init_matcher_and_loss(self):
        self.matcher = SimpleNamespace(
            model_task_type="causal_video_matching",
            set_task_mode=lambda mode: None,
            mem_manager=SimpleNamespace(reset_all_memory=lambda: None),
        )
        self.loss = SimpleNamespace()

    monkeypatch.setattr(PL_ReTracker, "_initialize_matcher_and_loss", fake_init_matcher_and_loss)

    module = PL_ReTracker(config=_minimal_pl_config(), profiler=None)
    module.trainer = SimpleNamespace(global_rank=0, world_size=1)
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    calls = {"video": 0}

    def fake_video(batch, type):
        calls["video"] += 1
        batch["loss_scalars"] = {"loss": torch.tensor(0.0)}
        return {"loss": torch.tensor(0.0)}

    monkeypatch.setattr(module, "_causal_trainval_video", fake_video)

    batch = {
        "images": torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32),
        "dataset_name": ["PointOdyssey"],
        "scene_name": ["dummy_scene"],
        # The video path will likely expect these in real training, but the
        # stubbed fake_video avoids touching them in this unit test.
    }

    module._unified_training_step(batch, batch_idx=0)

    assert calls["video"] == 1


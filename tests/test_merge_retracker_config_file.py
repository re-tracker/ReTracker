from pathlib import Path

from omegaconf import OmegaConf

from retracker.config_utils import merge_retracker_config_file


def test_merge_retracker_config_file_merges_and_removes_key():
    cfg = OmegaConf.create(
        {
            "model": {
                "config": {
                    "retracker_config_file": "configs/model/retracker.yaml",
                    "retracker_config": {"debug_mode": True},
                }
            }
        }
    )

    out = merge_retracker_config_file(cfg, project_root=Path.cwd())
    assert "retracker_config_file" not in out.model.config
    assert out.model.config.retracker_config.debug_mode is True
    assert "dino" in out.model.config.retracker_config


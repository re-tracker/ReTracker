from omegaconf import OmegaConf


def test_shared_model_config_file_exists_and_loads():
    cfg = OmegaConf.load("configs/model/retracker.yaml")
    assert "dino" in cfg


import importlib.resources as resources

from retracker.models import get_cfg_defaults


def test_default_config_is_packaged_and_loadable():
    cfg_path = resources.files("retracker") / "configs" / "default_retracker_config.yaml"
    assert cfg_path.is_file()

    cfg = get_cfg_defaults()
    assert "dino" in cfg


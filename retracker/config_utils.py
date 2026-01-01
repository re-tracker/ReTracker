from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def merge_retracker_config_file(cfg: OmegaConf, project_root: Path) -> OmegaConf:
    """Merge a shared ReTracker model config file into `model.config.retracker_config`.

    Training configs can specify:
      model:
        config:
          retracker_config_file: configs/model/retracker.yaml
          retracker_config: { ... overrides ... }

    Merge order:
      base_file < existing overrides

    The helper deletes `retracker_config_file` after merging so downstream consumers
    (LightningCLI, saved configs) only see the resolved `retracker_config` dict.
    """
    if "model" not in cfg or "config" not in cfg.model:
        return cfg

    model_cfg = cfg.model.config
    cfg_file: Optional[str] = model_cfg.get("retracker_config_file")
    if not cfg_file:
        return cfg

    p = Path(cfg_file)
    if not p.is_absolute():
        p = project_root / p

    if not p.exists():
        raise FileNotFoundError(
            f"ReTracker model config file not found: {p} (from retracker_config_file={cfg_file})"
        )

    base = OmegaConf.load(str(p))
    overrides = model_cfg.get("retracker_config") or OmegaConf.create()
    model_cfg["retracker_config"] = OmegaConf.merge(base, overrides)

    # Keep the final config clean: this key is only for preprocessing.
    del model_cfg["retracker_config_file"]
    return cfg


# get file path
import importlib.resources as resources
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from retracker.config_utils import merge_retracker_config_file
from retracker.utils.checkpoint import safe_torch_load

from .retracker import ReTracker


def _load_paths_config(project_root: str) -> OmegaConf:
    """Load paths config, with local override if available."""
    paths_config = OmegaConf.create({})

    # Load base paths.yaml
    paths_file = os.path.join(project_root, "configs/paths.yaml")
    if os.path.exists(paths_file):
        paths_config = OmegaConf.load(paths_file)

    # Override with paths_local.yaml if exists
    paths_local_file = os.path.join(project_root, "configs/paths_local.yaml")
    if os.path.exists(paths_local_file):
        paths_local = OmegaConf.load(paths_local_file)
        paths_config = OmegaConf.merge(paths_config, paths_local)

    return paths_config


def get_cfg_defaults():
    """Return the default ReTracker config.

    Packaging note:
    - For wheel installs, repo-local `configs/` may not exist.
    - We therefore prefer a packaged, resolved YAML under `retracker/configs/`.
    - We keep the repo-local path as a fallback for local development/training.
    """
    # 1) Wheel-safe default (packaged with the library)
    packaged = resources.files("retracker") / "configs" / "default_retracker_config.yaml"
    if packaged.is_file():
        return OmegaConf.load(str(packaged))

    # 2) Fallback for editable/dev checkouts (repo root has configs/)
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_path, "../..")
    # Keep repo-local defaults inference-first: do not depend on training configs.
    default_config_path = os.path.join(project_root, "configs/model/retracker.yaml")

    paths_config = _load_paths_config(project_root)
    retracker_config = OmegaConf.load(default_config_path)
    merged_config = OmegaConf.merge(paths_config, retracker_config)
    resolved = OmegaConf.to_container(merged_config, resolve=True)
    # Ensure we return only the model config, not the `paths` helper tree.
    resolved.pop("paths", None)
    return OmegaConf.create(resolved)


def build_retracker(config_path=None, checkpoint_path=None, device=None, inference_mode=True):
    """
    Build ReTracker model with config and checkpoint.

    Args:
        config_path: Path to config YAML file. If None, uses default config.
        checkpoint_path: Path to checkpoint file. If None, uses default pretrained weights.
        device: Target device (e.g., 'cuda', 'cpu'). If None, stays on CPU.
        inference_mode: If True, sets model to eval mode.
    """
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")

    if config_path is None:
        # Prefer the packaged inference config (or repo-local inference defaults).
        retracker_config = get_cfg_defaults()
    else:
        # Load paths config first for interpolation
        paths_config = _load_paths_config(project_root)

        # Load main config
        config = OmegaConf.load(config_path)

        # Merge paths into config for interpolation resolution
        merged_config = OmegaConf.merge(paths_config, config)

        # Support both:
        # - training-style configs (model.config.*) that may reference `retracker_config_file`
        # - direct `retracker_config` YAMLs (inference-only)
        merged_config = merge_retracker_config_file(merged_config, project_root=Path(project_root))

        # Handle both old and new config structures
        if "model" in merged_config and "config" in merged_config.model:
            retracker_config = merged_config.model.config.retracker_config
        else:
            # Assume it's already the retracker_config
            retracker_config = merged_config

        # Resolve all interpolations
        retracker_config = OmegaConf.to_container(retracker_config, resolve=True)
        retracker_config = OmegaConf.create(retracker_config)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            project_root, "weights/flyingthings_megadepth_pretrained_lightning.ckpt"
        )

    retracker_model = ReTracker(config=retracker_config)
    if device is not None:
        retracker_model = retracker_model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        retracker_model.load_state_dict(state_dict, strict=False)

    if inference_mode:
        retracker_model.eval()
        retracker_model.training = False

    return retracker_model


# Alias for convenience
build_model = build_retracker

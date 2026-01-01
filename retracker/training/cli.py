from __future__ import annotations

import ast
import os
import sys
import argparse
import tempfile
import time
from pathlib import Path
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from retracker.config_utils import merge_retracker_config_file
from retracker.training.profiler import build_profiler
from retracker.training.lightning_module import PL_ReTracker
from retracker.utils.device import cuda_mem_info, format_cuda_mem_info
from retracker.utils.rich_utils import CONSOLE, configure_python_logging, enable_file_logging

enable_file_logging()
configure_python_logging()


def get_project_root() -> Path:
    """Get the project root directory (where configs/ is located)."""
    # Prefer the repository root (contains pyproject.toml and top-level configs/paths.yaml).
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
        if (parent / "configs" / "paths.yaml").is_file():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def load_paths_config() -> OmegaConf:
    """
    Load path configuration from configs/paths.yaml and configs/paths_local.yaml.

    Priority:
    1. paths_local.yaml (machine-specific, gitignored)
    2. paths.yaml (defaults)

    Returns:
        OmegaConf: Merged path configuration
    """
    project_root = get_project_root()
    configs_dir = project_root / "configs"

    paths_default = configs_dir / "paths.yaml"
    paths_local = configs_dir / "paths_local.yaml"

    paths_cfg = OmegaConf.create()

    # Load default paths
    if paths_default.exists():
        paths_cfg = OmegaConf.load(paths_default)
        CONSOLE.print(f"[dim]Loaded default paths from: {paths_default}[/dim]")

    # Override with local paths if exists
    if paths_local.exists():
        local_cfg = OmegaConf.load(paths_local)
        paths_cfg = OmegaConf.merge(paths_cfg, local_cfg)
        CONSOLE.print(f"[dim]Merged local paths from: {paths_local}[/dim]")

    return paths_cfg


def resolve_paths_in_config(cfg: OmegaConf, paths_cfg: OmegaConf) -> OmegaConf:
    """
    Resolve path interpolations in config using paths configuration.

    This function merges paths config with the main config so that
    ${paths.DATA_ROOT} style interpolations work, then removes the
    paths key so LightningCLI doesn't complain.

    Args:
        cfg: Main configuration
        paths_cfg: Paths configuration

    Returns:
        OmegaConf: Config with paths resolved and paths key removed
    """
    # Merge paths into the config so interpolation works
    merged = OmegaConf.merge(paths_cfg, cfg)

    # Resolve all interpolations
    OmegaConf.resolve(merged)

    # Remove the 'paths' key as LightningCLI doesn't accept it
    if 'paths' in merged:
        del merged['paths']

    return merged

def is_distributed():
    """Check if running in distributed mode."""
    return int(os.environ.get('WORLD_SIZE', 1)) > 1

def get_rank():
    """Get current process rank."""
    return int(os.environ.get('RANK', 0))

def _argv_value(argv: list[str], key: str) -> str | None:
    """Return the value for an argparse-style key in argv.

    Supports both:
      --key value
      --key=value
    """
    for i, arg in enumerate(argv):
        if arg == key and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
    return None


def _preflight_cuda_memory(argv: list[str]) -> None:
    """Fail fast when CUDA is requested but free memory is clearly insufficient.

    This avoids spending minutes loading checkpoints/backbones only to OOM when
    Lightning moves the model to GPU.
    """
    accelerator = _argv_value(argv, "--trainer.accelerator")
    if accelerator is None:
        return
    accelerator = str(accelerator).lower()
    if accelerator not in {"cuda", "gpu"}:
        return

    infos = cuda_mem_info()
    if not infos:
        return

    min_free_gb = float(os.getenv("RETRACKER_MIN_CUDA_FREE_GB", "2.0"))
    devices_raw = _argv_value(argv, "--trainer.devices")

    # Default selection: if devices are not specified, assume Lightning may use
    # multiple GPUs (depending on config), so check all visible devices.
    selected = [d["index"] for d in infos]
    if devices_raw is not None:
        s = str(devices_raw).strip()
        if s.lower() == "auto":
            selected = [d["index"] for d in infos]
        elif s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    selected = [int(x) for x in parsed]
            except Exception:
                # Fall back to checking all visible devices.
                selected = [d["index"] for d in infos]
        else:
            # `devices=1` means "use 1 GPU" (implicitly cuda:0).
            try:
                n = int(s)
                if n <= 0:
                    selected = [d["index"] for d in infos]
                else:
                    selected = list(range(min(n, len(infos))))
            except ValueError:
                selected = [d["index"] for d in infos]

    free_by_idx = {d["index"]: d["free_bytes"] / (1024**3) for d in infos}
    bad = [idx for idx in selected if free_by_idx.get(idx, 0.0) < min_free_gb]
    if not bad:
        return

    msg = (
        "ERROR: Insufficient free CUDA memory for the requested training run.\n"
        f"  trainer.accelerator={accelerator}\n"
        f"  trainer.devices={devices_raw!r}\n"
        f"  min_free_gb={min_free_gb:.2f}\n"
        "\n"
        "Visible CUDA memory:\n"
        f"{format_cuda_mem_info(infos)}\n"
        "\n"
        "Tip: free GPU memory, choose a different GPU (e.g. --trainer.devices='[1]'), "
        "or run on CPU (--trainer.accelerator=cpu)."
    )
    CONSOLE.print(msg, markup=False)
    raise SystemExit(2)


def merge_configs_and_update_argv():
    """
    Imitating hydra default config merging logic.
    Parses custom config arguments, merges the files, writes to a temp file,
    and updates sys.argv for LightningCLI.
    
    Supports both single-node and multi-node training:
    - Single-node: Uses temporary file in /tmp
    - Multi-node: Rank 0 creates config in shared location, other ranks wait and use it
    """
    # 1. Pre-parse only our custom arguments, ignore others.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--defaults_config", type=str, required=True)
    parser.add_argument("--exp_config", type=str, default=None)
    
    args, remaining_argv = parser.parse_known_args()
    
    distributed = is_distributed()
    rank = get_rank()
    
    # For multi-node: use a shared location based on output dir
    if distributed:
        # Extract output dir from remaining_argv to create shared config location
        output_dir = "./outputs"  # default
        for i, arg in enumerate(remaining_argv):
            if '--trainer.logger.init_args.save_dir' in arg:
                output_dir = arg.split('=')[1]
                break
        
        # Create shared config directory
        shared_config_dir = Path(output_dir) / ".tmp_configs"
        shared_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a consistent filename based on exp config
        config_basename = Path(args.exp_config or args.defaults_config).stem
        temp_config_path = str(shared_config_dir / f"merged_{config_basename}.yaml")
    else:
        # Single-node: use system temp
        temp_config_path = None

    # 2. Rank 0 merges and saves config
    if rank == 0:
        project_root = get_project_root()
        # Load paths configuration first
        paths_cfg = load_paths_config()

        # Load and merge configurations
        if not os.path.exists(args.defaults_config):
            raise FileNotFoundError(f"Default config file not found: {args.defaults_config}")

        merged_cfg = OmegaConf.load(args.defaults_config)
        CONSOLE.print(f"[dim]Loaded default config from: {args.defaults_config}[/dim]")

        if args.exp_config:
            if not os.path.exists(args.exp_config):
                raise FileNotFoundError(f"Experiment config file not found: {args.exp_config}")
            exp_cfg = OmegaConf.load(args.exp_config)
            merged_cfg = OmegaConf.merge(merged_cfg, exp_cfg)
            CONSOLE.print(f"[dim]Merged experiment config from: {args.exp_config}[/dim]")

        # Allow training configs to reference a shared, reusable model config file.
        merged_cfg = merge_retracker_config_file(merged_cfg, project_root=project_root)

        # Resolve path interpolations
        merged_cfg = resolve_paths_in_config(merged_cfg, paths_cfg)
        
        # Write the merged config
        if distributed:
            # Multi-node: write to shared location
            with open(temp_config_path, 'w') as f:
                OmegaConf.save(config=merged_cfg, f=f)
            CONSOLE.print(f"[dim][Rank 0] Saved merged config to shared file: {temp_config_path}[/dim]")
        else:
            # Single-node: write to temp file
            with tempfile.NamedTemporaryFile(mode='w', prefix='training_', suffix='.yaml', delete=False) as tmp:
                temp_config_path = tmp.name
                OmegaConf.save(config=merged_cfg, f=tmp.name)
                CONSOLE.print(f"[dim]Saved merged config to temporary file: {temp_config_path}[/dim]")
    
    # 3. Non-rank-0 processes wait for config file
    if distributed and rank != 0:
        max_wait = 300  # 5 minutes
        wait_interval = 1
        elapsed = 0
        while not os.path.exists(temp_config_path):
            time.sleep(wait_interval)
            elapsed += wait_interval
            if elapsed >= max_wait:
                raise TimeoutError(f"[Rank {rank}] Timeout waiting for config file: {temp_config_path}")
        CONSOLE.print(f"[dim][Rank {rank}] Found shared config file: {temp_config_path}[/dim]")
    
    # 4. Rebuild sys.argv for LightningCLI.
    #
    # Important: jsonargparse processes config files in argv order, so `--config`
    # must appear *before* any explicit overrides (e.g. --model.dump_dir=...),
    # otherwise the config file would overwrite the overrides.
    original_script_name = sys.argv[0]
    new_argv = [original_script_name]

    if remaining_argv and not remaining_argv[0].startswith("-"):
        # Typical LightningCLI pattern: <subcommand> [args...]
        subcommand = remaining_argv[0]
        rest = list(remaining_argv[1:])

        # jsonargparse loads values in argv order. We want:
        # 1) optional trainer defaults (e.g. --trainer configs/train/trainer_default.yaml)
        # 2) merged config file (--config <tmp>)
        # 3) explicit CLI overrides (rest args like --trainer.devices=..., --model.dump_dir=...)
        #
        # Therefore, if the user provided a trainer config file via `--trainer <path.yaml>`,
        # move that pair in front of `--config` so it behaves like a lower-precedence default.
        trainer_cfg_tokens: list[str] = []
        for i in range(len(rest) - 1):
            if rest[i] != "--trainer":
                continue
            cand = rest[i + 1]
            if cand.startswith("-"):
                continue
            if not cand.endswith((".yaml", ".yml")):
                continue
            trainer_cfg_tokens = [rest[i], cand]
            del rest[i : i + 2]
            break

        new_argv += [subcommand] + trainer_cfg_tokens + ["--config", temp_config_path] + rest
    else:
        # Fallback: no explicit subcommand detected.
        new_argv += ["--config", temp_config_path] + remaining_argv

    sys.argv = new_argv
    
    CONSOLE.print(f"[dim][Rank {rank}] Updated sys.argv for LightningCLI: {sys.argv}[/dim]")
    return temp_config_path

def _preprocess_direct_config_and_update_argv() -> str | None:
    """Resolve `${paths.*}` interpolations for `--config` mode.

    LightningCLI / jsonargparse does not know about our `paths.yaml` convention.
    When users run:
        retracker train fit --config configs/train/stage4_unified.yaml
    we still want `${paths.*}` interpolations to work.

    This function:
    - locates the `--config` (or `-c`) argument in sys.argv,
    - loads it with OmegaConf,
    - merges in configs/paths.yaml (+ paths_local.yaml),
    - resolves interpolations,
    - writes a temporary resolved yaml,
    - replaces argv to point to the resolved file.
    """
    argv = list(sys.argv)
    cfg_path = None
    cfg_idx = None

    for i, arg in enumerate(argv):
        if arg in {"--config", "-c"} and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            cfg_idx = i + 1
            break
        if arg.startswith("--config="):
            cfg_path = arg.split("=", 1)[1]
            cfg_idx = i
            break

    if not cfg_path:
        return None

    project_root = get_project_root()
    paths_cfg = load_paths_config()
    merged_cfg = OmegaConf.load(cfg_path)

    merged_cfg = merge_retracker_config_file(merged_cfg, project_root=project_root)
    merged_cfg = resolve_paths_in_config(merged_cfg, paths_cfg)

    with tempfile.NamedTemporaryFile(mode="w", prefix="training_resolved_", suffix=".yaml", delete=False) as tmp:
        OmegaConf.save(config=merged_cfg, f=tmp.name)
        resolved_path = tmp.name

    # Replace argv with resolved config path.
    if cfg_idx is None:
        return None
    if argv[cfg_idx - 1] in {"--config", "-c"}:
        argv[cfg_idx] = resolved_path
    else:
        argv[cfg_idx] = f"--config={resolved_path}"

    sys.argv = argv
    CONSOLE.print(f"[dim]Resolved paths interpolations for --config mode: {cfg_path} -> {resolved_path}[/dim]")
    return resolved_path

def cli_main():
    """
    Main function to run the pre-processing and then the LightningCLI.
    Supports both single-node and multi-node training.
    """
    
    # Check if config merging is needed (--defaults_config in argv)
    if '--defaults_config' in sys.argv:
        temp_file_path = merge_configs_and_update_argv()
        CONSOLE.print("[dim]Config merging enabled - works in both single-node and multi-node![/dim]")
    else:
        temp_file_path = _preprocess_direct_config_and_update_argv()
        CONSOLE.print("[dim]Direct config mode - using --config (with paths interpolation resolution)[/dim]")

    _preflight_cuda_memory(sys.argv)
    
    try:
        # Now, instantiate the standard LightningCLI. It will read the updated sys.argv.
        # Use pl.LightningDataModule to accept any datamodule subclass
        # (SequencialDataModule, UnifiedDataModule, etc.)
        cli = LightningCLI(
            model_class=PL_ReTracker,
            datamodule_class=pl.LightningDataModule,
            subclass_mode_data=True,
            save_config_kwargs={'overwrite': True}
        )
    finally:
        # Cleanup temp file only for single-node (multi-node uses shared location)
        if temp_file_path and not is_distributed() and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                CONSOLE.print(f"[dim]Cleaned up temporary config file: {temp_file_path}[/dim]")
            except Exception as e:
                CONSOLE.print(f"[yellow]Failed to cleanup temp file {temp_file_path}: {e}[/yellow]")


if __name__ == '__main__':
    enable_file_logging()
    configure_python_logging()
    CONSOLE.print("[dim]Starting ReTracker CLI (supports single-node & multi-node)![/dim]")
    cli_main()

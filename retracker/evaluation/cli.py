# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from retracker.data.datasets.tap_vid_datasets import TapVidDataset
from retracker.data.datasets.utils import collate_fn
from retracker.evaluation.publicevaluator import PublicEvaluator
from retracker.inference.engine import ReTrackerEngine
from retracker.utils.device import cuda_mem_info, format_cuda_mem_info
from retracker.utils.rich_utils import CONSOLE, configure_python_logging, enable_file_logging

@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs/eval"
    dataset_name: str = "tapvid_davis_first"
    dataset_root: str = "./"
    method_name: str = "retracker"  # "cotracker" (legacy alias for backward compat)
    checkpoint: Optional[str] = None
    locotrack_ckpt_path: Optional[str] = None
    seed: int = 0
    gpu_idx: int = 0
    interp_shape: tuple[int, int] = (512, 512)
    enable_highres_inference: bool = False
    coarse_resolution: tuple[int, int] = (512, 512)
    tapvid_resize_to: tuple[int, int] | None = (256, 256)
    # Multi-GPU / multi-process sharding (sequence-level parallelism).
    shard_id: int = 0
    num_shards: int = 1
    # Caching / resume.
    load_dump: bool = True
    save_dump: bool = False
    skip_if_done: bool = False
    # Visualization.
    visualize_every: int = 1
    save_video: bool = True
    # DataLoader.
    num_workers: int = 14
    # TAP-Vid helper: subsample for quick smoke tests.
    fast_eval: bool = False


def _get_project_root() -> Path:
    """Find the repository root (contains pyproject.toml / configs/paths.yaml)."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
        if (parent / "configs" / "paths.yaml").is_file():
            return parent
    return Path.cwd()


def _load_paths_config(project_root: Path) -> OmegaConf:
    """Load configs/paths.yaml and merge in configs/paths_local.yaml if present."""
    cfg = OmegaConf.create()
    defaults = project_root / "configs" / "paths.yaml"
    local = project_root / "configs" / "paths_local.yaml"

    if defaults.is_file():
        cfg = OmegaConf.load(str(defaults))
    if local.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(local)))
    return cfg


def _resolve_path(path: str, *, project_root: Path) -> str:
    """Resolve a (possibly relative) path against the repo root.

    Hydra often changes CWD into `hydra.run.dir`, so we must not rely on CWD
    when interpreting relative paths from configs.
    """
    p = Path(str(path))
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def run_eval(cfg: DefaultConfig):
    """Evaluate ReTracker on a benchmark dataset based on a provided configuration.

    Args:
        cfg (DefaultConfig): An instance of DefaultConfig class which includes:
            - exp_dir (str): The directory path for the experiment.
            - dataset_name (str): The name of the dataset to be used.
            - dataset_root (str): The root directory of the dataset.
            - checkpoint (str): The path to the CoTracker model's checkpoint.
            - single_point (bool): A flag indicating whether to evaluate one ground truth point at a time.
            - n_iters (int): The number of iterative updates for each sliding window.
            - seed (int): The seed for setting the random state for reproducibility.
            - gpu_idx (int): The index of the GPU to be used.
    """
    # Best-effort logging even before we resolve `exp_dir`.
    enable_file_logging()
    configure_python_logging()
    # Merge in local path configuration so `${paths.*}` interpolations work in eval configs.
    project_root = _get_project_root()
    paths_cfg = _load_paths_config(project_root)
    cfg = OmegaConf.merge(paths_cfg, cfg)
    OmegaConf.resolve(cfg)

    if not cfg.checkpoint:
        raise ValueError(
            "Missing checkpoint path. Provide `checkpoint=/path/to/last.ckpt` "
            "or use `scripts/evaluation/retracker_*_first.sh <run_id|ckpt_path>`."
        )

    # Resolve paths against the repo root (Hydra may chdir into hydra.run.dir).
    cfg.exp_dir = _resolve_path(cfg.exp_dir, project_root=project_root)
    cfg.dataset_root = _resolve_path(cfg.dataset_root, project_root=project_root)
    cfg.checkpoint = _resolve_path(cfg.checkpoint, project_root=project_root)

    # Respect gpu_idx when multiple GPUs are visible to the process.
    # Must happen before any `.cuda()` calls.
    if torch.cuda.is_available():
        infos = cuda_mem_info()
        if infos and 0 <= int(cfg.gpu_idx) < len(infos):
            min_free_gb = float(os.getenv("RETRACKER_MIN_CUDA_FREE_GB", "2.0"))
            free_gb = infos[int(cfg.gpu_idx)]["free_bytes"] / (1024**3)
            if free_gb < min_free_gb:
                msg = (
                    "ERROR: Insufficient free CUDA memory for evaluation.\n"
                    f"  gpu_idx={cfg.gpu_idx}\n"
                    f"  min_free_gb={min_free_gb:.2f}\n"
                    "\n"
                    "Visible CUDA memory:\n"
                    f"{format_cuda_mem_info(infos)}\n"
                    "\n"
                    "Tip: free GPU memory, choose a different gpu_idx, or run on CPU by unsetting CUDA_VISIBLE_DEVICES."
                )
                CONSOLE.print(msg, markup=False)
                raise SystemExit(2)
        torch.cuda.set_device(cfg.gpu_idx)

    interp_shape = cfg.interp_shape
    if isinstance(interp_shape, (int, float)):
        interp_shape = (int(interp_shape), int(interp_shape))
    elif not isinstance(interp_shape, tuple):
        interp_shape = tuple(int(x) for x in interp_shape)
    if len(interp_shape) != 2:
        raise ValueError(f"interp_shape must be a 2-tuple/list, got: {cfg.interp_shape!r}")

    coarse_resolution = getattr(cfg, "coarse_resolution", (512, 512))
    if isinstance(coarse_resolution, (int, float)):
        coarse_resolution = (int(coarse_resolution), int(coarse_resolution))
    elif not isinstance(coarse_resolution, tuple):
        coarse_resolution = tuple(int(x) for x in coarse_resolution)
    if len(coarse_resolution) != 2:
        raise ValueError(
            f"coarse_resolution must be a 2-tuple/list, got: {cfg.coarse_resolution!r}"
        )

    tapvid_resize_to = getattr(cfg, "tapvid_resize_to", (256, 256))
    if tapvid_resize_to is not None:
        if isinstance(tapvid_resize_to, (int, float)):
            tapvid_resize_to = (int(tapvid_resize_to), int(tapvid_resize_to))
        elif not isinstance(tapvid_resize_to, tuple):
            tapvid_resize_to = tuple(int(x) for x in tapvid_resize_to)
        if len(tapvid_resize_to) != 2:
            raise ValueError(f"tapvid_resize_to must be a 2-tuple/list, got: {cfg.tapvid_resize_to!r}")

    # Creating the experiment directory if it doesn't exist.
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # Mirror CONSOLE output into a per-run log file under the experiment directory.
    log_path = Path(cfg.exp_dir) / "logs" / "retracker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    enabled = enable_file_logging(log_file=log_path)
    if enabled is not None:
        CONSOLE.print(f"[dim]Logging to: {enabled}[/dim]")

    # Saving the experiment configuration to a .yaml file in the experiment directory
    shard_suffix = f"_shard{cfg.shard_id}" if cfg.num_shards > 1 else ""
    cfg_file = os.path.join(cfg.exp_dir, f"expconfig{shard_suffix}.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    evaluator = PublicEvaluator(
        cfg.exp_dir,
        load_dump=cfg.load_dump,
        save_dump=cfg.save_dump,
        skip_if_done=cfg.skip_if_done,
        save_video=cfg.save_video,
    )

    if cfg.method_name in {"retracker"}:
        engine = ReTrackerEngine(
            retracker_model=None,
            ckpt_path=cfg.checkpoint,
            interp_shape=interp_shape,
            enable_highres_inference=bool(getattr(cfg, "enable_highres_inference", False)),
            coarse_resolution=coarse_resolution,
        )
    else:
        raise ValueError(
            f"Unsupported method_name={cfg.method_name!r}. "
            "Only 'retracker' is supported in the open-source evaluation CLI."
        )

    if torch.cuda.is_available():
        engine.model = engine.model.cuda()

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    if "tapvid" in cfg.dataset_name:
        dataset_type = cfg.dataset_name.split("_")[1]
        if dataset_type == "davis":
            data_root = os.path.join(cfg.dataset_root, "tapvid_davis", "tapvid_davis.pkl")
        elif dataset_type == "kinetics":
            data_root = os.path.join(
                cfg.dataset_root, "kinetics/kinetics-dataset/k700-2020/tapvid_kinetics"
            )
        elif dataset_type == "robotap":
            data_root = os.path.join(cfg.dataset_root, "tapvid_robotap")
        elif dataset_type == "stacking":
            data_root = os.path.join(cfg.dataset_root, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl")
        elif dataset_type == "retrack":
            data_root = os.path.join(cfg.dataset_root, "retrack.pkl")
        test_dataset = TapVidDataset(
            dataset_type=dataset_type,
            data_root=data_root,
            resize_to=tapvid_resize_to,
            queried_first=not "strided" in cfg.dataset_name,
            fast_eval=cfg.fast_eval,
        )
    # elif cfg.dataset_name == "dynamic_replica":
    #     test_dataset = DynamicReplicaDataset(sample_len=300, only_first_n_samples=1)

    # Sequence-level sharding for multi-GPU evaluation (run multiple processes).
    if cfg.num_shards > 1:
        if cfg.shard_id < 0 or cfg.shard_id >= cfg.num_shards:
            raise ValueError(f"Invalid shard_id={cfg.shard_id} for num_shards={cfg.num_shards}")
        indices = list(range(cfg.shard_id, len(test_dataset), cfg.num_shards))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=curr_collate_fn,
    )

    # Timing and conducting the evaluation
    import time

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    # Engine API: (video [B,T,C,H,W], queries [B,N,D=3])
    evaluate_result = evaluator.evaluate(
        engine,
        dataset_name=cfg.dataset_name,
        dataloader=test_dataloader,
        visualize_every=cfg.visualize_every,
    )
    end = time.time()
    CONSOLE.print(f"[dim]elapsed={end - start:.3f}s[/dim]")

    # Persist both per-seq metrics (written by PublicEvaluator) and aggregated metrics.
    full_metrics = evaluate_result
    avg_metrics = full_metrics.get("avg", {})
    avg_metrics = {k: float(v) for k, v in avg_metrics.items()}
    avg_metrics["time"] = float(end - start)
    avg_metrics["interp_shape"] = [int(interp_shape[0]), int(interp_shape[1])]
    avg_metrics["enable_highres_inference"] = bool(getattr(cfg, "enable_highres_inference", False))
    avg_metrics["coarse_resolution"] = [int(coarse_resolution[0]), int(coarse_resolution[1])]
    if tapvid_resize_to is None:
        avg_metrics["tapvid_resize_to"] = None
    else:
        avg_metrics["tapvid_resize_to"] = [int(tapvid_resize_to[0]), int(tapvid_resize_to[1])]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        avg_metrics["gpu_peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / (1024**2))
        avg_metrics["gpu_peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / (1024**2))

    result_file = os.path.join(cfg.exp_dir, f"result_eval_{cfg.dataset_name}{shard_suffix}.json")
    full_file = os.path.join(cfg.exp_dir, f"metrics_eval_{cfg.dataset_name}{shard_suffix}.json")
    CONSOLE.print(f"[cyan]evaluate_result[/cyan] {avg_metrics}")
    CONSOLE.print(f"[cyan]Dumping eval results to[/cyan] {result_file}.")
    with open(result_file, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    with open(full_file, "w") as f:
        json.dump(full_metrics, f, indent=2)



@hydra.main(config_path="../../configs/eval", config_name="default_config_eval", version_base="1.1")
def evaluate(cfg: DefaultConfig) -> None:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()

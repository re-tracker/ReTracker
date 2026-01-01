"""Builder for tracking/video datasets.

Keep imports lazy: many datasets have optional heavyweight dependencies. This
module should be importable as long as the *training stack* is installed.
"""

from __future__ import annotations

from typing import List, Optional

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from retracker.utils.rich_utils import CONSOLE

def build_tracking_datasets(
    config: DictConfig,
    mode: str = 'train',
    rank: int = 0,
    world_size: int = 1,
) -> List[Dataset]:
    """
    Build tracking/video datasets based on configuration.

    Args:
        config: Dataset configuration (video_config.dataset section)
        mode: 'train' or 'val'
        rank: Global rank for distributed training
        world_size: Total number of processes

    Returns:
        List of Dataset objects
    """
    datasets = []

    # Handle both list and single string for dataset_name
    dataset_names = config.dataset_name
    if isinstance(dataset_names, (ListConfig, list, tuple)):
        dataset_names = list(dataset_names)  # Convert to Python list
    else:
        dataset_names = [dataset_names]

    for dataset_name in dataset_names:
        dataset_name_lower = str(dataset_name).lower()

        if dataset_name_lower == "dummy":
            ds = _build_dummy(config, mode, rank, world_size)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'flyingthings':
            ds = _build_flyingthings(config, mode, rank, world_size)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'panning_movie':
            ds = _build_panning_movie(config, mode)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'kubrics':
            ds = _build_kubrics(config, mode)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'pointodyssey':
            ds = _build_pointodyssey(config, mode, rank, world_size)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'cotracker3_kubric':
            ds = _build_cotracker3_kubric(config, mode)
            if ds:
                datasets.append(ds)

        elif dataset_name_lower == 'k_epic':
            ds = _build_k_epic(config, mode)
            if ds:
                datasets.append(ds)
        else:
            CONSOLE.print(f"[yellow]Unknown tracking dataset: {dataset_name}[/yellow]")

    CONSOLE.print(f"[dim][rank:{rank}] Built {len(datasets)} tracking dataset(s): {dataset_names}[/dim]")
    return datasets


def _build_dummy(config: DictConfig, mode: str, rank: int, world_size: int) -> Optional[Dataset]:
    """Build a synthetic dataset for smoke tests."""
    _ = mode, rank, world_size
    from retracker.data.datasets.dummy_tracking import DummyTrackingDataset

    # Reuse common config fields when present.
    seq_len = int(config.get("s", config.get("s_p", 8)))
    n_points = int(config.get("n_per_image", config.get("n_per_image_p", 128)))
    crop = config.get("crop_size", [64, 64])
    h, w = int(crop[0]), int(crop[1])

    return DummyTrackingDataset(
        length=64,
        sequence_length=seq_len,
        num_points=n_points,
        image_size=(h, w),
        channels=3,
        seed=int(config.get("seed", 0)),
    )


def _build_flyingthings(config: DictConfig, mode: str, rank: int, world_size: int) -> Optional[Dataset]:
    """Build FlyingThings dataset."""
    from retracker.data.datasets.flyingthings import FlyingThingsDataset

    dset = 'TRAIN' if mode == 'train' else 'TEST'
    return FlyingThingsDataset(
        dataset_location=config.data_root,
        dset=dset,
        subset=config.subset,
        mode=config.get('mode', 'videos'),
        use_augs=config.use_augs,
        N=config.n_per_image,
        S_load=config.s_frames,
        S=config.s,
        crop_size=config.crop_size,
        version=config.version,
        occ_version=config.occ_version,
        force_twice_vis=config.force_twice_vis,
        force_last_vis=config.force_last_vis,
        force_all_inb=config.force_all_inb,
        max_occ=config.max_occ,
        global_rank=rank,
        world_size=world_size,
        real_batch_size=config.get('real_batch_size', 1),
    )


def _build_panning_movie(config: DictConfig, mode: str) -> Optional[Dataset]:
    """Build Panning Movie (KubricData) dataset."""
    try:
        from retracker.data.datasets.kubric_movie import KubricData
    except ModuleNotFoundError as exc:
        CONSOLE.print(
            "[yellow]KubricData not available. Install TensorFlow + TFDS to enable panning_movie.[/yellow]\n"
            f"[dim]{exc}[/dim]"
        )
        return None

    subset = config.subset_k if mode == 'train' else config.get('subset_k_val', 'train[99%:]')
    try:
        return KubricData(
            data_dir=config.data_root_k,
            batch_size=1,  # Batch size is handled by DataLoader
            S=config.s_k,
            S_load=config.s_frames_k,
            N=config.n_per_image,
            crop_size=config.crop_size,
            subset=subset,
        )
    except (ModuleNotFoundError, NotImplementedError) as exc:
        CONSOLE.print(f"[yellow]panning_movie dataset unavailable: {exc}[/yellow]")
        return None


def _build_kubrics(config: DictConfig, mode: str) -> Optional[Dataset]:
    """Build Kubrics (MOViE) dataset."""
    from retracker.data.datasets.movi_e import MOViEDataset

    split = 'train' if mode == 'train' else 'val'
    return MOViEDataset(
        config.data_root_m,
        split,
        config.subset_m,
        config.use_augs_m,
        config.n_per_image_m,
        config.s_frames_m,
        config.s_m,
        config.crop_size,
        real_batch_size=config.get('real_batch_size', 1),
    )


def _build_pointodyssey(config: DictConfig, mode: str, rank: int, world_size: int) -> Optional[Dataset]:
    """Build PointOdyssey dataset."""
    from retracker.data.datasets.pointodyssey import PointOdysseyDataset

    dset = 'train' if mode == 'train' else 'val'
    subset = config.subset_p if mode == 'train' else config.get('subset_p_val', 'sample')
    return PointOdysseyDataset(
        dataset_location=config.data_root_p,
        dset=dset,
        mode=config.get('mode', 'videos'),
        use_augs=config.use_augs_p,
        N=config.n_per_image_p,
        S=config.s_p,
        crop_size=config.crop_size,
        subset=subset,
        global_rank=rank,
        world_size=world_size,
        real_batch_size=config.get('real_batch_size', 1),
    )


def _build_cotracker3_kubric(config: DictConfig, mode: str) -> Optional[Dataset]:
    """Build CoTracker3 Kubric dataset."""
    from retracker.data.datasets.kubric_movif_dataset import KubricMovifDataset

    split = 'train' if mode == 'train' else 'val'
    return KubricMovifDataset(
        data_root=config.data_root_c3k,
        crop_size=config.crop_size,
        seq_len=config.S_c3k,
        traj_per_sample=config.N_per_image_c3k,
        split=split,
    )


def _build_k_epic(config: DictConfig, mode: str) -> Optional[Dataset]:
    """Build K-EPIC dataset."""
    from retracker.data.datasets.cotracker_movie import KubricMovifEpicDataset

    split = 'train' if mode == 'train' else 'val'
    return KubricMovifEpicDataset(
        data_root=config.data_root_e,
        crop_size=config.crop_size,
        seq_len=config.s_e,
        traj_per_sample=config.n_per_image,
        split=split,
    )

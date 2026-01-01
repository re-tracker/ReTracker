"""Builder for matching/pairs datasets."""
import random
from os import path as osp
from typing import List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import DictConfig, ListConfig

from retracker.training.utils.augment import build_augmentor
from retracker.training.utils.dataloader import get_local_split
from retracker.data.datasets.Megadepth import MegaDepthDataset
from retracker.utils.rich_utils import CONSOLE


def _to_list(value):
    """Convert OmegaConf ListConfig or single value to Python list."""
    if isinstance(value, (ListConfig, list, tuple)):
        return list(value)
    return [value]


def build_matching_datasets(
    config: DictConfig,
    mode: str = 'train',
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Tuple[List[Dataset], float]:
    """
    Build matching/pairs datasets based on configuration.

    Args:
        config: Dataset configuration (matching_config section)
        mode: 'train', 'val', or 'test'
        rank: Global rank for distributed training
        world_size: Total number of processes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (List of Dataset objects, sample_ratio)
    """
    dataset_cfg = config.DATASET
    trainer_cfg = config.get('TRAINER', {})

    # Determine data sources and paths based on mode
    if mode == 'train':
        data_root = dataset_cfg.TRAIN_DATA_ROOT
        data_sources = _to_list(dataset_cfg.TRAIN_DATA_SOURCE)
        npz_roots = _to_list(dataset_cfg.TRAIN_NPZ_ROOT)
        list_paths = _to_list(dataset_cfg.TRAIN_LIST_PATH)
        sample_ratios = _to_list(dataset_cfg.get('TRAIN_DATA_SAMPLE_RATIO', [1.0]))
        min_overlap_score = dataset_cfg.MIN_OVERLAP_SCORE_TRAIN
    elif mode == 'val':
        data_root = dataset_cfg.VAL_DATA_ROOT
        data_sources = _to_list(dataset_cfg.VAL_DATA_SOURCE)
        npz_roots = _to_list(dataset_cfg.VAL_NPZ_ROOT)
        list_paths = _to_list(dataset_cfg.VAL_LIST_PATH)
        sample_ratios = [1.0]
        min_overlap_score = dataset_cfg.MIN_OVERLAP_SCORE_TEST
    else:  # test
        data_root = dataset_cfg.TEST_DATA_ROOT
        data_sources = _to_list(dataset_cfg.TEST_DATA_SOURCE)
        npz_roots = _to_list(dataset_cfg.TEST_NPZ_ROOT)
        list_paths = _to_list(dataset_cfg.TEST_LIST_PATH)
        sample_ratios = [1.0]
        min_overlap_score = dataset_cfg.MIN_OVERLAP_SCORE_TEST

    # Build augmentor
    augment_fn = build_augmentor(dataset_cfg.AUGMENTATION_TYPE) if mode == 'train' else None

    # Build datasets
    all_datasets = []
    total_sample_ratio = 0.0

    for idx, (data_source, npz_root, list_path) in enumerate(zip(data_sources, npz_roots, list_paths)):
        sample_ratio = sample_ratios[idx] if idx < len(sample_ratios) else 1.0
        total_sample_ratio += sample_ratio

        # Load scene list
        with open(list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        # Distribute scenes across ranks for training
        if mode == 'train':
            local_npz_names = get_local_split(npz_names, world_size, rank, seed)
        else:
            local_npz_names = npz_names

        CONSOLE.print(f"[dim][rank:{rank}] Loading {mode} {data_source}: {len(local_npz_names)} scene(s)[/dim]")

        # Build individual datasets for each scene
        datasets = _build_scene_datasets(
            data_root=data_root,
            data_source=data_source,
            npz_names=local_npz_names,
            npz_root=npz_root,
            sample_ratio=sample_ratio,
            config=dataset_cfg,
            augment_fn=augment_fn,
            mode=mode,
            min_overlap_score=min_overlap_score,
            rank=rank,
        )
        all_datasets.extend(datasets)

    # Shuffle for training
    if mode == 'train':
        random.shuffle(all_datasets)

    CONSOLE.print(f"[dim][rank:{rank}] Built {len(all_datasets)} matching dataset(s)[/dim]")
    return all_datasets, total_sample_ratio / len(data_sources) if data_sources else 1.0


def _build_scene_datasets(
    data_root: str,
    data_source: str,
    npz_names: List[str],
    npz_root: str,
    sample_ratio: float,
    config: DictConfig,
    augment_fn,
    mode: str,
    min_overlap_score: float,
    rank: int,
) -> List[Dataset]:
    """Build datasets for each scene."""
    datasets = []
    npz_names = [f'{n}.npz' for n in npz_names]

    for npz_name in tqdm(
        npz_names,
        desc=f'[rank:{rank}] loading {mode} {data_source}',
        disable=int(rank) != 0
    ):
        npz_path = osp.join(npz_root, npz_name)

        if 'megadepth' in data_source.lower():
            dataset = MegaDepthDataset(
                data_root,
                npz_path,
                mode=mode,
                min_overlap_score=min_overlap_score,
                img_resize=config.MGDPT_IMG_RESIZE,
                df=config.MGDPT_DF,
                img_padding=config.MGDPT_IMG_PAD,
                depth_padding=config.MGDPT_DEPTH_PAD,
                augment_fn=augment_fn,
                coarse_scale=None,
                testNpairs=config.TEST_N_PAIRS,
                fp16=config.FP16,
                load_origin_rgb=config.LOAD_ORIGIN_RGB,
                read_gray=config.READ_GRAY,
                fix_bias=False,
                read_depth=True,
                sample_ratio=sample_ratio,
            )
            datasets.append(dataset)

        elif 'scannet' in data_source.lower():
            # ScanNet dataset support can be added here
            # For now, we use MegaDepth-like interface
            dataset = MegaDepthDataset(
                data_root,
                npz_path,
                mode=mode,
                min_overlap_score=min_overlap_score,
                img_resize=[config.SCAN_IMG_RESIZEX, config.SCAN_IMG_RESIZEY],
                df=config.MGDPT_DF,
                img_padding=config.MGDPT_IMG_PAD,
                depth_padding=config.MGDPT_DEPTH_PAD,
                augment_fn=augment_fn,
                coarse_scale=None,
                testNpairs=config.TEST_N_PAIRS,
                fp16=config.FP16,
                load_origin_rgb=config.LOAD_ORIGIN_RGB,
                read_gray=config.READ_GRAY,
                fix_bias=False,
                read_depth=True,
                sample_ratio=sample_ratio,
            )
            datasets.append(dataset)

        else:
            CONSOLE.print(f"[yellow]Unknown matching data source: {data_source}[/yellow]")

    return datasets

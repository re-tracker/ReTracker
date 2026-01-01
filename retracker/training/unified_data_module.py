"""Unified DataModule for both matching and tracking tasks.

This module provides a single LightningDataModule that can handle:
- Matching datasets (MegaDepth, ScanNet)
- Tracking datasets (FlyingThings, PointOdyssey, Kubrics, etc.)
- Mixed training with both task types

Usage:
    data_module = UnifiedDataModule(config)
    trainer.fit(model, data_module)

Sample Ratio Configuration:
    The final sample_ratio for each dataset is computed as:
        final_ratio = task_ratio * dataset_ratio

    Where:
    - task_ratio: Weight for the task type (matching/tracking), from unified_config.task_ratios
    - dataset_ratio: Weight for individual dataset within its task type

    Example config:
        unified_config:
            task_ratios:
                matching: 0.3
                tracking: 0.7

        video_config:
            dataset_ratios:  # Per-dataset weights (will be normalized)
                panning_movie: 1.0
                pointodyssey: 0.5
"""
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Dict, Any, List, Optional, Union
from omegaconf import DictConfig, OmegaConf, ListConfig
import lightning.pytorch as pl
from torch import distributed as dist

from retracker.data.datasets.unified_dataset.dataclasses_utils import unified_matching_tracking_dataset
from retracker.data.datasets.sampler import TaskSynchronizedBatchSampler
from retracker.training.dataset_builders import build_tracking_datasets, build_matching_datasets
from retracker.utils.rich_utils import CONSOLE


class UnifiedDatasetWrapper(Dataset):
    """
    Wrapper that unifies output format for any sequential vision dataset.
    Handles both matching (image pairs) and tracking (video sequences) data.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        config: Union[DictConfig, Dict],
        sample_ratio: float = 1.0,
        task_type: str = 'tracking',
        dataset_name: str = 'unknown',
    ):
        """
        Args:
            original_dataset: The underlying dataset
            config: Configuration for data processing
            sample_ratio: Sampling ratio for this dataset (already computed as task_ratio * dataset_ratio)
            task_type: 'matching' or 'tracking'
            dataset_name: Name of the dataset for logging
        """
        self.original_dataset = original_dataset
        self.config = config if isinstance(config, DictConfig) else OmegaConf.create(config)
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.sample_ratio = sample_ratio

    def __len__(self) -> int:
        return len(self.original_dataset)  # type: ignore

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_data = self.original_dataset[index]
        unified_data = unified_matching_tracking_dataset(raw_data, self.config)
        unified_data['task_type'] = self.task_type
        return unified_data


def unified_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for unified data batches.

    Handles homogeneous batches where all items come from the same task type.
    The TaskSynchronizedBatchSampler ensures this property.

    For safety, we only collate keys that exist in ALL items (intersection).
    """
    if len(batch) == 0:
        return {}

    # Find keys that exist in ALL items (intersection)
    common_keys = set(batch[0].keys())
    for item in batch[1:]:
        common_keys &= set(item.keys())

    collated = {}
    for key in common_keys:
        items = [item[key] for item in batch]

        if isinstance(items[0], torch.Tensor):
            try:
                collated[key] = torch.stack(items, dim=0)
            except RuntimeError:
                # Shapes don't match, keep as list
                collated[key] = items
        else:
            # Non-tensor data (strings, metadata, etc.)
            collated[key] = items

    # Add task_type indicator based on data content
    if 'image0' in batch[0]:
        collated['task_type'] = ['matching'] * len(batch)
    else:
        collated['task_type'] = ['tracking'] * len(batch)

    return collated


class UnifiedDataModule(pl.LightningDataModule):
    """
    Unified DataModule for matching and tracking tasks.

    This replaces the previous design of separate VideoDataModule and
    MultiSceneDataModule combined by SequencialDataModule.

    Config structure:
        unified_config:
            training_tasks: ['matching', 'tracking']  # Which tasks to include
            batch_size: 8
            num_workers: 8
            pin_memory: true
            n_samples_per_subset: 25
            data_sampler: 'scene_balance'
            seed: 42

            # Task-level sampling ratios (optional)
            # Controls the proportion of samples from each task type
            task_ratios:
                matching: 0.3   # 30% of samples from matching
                tracking: 0.7   # 70% of samples from tracking

        tracking_config/video_config:
            # Per-dataset sampling ratios (optional)
            # Controls the proportion within tracking datasets
            dataset_ratios:
                panning_movie: 1.0
                pointodyssey: 0.5
                flyingthings: 0.3
            dataset:
                dataset_name: ['panning_movie', 'pointodyssey']
                # ... tracking-specific configs

        matching_config:
            # Per-dataset sampling ratios (optional)
            dataset_ratios:
                MegaDepth: 1.0
                ScanNet: 0.5
            DATASET:
                TRAIN_DATA_SOURCE: ['MegaDepth']
                # ... matching-specific configs
    """

    def __init__(
        self,
        unified_config: Dict[str, Any],
        tracking_config: Optional[Dict[str, Any]] = None,
        matching_config: Optional[Dict[str, Any]] = None,
        # Backward compatibility: accept video_config as alias for tracking_config
        video_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Backward compatibility: use video_config if tracking_config is not provided
        if tracking_config is None and video_config is not None:
            tracking_config = video_config

        # Convert to OmegaConf for dot notation access
        self.unified_config = OmegaConf.create(unified_config) if not isinstance(unified_config, DictConfig) else unified_config
        self.tracking_config = OmegaConf.create(tracking_config) if tracking_config and not isinstance(tracking_config, DictConfig) else tracking_config
        self.matching_config = OmegaConf.create(matching_config) if matching_config and not isinstance(matching_config, DictConfig) else matching_config

        # Extract common parameters
        self.training_tasks = self._to_list(self.unified_config.get('training_tasks', ['tracking']))
        self.batch_size = self.unified_config.get('batch_size', 1)
        self.num_workers = self.unified_config.get('num_workers', 4)
        self.pin_memory = self.unified_config.get('pin_memory', True)
        self.n_samples_per_subset = self.unified_config.get('n_samples_per_subset', 25)
        self.data_sampler = self.unified_config.get('data_sampler', 'scene_balance')
        self.seed = self.unified_config.get('seed', 42)

        # Task-level ratios (defaults to equal weight if not specified)
        task_ratios_cfg = self.unified_config.get('task_ratios', {})
        self.task_ratios = {
            'matching': task_ratios_cfg.get('matching', 1.0),
            'tracking': task_ratios_cfg.get('tracking', 1.0),
        }

        # Sampler settings
        self.subset_replacement = self.unified_config.get('subset_replacement', True)
        self.shuffle = self.unified_config.get('shuffle', True)
        self.repeat = self.unified_config.get('repeat', 1)

        # DataLoader params
        self.train_loader_params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }

        # State
        self.train_dataset = None
        self.val_dataset = None
        self._train_batch_sampler = None

    @staticmethod
    def _to_list(value):
        """Convert OmegaConf ListConfig or single value to Python list."""
        if isinstance(value, (ListConfig, list, tuple)):
            return list(value)
        return [value]

    def prepare_data(self):
        """Download or prepare data (called only on rank 0)."""
        pass  # Data is assumed to be already available

    def _get_dataset_ratios(self, config: DictConfig, task_type: str) -> Dict[str, float]:
        """
        Get per-dataset ratios from config.

        Args:
            config: tracking_config or matching_config
            task_type: 'tracking' or 'matching'

        Returns:
            Dict mapping dataset_name -> ratio
        """
        if config is None:
            return {}

        # Try to get dataset_ratios from config
        ratios = config.get('dataset_ratios', {})
        if ratios:
            return dict(ratios)

        # Fallback: try to get from DATASET.TRAIN_DATA_SAMPLE_RATIO for matching
        if task_type == 'matching' and 'DATASET' in config:
            dataset_cfg = config.DATASET
            sources = self._to_list(dataset_cfg.get('TRAIN_DATA_SOURCE', []))
            sample_ratios = self._to_list(dataset_cfg.get('TRAIN_DATA_SAMPLE_RATIO', [1.0]))

            ratios = {}
            for i, src in enumerate(sources):
                ratio = sample_ratios[i] if i < len(sample_ratios) else 1.0
                ratios[str(src).lower()] = ratio
            return ratios

        return {}

    def _get_dataset_name(self, dataset: Dataset, index: int, dataset_names: List[str]) -> str:
        """Try to determine the dataset name."""
        # If we have a list of dataset names from config, use index
        if index < len(dataset_names):
            return str(dataset_names[index]).lower()

        # Try to get name from dataset class
        class_name = dataset.__class__.__name__.lower()

        # Common mappings
        name_mapping = {
            'kubricdata': 'panning_movie',
            'moviedataset': 'kubrics',
            'flyingthingsdataset': 'flyingthings',
            'pointodysseydataset': 'pointodyssey',
            'kubricmovifdataset': 'cotracker3_kubric',
            'kubricmovifepic': 'k_epic',
            'megadepthdataset': 'megadepth',
        }

        for key, value in name_mapping.items():
            if key in class_name:
                return value

        return f'dataset_{index}'

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation."""
        if self.train_dataset is not None and self.val_dataset is not None:
            return  # Already set up

        # Get distributed info
        try:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        except (RuntimeError, ValueError):
            # ValueError happens when torch.distributed is available but the default
            # process group has not been initialized (common in single-process runs).
            world_size = 1
            rank = 0

        CONSOLE.print(
            f"[dim][rank:{rank}] Setting up UnifiedDataModule with tasks: {self.training_tasks}[/dim]"
        )
        CONSOLE.print(f"[dim][rank:{rank}] Task ratios: {self.task_ratios}[/dim]")

        train_datasets = []
        val_datasets = []

        # Build tracking datasets
        if 'tracking' in self.training_tasks and self.tracking_config is not None:
            tracking_task_ratio = self.task_ratios.get('tracking', 1.0)
            tracking_dataset_ratios = self._get_dataset_ratios(self.tracking_config, 'tracking')

            # Get dataset names from config
            dataset_names = self._to_list(self.tracking_config.dataset.get('dataset_name', []))

            tracking_train = build_tracking_datasets(
                config=self.tracking_config.dataset,
                mode='train',
                rank=rank,
                world_size=world_size,
            )

            # Wrap with unified output format and computed sample_ratios
            for i, ds in enumerate(tracking_train):
                ds_name = self._get_dataset_name(ds, i, dataset_names)
                ds_ratio = tracking_dataset_ratios.get(ds_name, 1.0)
                final_ratio = tracking_task_ratio * ds_ratio

                if rank == 0:
                    CONSOLE.print(
                        f"  Tracking dataset '{ds_name}': "
                        f"task_ratio={tracking_task_ratio:.2f} × "
                        f"ds_ratio={ds_ratio:.2f} = "
                        f"final_ratio={final_ratio:.2f}"
                    )

                wrapped = UnifiedDatasetWrapper(
                    ds, self.tracking_config,
                    sample_ratio=final_ratio,
                    task_type='tracking',
                    dataset_name=ds_name,
                )
                train_datasets.append(wrapped)

            # Validation datasets (use ratio=1.0 for validation)
            tracking_val = build_tracking_datasets(
                config=self.tracking_config.dataset,
                mode='val',
                rank=rank,
                world_size=world_size,
            )
            for i, ds in enumerate(tracking_val):
                ds_name = self._get_dataset_name(ds, i, dataset_names)
                wrapped = UnifiedDatasetWrapper(
                    ds, self.tracking_config,
                    sample_ratio=1.0,
                    task_type='tracking',
                    dataset_name=ds_name,
                )
                val_datasets.append(wrapped)

        # Build matching datasets
        if 'matching' in self.training_tasks and self.matching_config is not None:
            matching_task_ratio = self.task_ratios.get('matching', 1.0)
            matching_dataset_ratios = self._get_dataset_ratios(self.matching_config, 'matching')

            # Get dataset names from config
            dataset_names = self._to_list(self.matching_config.DATASET.get('TRAIN_DATA_SOURCE', []))

            matching_train, _ = build_matching_datasets(
                config=self.matching_config,
                mode='train',
                rank=rank,
                world_size=world_size,
                seed=self.seed,
            )

            # Wrap with unified output format and computed sample_ratios
            for i, ds in enumerate(matching_train):
                ds_name = self._get_dataset_name(ds, i, dataset_names)
                ds_ratio = matching_dataset_ratios.get(ds_name, 1.0)
                final_ratio = matching_task_ratio * ds_ratio

                if rank == 0:
                    CONSOLE.print(
                        f"  Matching dataset '{ds_name}': "
                        f"task_ratio={matching_task_ratio:.2f} × "
                        f"ds_ratio={ds_ratio:.2f} = "
                        f"final_ratio={final_ratio:.2f}"
                    )

                wrapped = UnifiedDatasetWrapper(
                    ds, self.matching_config,
                    sample_ratio=final_ratio,
                    task_type='matching',
                    dataset_name=ds_name,
                )
                train_datasets.append(wrapped)

            # Validation datasets
            matching_val, _ = build_matching_datasets(
                config=self.matching_config,
                mode='val',
                rank=rank,
                world_size=world_size,
                seed=self.seed,
            )
            for i, ds in enumerate(matching_val):
                ds_name = self._get_dataset_name(ds, i, dataset_names)
                wrapped = UnifiedDatasetWrapper(
                    ds, self.matching_config,
                    sample_ratio=1.0,
                    task_type='matching',
                    dataset_name=ds_name,
                )
                val_datasets.append(wrapped)

        # Combine all datasets
        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
        else:
            raise ValueError("No training datasets were created. Check your configuration.")

        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
        else:
            # Use a subset of training data for validation if no val datasets
            CONSOLE.print("[yellow]No validation datasets found, using training data for validation[/yellow]")
            self.val_dataset = self.train_dataset

        CONSOLE.print(
            f"[rank:{rank}] Setup complete. "
            f"Train: {len(self.train_dataset)} samples, "
            f"Val: {len(self.val_dataset)} samples"
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with task-synchronized batch sampling."""
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        global_rank = getattr(self.trainer, "global_rank", 0) if self.trainer else 0

        # Log re-init (should only happen once per training)
        CONSOLE.print(
            f"[dim][rank:{global_rank}/{world_size}] Creating train dataloader "
            f"(tasks: {self.training_tasks})[/dim]"
        )

        # Calculate local samples per subset
        local_n_samples = self.n_samples_per_subset // world_size

        # Ensure consistent seed across all ranks
        safe_seed = int(self.seed) if self.seed is not None else 42

        # Create batch sampler
        batch_sampler = None
        if self.data_sampler == 'scene_balance':
            batch_sampler = TaskSynchronizedBatchSampler(
                data_source=self.train_dataset,
                batch_size=self.batch_size,
                n_samples_per_subset=local_n_samples,
                subset_replacement=self.subset_replacement,
                shuffle=self.shuffle,
                repeat=self.repeat,
                seed=safe_seed,
                rank=global_rank,
                world_size=world_size,
            )
            self._train_batch_sampler = batch_sampler

        # Prepare loader params
        loader_params = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }

        if batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                collate_fn=unified_collate_fn,
                batch_sampler=batch_sampler,
                **loader_params,
            )
        else:
            loader_params['batch_size'] = self.batch_size
            return DataLoader(
                self.train_dataset,
                collate_fn=unified_collate_fn,
                **loader_params,
            )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with task-synchronized batch sampling."""
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        global_rank = getattr(self.trainer, "global_rank", 0) if self.trainer else 0

        # Use fewer samples for validation
        val_n_samples = max(1, 128 // world_size)

        batch_sampler = None
        if self.data_sampler == 'scene_balance':
            batch_sampler = TaskSynchronizedBatchSampler(
                data_source=self.val_dataset,
                batch_size=self.batch_size,
                n_samples_per_subset=val_n_samples,
                subset_replacement=self.subset_replacement,
                shuffle=False,  # No shuffle for validation
                repeat=1,
                seed=self.seed,
                rank=global_rank,
                world_size=world_size,
            )

        loader_params = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }

        if batch_sampler is not None:
            return DataLoader(
                self.val_dataset,
                collate_fn=unified_collate_fn,
                batch_sampler=batch_sampler,
                **loader_params,
            )
        else:
            loader_params['batch_size'] = self.batch_size
            return DataLoader(
                self.val_dataset,
                collate_fn=unified_collate_fn,
                **loader_params,
            )

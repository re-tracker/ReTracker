"""Sampling utilities for training datasets.

The project historically used a ConcatDataset + custom sampler approach.
In the unified datamodule, we additionally need to ensure that a batch contains
examples from only one task type (matching OR tracking). The simplest way to
guarantee this is to sample batches from a single sub-dataset at a time.
"""

from __future__ import annotations

from typing import Iterator, List

import torch
from torch.utils.data import ConcatDataset, Sampler


def _seed_for_epoch(base_seed: int, epoch: int, rank: int) -> int:
    # Make per-rank streams deterministic but distinct.
    return int(base_seed) + int(epoch) * 1000 + int(rank)


class RandomConcatSampler(Sampler[int]):
    """Random sampler for a ConcatDataset.

    At each epoch, draws `n_samples_per_subset` samples from each subset, with an
    optional per-subset `sample_ratio` multiplier.

    NOTE: This sampler yields single indices (not batches). Prefer
    :class:`TaskSynchronizedBatchSampler` when you need task-homogeneous batches.
    """

    def __init__(
        self,
        data_source: ConcatDataset,
        n_samples_per_subset: int,
        subset_replacement: bool = True,
        shuffle: bool = True,
        repeat: int = 1,
        seed: int | None = None,
        rank: int = 0,
    ) -> None:
        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source must be a torch.utils.data.ConcatDataset")

        self.data_source = data_source
        self.n_samples_per_subset = int(n_samples_per_subset)
        self.subset_replacement = bool(subset_replacement)
        self.shuffle = bool(shuffle)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

        self.base_seed = int(seed) if seed is not None else 0
        self.rank = int(rank)
        self.epoch = 0

        self._n_samples = self._compute_n_samples()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _compute_n_samples(self) -> int:
        total = 0
        for ds in self.data_source.datasets:
            ratio = float(getattr(ds, "sample_ratio", 1.0))
            total += int(ratio * self.n_samples_per_subset)
        return total * self.repeat

    def __len__(self) -> int:
        return self._n_samples

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator()
        gen.manual_seed(_seed_for_epoch(self.base_seed, self.epoch, self.rank))

        indices: List[torch.Tensor] = []
        for d_idx, ds in enumerate(self.data_source.datasets):
            low = 0 if d_idx == 0 else self.data_source.cumulative_sizes[d_idx - 1]
            high = self.data_source.cumulative_sizes[d_idx]

            ratio = float(getattr(ds, "sample_ratio", 1.0))
            n_samples = int(ratio * self.n_samples_per_subset)
            if n_samples <= 0:
                continue

            if self.subset_replacement:
                rand = torch.randint(low, high, (n_samples,), generator=gen, dtype=torch.int64)
            else:
                len_subset = len(ds)
                perm = torch.randperm(len_subset, generator=gen, dtype=torch.int64) + low
                if len_subset >= n_samples:
                    rand = perm[:n_samples]
                else:
                    pad = torch.randint(low, high, (n_samples - len_subset,), generator=gen, dtype=torch.int64)
                    rand = torch.cat([perm, pad], dim=0)

            indices.append(rand)

        if not indices:
            return iter(())

        flat = torch.cat(indices, dim=0)
        if self.shuffle:
            flat = flat[torch.randperm(len(flat), generator=gen)]

        if self.repeat > 1:
            repeats = [flat]
            for _ in range(self.repeat - 1):
                r = flat.clone()
                if self.shuffle:
                    r = r[torch.randperm(len(r), generator=gen)]
                repeats.append(r)
            flat = torch.cat(repeats, dim=0)

        return iter(flat.tolist())


class TaskSynchronizedBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields task-homogeneous batches from a ConcatDataset.

    This sampler draws indices within each *sub-dataset* (i.e. each member of
    `ConcatDataset.datasets`) and groups them into batches. Because each sub-dataset
    corresponds to a single task (tracking or matching), batches are homogeneous.
    """

    def __init__(
        self,
        data_source: ConcatDataset,
        batch_size: int,
        n_samples_per_subset: int,
        subset_replacement: bool = True,
        shuffle: bool = True,
        repeat: int = 1,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source must be a torch.utils.data.ConcatDataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.n_samples_per_subset = int(n_samples_per_subset)
        self.subset_replacement = bool(subset_replacement)
        self.shuffle = bool(shuffle)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

        self.base_seed = int(seed) if seed is not None else 0
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        self._n_batches = self._compute_n_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _compute_n_batches(self) -> int:
        total_batches = 0
        for ds in self.data_source.datasets:
            ratio = float(getattr(ds, "sample_ratio", 1.0))
            n_samples = int(ratio * self.n_samples_per_subset)
            total_batches += n_samples // self.batch_size
        return total_batches * self.repeat

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[List[int]]:
        gen = torch.Generator()
        gen.manual_seed(_seed_for_epoch(self.base_seed, self.epoch, self.rank))

        all_batches: List[List[int]] = []

        for d_idx, ds in enumerate(self.data_source.datasets):
            low = 0 if d_idx == 0 else self.data_source.cumulative_sizes[d_idx - 1]
            high = self.data_source.cumulative_sizes[d_idx]

            ratio = float(getattr(ds, "sample_ratio", 1.0))
            n_samples = int(ratio * self.n_samples_per_subset)
            if n_samples <= 0:
                continue

            if self.subset_replacement:
                idxs = torch.randint(low, high, (n_samples,), generator=gen, dtype=torch.int64)
            else:
                len_subset = len(ds)
                perm = torch.randperm(len_subset, generator=gen, dtype=torch.int64) + low
                if len_subset >= n_samples:
                    idxs = perm[:n_samples]
                else:
                    pad = torch.randint(low, high, (n_samples - len_subset,), generator=gen, dtype=torch.int64)
                    idxs = torch.cat([perm, pad], dim=0)

            # Repeat sampling if requested.
            idxs_list: List[torch.Tensor] = [idxs]
            for _ in range(self.repeat - 1):
                r = idxs.clone()
                if self.shuffle:
                    r = r[torch.randperm(len(r), generator=gen)]
                idxs_list.append(r)
            idxs = torch.cat(idxs_list, dim=0) if len(idxs_list) > 1 else idxs

            # Chunk into full batches (drop remainder to keep a constant batch size).
            n_full = (len(idxs) // self.batch_size) * self.batch_size
            if n_full <= 0:
                continue
            idxs = idxs[:n_full]
            idxs = idxs.view(-1, self.batch_size)
            all_batches.extend([b.tolist() for b in idxs])

        if self.shuffle and len(all_batches) > 1:
            order = torch.randperm(len(all_batches), generator=gen).tolist()
            all_batches = [all_batches[i] for i in order]

        return iter(all_batches)


__all__ = [
    "RandomConcatSampler",
    "TaskSynchronizedBatchSampler",
]

from typing import Literal

import torch
from torch import Tensor


class MemoryManager:
    """FIFO-style memory bank used by ReTracker.

    Stores per-key tensors (features, predictions, etc.) and keeps the memory size
    bounded during long sequences via simple sampling strategies.
    """

    def __init__(self, config: dict, getter_callback_dict: dict | None = None):
        if getter_callback_dict is None:
            getter_callback_dict = {}
        self.MAX_MEMORY_SIZE: int = config["max_memory_size"]
        self.sample_method: Literal["foremost", "square", "foremost2", "foremost4"] = config[
            "sample_method"
        ]
        self._memory: DictWithSetter = DictWithSetter(getter_callback_dict)

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        if isinstance(value, DictWithSetter):
            self._memory = value
        else:
            raise ValueError("Data must be a dictionary.")

    def set_memory(
        self,
        key: str,
        value: Tensor,
        autofill: bool = False,
        detach: bool = True,
        stack_dim: int = 0,
        fill_length: int = -1,
    ):
        value = value.clone().detach() if detach else value  # assert value.requires_grad == False
        self.memory[key] = value
        if autofill:
            if fill_length == -1:
                self.memory[key] = torch.cat([value] * self.MAX_MEMORY_SIZE, dim=stack_dim)
            else:
                self.memory[key] = torch.cat([value] * fill_length, dim=stack_dim)

    def reset_memory(self, key: str):
        self.memory[key].clear()

    def reset_all_memory(self):
        self.memory.clear()

    def exists(self, key: str):
        return key in self.memory.keys()

    def push_memory(
        self,
        key: str,
        value: Tensor,
        stack_dim: int,
        detach: bool = True,
        auto_pop: bool = True,
        custom_size: int = None,
        auto_fill: bool = False,
        updated_mask: Tensor = None,
    ):
        """Push data into memory, FIFO:
        Args:
            updated_mask: True for updated
        """
        value = value.clone().detach() if detach else value  # assert value.requires_grad == False
        is_empty = self.memory.get(key, None) is None  # or len(self.memory[key]) == 0
        if is_empty:
            if auto_fill:
                self.memory[key] = torch.cat([value] * custom_size, dim=stack_dim)
            else:
                self.memory[key] = value
            return

        if updated_mask is None:
            self.memory[key] = torch.cat([value, self.memory[key]], dim=stack_dim)  # (..., F, C)
        else:
            # Handle shape mismatch when query count changes (for streaming scenarios)
            # Check if BN dimension (first dim) matches between value and memory
            value_BN = value.shape[0] if value.dim() > 0 else 1
            mem_BN = self.memory[key].shape[0] if self.memory[key].dim() > 0 else 1

            if value_BN != mem_BN:
                # Shape mismatch: extend memory to match value shape
                # This happens when new queries are added dynamically
                device = value.device
                dtype = value.dtype

                # Get memory shape info
                mem_shape = list(self.memory[key].shape)
                mem_shape[0] = value_BN  # Update BN dimension

                # Create extended memory with zeros for new queries
                extended_mem = torch.zeros(mem_shape, device=device, dtype=dtype)

                # Copy existing data (assuming first min(BN) queries are the same)
                min_BN = min(value_BN, mem_BN)
                if min_BN > 0:
                    # Copy all frames for existing queries
                    if stack_dim == 1:
                        # Memory shape: [BN, F, ...]
                        extended_mem[:min_BN, :, ...] = self.memory[key][:min_BN, :, ...]
                    else:
                        # For other stack_dim, copy along first dimension
                        extended_mem[:min_BN, ...] = self.memory[key][:min_BN, ...]

                # Replace memory with extended version
                self.memory[key] = extended_mem

            backup = self.memory[key].clone()  # ..., F, ..., C
            backup_1_N = backup.index_select(  # ..., F-1, ... ,C
                stack_dim, torch.arange(1, backup.shape[stack_dim], device=backup.device)
            )
            backup_0_N_1 = backup.index_select(
                stack_dim, torch.arange(backup.shape[stack_dim] - 1, device=backup.device)
            )
            _mem_first = self.memory[key].index_select(
                stack_dim, torch.arange(value.shape[stack_dim], device=self.memory[key].device)
            )
            new_head = torch.where(updated_mask, value, _mem_first)  # frame 0

            new_body = torch.where(updated_mask, backup_0_N_1, backup_1_N)  # frame 1-N
            self.memory[key] = torch.cat([new_head, new_body], dim=stack_dim)

        if auto_pop:
            max_size = self.MAX_MEMORY_SIZE if custom_size is None else custom_size
            while self.memory[key].size(stack_dim) > max_size:
                self.pop_memory(key, stack_dim)

    def pop_memory(self, key: str, stack_dim: int):
        """Pop value from memory, FIFO"""
        if self.memory[key] is None or self.memory[key].shape[-1] == 0:
            return

        index = torch.arange(self.memory[key].shape[stack_dim], device=self.memory[key].device)
        self.memory[key] = self.memory[key].index_select(stack_dim, index[:-1])

    def get_memory(self, key: str, default: Tensor | None = None):
        if key not in self.memory.keys():
            return default
        return self.memory[key]

    def sample_memory(
        self,
        key: str,
        stack_dim: int,
        samples_length: int = 8,
        detach: bool = True,
        default=None,
        drop_out=0,
        **kwargs,
    ):
        """return sub-dict from memory, sampled by assigned method.
        samples_length = pips_input_length - 1
        """
        is_empty = self.memory.get(key, None) is None  # or len(self.memory[key]) == 0
        if is_empty:
            return default

        sample_method = kwargs.get("sample_method", self.sample_method)
        device = self.memory[key].device
        max_memory = self.memory[key].shape[stack_dim]

        if sample_method == "square":
            sampled_idx = 2 ** torch.arange(samples_length, device=device) - 1
        elif sample_method == "foremost":
            sampled_idx = torch.arange(samples_length, device=device)
        elif sample_method == "foremost2":
            sampled_idx = torch.arange(0, samples_length * 2, step=2, device=device)
        elif sample_method == "foremost4":
            sampled_idx = torch.arange(0, samples_length * 4, step=4, device=device)
        elif sample_method == "adaptive_12":
            # Legacy strategy - fixed indices, may exceed memory bounds for large samples_length
            sampled_idx = (
                torch.Tensor(
                    [0, 1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 22, 26, 30, 34, 38, 44, 50][
                        :samples_length
                    ]
                )
                .long()
                .to(device)
            )
        elif sample_method == "balanced":
            # Balanced sampling: first half recent frames, second half distant frames
            # Adapts to available memory size and requested sample length
            sampled_idx = self._balanced_sampling(samples_length, max_memory, device)
        else:
            raise NotImplementedError(f"Unknown sample_method: {sample_method}")

        # Clamp indices to valid range to prevent out-of-bounds errors
        sampled_idx = torch.clamp(sampled_idx, 0, max_memory - 1)

        value = self.memory[key].index_select(stack_dim, sampled_idx)
        if drop_out > 0:
            assert stack_dim == 1
            if value.dim() == 3:
                mask = (
                    torch.rand_like(value[:1, :, :1], device=value.device) > drop_out
                ).float()  # 1, F, 1, 1
            elif value.dim() == 4:
                mask = (
                    torch.rand_like(value[:1, :, :1, :1], device=value.device) > drop_out
                ).float()  # 1, F, 1, 1
            value = value * mask
        return value.detach() if detach else value

    def _balanced_sampling(self, samples_length: int, max_memory: int, device) -> Tensor:
        """
        Generate balanced sampling indices: first half recent, second half distant.

        Strategy:
        - First half: consecutive recent frames (0, 1, 2, ...)
        - Second half: exponentially spaced distant frames

        Examples:
            samples_length=6,  max_memory=32: [0,1,2] + [8,16,24]
            samples_length=12, max_memory=32: [0,1,2,3,4,5] + [8,12,16,20,24,28]
            samples_length=18, max_memory=64: [0,1,2,3,4,5,6,7,8] + [12,18,24,32,40,48,54,60,63]

        Args:
            samples_length: Total number of samples to return
            max_memory: Maximum available memory size
            device: Target device for the tensor

        Returns:
            Tensor of sampling indices
        """
        if samples_length <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # Split into recent and distant halves
        distant_count = samples_length // 2
        recent_count = samples_length - distant_count  # Ceiling for odd numbers

        # Recent frames: consecutive from 0
        recent_idx = list(range(recent_count))

        # Distant frames: exponentially or linearly spaced
        if distant_count > 0:
            # Start position for distant frames (after recent frames with a gap)
            start = max(
                recent_count, min(8, max_memory // 4)
            )  # At least start from 8 or max_memory//4
            end = max_memory - 1

            if distant_count == 1:
                # Single distant frame: pick middle of available range
                distant_idx = [(start + end) // 2]
            else:
                # Multiple distant frames: use exponential spacing for better temporal coverage
                # Generate exponentially spaced values and scale to [start, end]
                exp_values = [1.5**i for i in range(distant_count)]
                max_exp = exp_values[-1]

                # Scale to [start, end] range
                distant_idx = []
                for exp_val in exp_values:
                    idx = int(start + (exp_val / max_exp) * (end - start))
                    idx = min(idx, end)  # Ensure we don't exceed end
                    distant_idx.append(idx)

                # Ensure indices are monotonically increasing and unique
                distant_idx = sorted(set(distant_idx))

                # If we lost some indices due to deduplication, fill with linear interpolation
                while len(distant_idx) < distant_count:
                    # Find the largest gap and insert a point
                    gaps = [
                        (distant_idx[i + 1] - distant_idx[i], i)
                        for i in range(len(distant_idx) - 1)
                    ]
                    if not gaps:
                        break
                    largest_gap, gap_idx = max(gaps)
                    if largest_gap <= 1:
                        # No more room to insert
                        # Append from end
                        last = distant_idx[-1]
                        if last < end:
                            distant_idx.append(min(last + 1, end))
                        else:
                            break
                    else:
                        new_idx = (distant_idx[gap_idx] + distant_idx[gap_idx + 1]) // 2
                        distant_idx.insert(gap_idx + 1, new_idx)

                # Trim to exact count
                distant_idx = distant_idx[:distant_count]
        else:
            distant_idx = []

        # Combine recent and distant indices
        all_idx = recent_idx + distant_idx

        return torch.tensor(all_idx, dtype=torch.long, device=device)


class DictWithSetter:
    def __init__(self, getter_callback_dict=None):
        self._data = {}
        self.getter_callback_dict = getter_callback_dict

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def keys(self):
        return self._data.keys()

    def get(self, key, default):
        if key not in self._data.keys():
            ret = default
        else:
            ret = self._data[key]

        if self.getter_callback_dict.get(key) is not None:
            ret = self.getter_callback_dict[key](ret)

        return ret

    def clear(self):
        self._data.clear()

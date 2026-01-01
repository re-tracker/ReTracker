"""Torch device helpers.

These helpers are deliberately small and dependency-light so they can be used by
both the CLI apps and library entry points.
"""

from __future__ import annotations

import warnings

import torch


def resolve_device(requested: str | None) -> str:
    """Resolve a requested device string to something usable.

    Supported inputs:
    - "auto": chooses "cuda" if available, otherwise "cpu"
    - "cuda", "cuda:0", ...: returns the same if CUDA is available, otherwise falls back to "cpu"
    - "cpu": always returns "cpu"

    Notes:
    - We load checkpoints on CPU first, then move models to the resolved device.
    - Falling back from CUDA to CPU emits a warning so users notice performance changes.
    """
    if requested is None or requested == "" or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        warnings.warn(
            f"CUDA requested ({requested!r}) but torch.cuda.is_available() is False; "
            "falling back to CPU. Pass --device cpu to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    return requested


def cuda_mem_info() -> list[dict[str, int]]:
    """Return CUDA free/total memory for each visible device.

    Uses `torch.cuda.mem_get_info`, which works even when NVML is unavailable on
    some clusters (where `nvidia-smi` may fail).
    """
    if not torch.cuda.is_available():
        return []

    infos: list[dict[str, int]] = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        infos.append({"index": int(i), "free_bytes": int(free_b), "total_bytes": int(total_b)})
    return infos


def format_cuda_mem_info(infos: list[dict[str, int]]) -> str:
    if not infos:
        return "<no cuda devices>"
    lines = []
    for info in infos:
        free_gb = info["free_bytes"] / (1024**3)
        total_gb = info["total_bytes"] / (1024**3)
        lines.append(f"cuda:{info['index']} free={free_gb:.2f}GB total={total_gb:.2f}GB")
    return "\n".join(lines)

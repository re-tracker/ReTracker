"""Optimizer and LR scheduler builders used by the training stack."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR


def build_optimizer(model: torch.nn.Module, config):
    """Build an optimizer from a config object.

    The config is expected to have fields:
      - name: "adam" | "adamw"
      - lr, eps, betas, fused
    """
    name = config.name
    lr = config.lr
    eps = config.eps
    betas = config.betas
    fused = config.fused

    params = (p for p in model.parameters() if p.requires_grad)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, fused=fused)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, fused=fused)

    raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """Build a Lightning-compatible LR scheduler dict from config.

    Returns:
        dict with keys:
          - scheduler: torch lr_scheduler
          - interval: 'step' or 'epoch'
          - monitor: (optional)
          - frequency: (optional)
    """
    scheduler = {"interval": config.scheduler_interval}
    name = config.scheduler

    if name == "MultiStepLR":
        scheduler["scheduler"] = MultiStepLR(optimizer, config.mslr_milestones, gamma=config.mslr_gamma)
        return scheduler
    if name == "CosineAnnealing":
        scheduler["scheduler"] = CosineAnnealingLR(optimizer, config.cosa_tmax)
        return scheduler
    if name == "ExponentialLR":
        scheduler["scheduler"] = ExponentialLR(optimizer, config.elr_gamma)
        return scheduler

    raise NotImplementedError(f"Unknown scheduler: {name}")


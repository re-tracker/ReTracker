import os
from typing import Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from retracker.utils.rich_utils import CONSOLE


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []
    
    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']
    
    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        CONSOLE.print(
            f"[yellow][Temporary Fix][/yellow] manually set CUDA_VISIBLE_DEVICES to: {visible_devices}"
        )
    else:
        CONSOLE.print(
            "[yellow][Temporary Fix][/yellow] CUDA_VISIBLE_DEVICES already set by user or the main process."
        )
    return len(gpu_ids)


def apply_activation_checkpointing(model: nn.Module, module_classes: list = None):
    if module_classes is None:
        return
    
    target_classes = tuple(module_classes)

    for child_module in model.children():
        if isinstance(child_module, target_classes):
            original_forward = child_module.forward
            
            def new_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, **kwargs, use_reentrant=True)
            
            child_module.forward = new_forward
        else:
            apply_activation_checkpointing(child_module, module_classes)

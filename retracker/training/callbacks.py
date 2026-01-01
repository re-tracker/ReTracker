'''callbacks for pl trainer'''
import os
from pathlib import Path

from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import lightning.pytorch as pl
import torch
import copy
import gc
from collections import defaultdict

from retracker.utils.rich_utils import CONSOLE

class FreezeBackboneCallback(pl.Callback):
    def __init__(self, freeze_epoch):
        super().__init__()
        self.freeze_epoch = freeze_epoch

    def on_epoch_start(self, trainer, pl_module):
        # Freezing feature backbone and coarse matching bkb
        if trainer.current_epoch == self.freeze_epoch:
            for name, param in pl_module.matcher.named_parameters():
                if 'backbone' in name or 'loftr_coarse' in name:
                    param.requires_grad = False
                else: 
                    param.requires_grad = True
            CONSOLE.print(f"[yellow]Backbone frozen from epoch {self.freeze_epoch} on.[/yellow]")

class FreezeAllButConvPredCallback(pl.Callback):
    def __init__(self, freeze_epoch):
        super().__init__()
        self.freeze_epoch = freeze_epoch

    def on_epoch_start(self, trainer, pl_module):
        # Freezing feature backbone and coarse matching bkb
        if trainer.current_epoch == self.freeze_epoch:
            for name, param in pl_module.matcher.named_parameters():
                if 'unc_refinement' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            CONSOLE.print(
                f"[yellow]All freeze but unc_predictor from epoch {self.freeze_epoch} on.[/yellow]"
            )


class NaNLossCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        loss = pl_module.batch_loss
        # todo check update for each inner branch
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            CONSOLE.print("[yellow]Loss is NaN/Inf, callback[/yellow]")
            optimizer = getattr(pl_module, "optimizer")
            optimizer.zero_grad()
            torch.cuda.empty_cache()


class ReduceBackboneLearningRateCallback(pl.Callback):
    def __init__(self, backbone_lr_multiplier=0.1):
        super().__init__()
        self.backbone_lr_multiplier = backbone_lr_multiplier

    def on_train_start(self, trainer, pl_module):
        model = pl_module
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        all_params = list(model.parameters())

        def is_backbone_param(name):
            return 'backbone' in name or 'loftr_coarse' in name

        # update their lr
        selected_param = filter(lambda p: is_backbone_param(p[0]), model.named_parameters())
        for param in selected_param:
            param_group = next((group for group in trainer.optimizers[0].param_groups if param[1] is group['params'][0]), None)
            if param_group is not None:
                param_group['lr'] = current_lr * self.backbone_lr_multiplier
        CONSOLE.print(
            f"[yellow]set backbone & loftr_coarse lr to {current_lr} * {self.backbone_lr_multiplier}[/yellow]"
        )

class AllowedUncertaintyLossCallback(pl.Callback):
    def __init__(self, from_epoch_n_on, per_n_epoch):
        super().__init__()
        self.from_epoch_n_on = from_epoch_n_on
        self.per_n_epoch = per_n_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        # get loss
        # allowed uncertainty loss every N epoch
        if trainer.current_epoch >= self.from_epoch_n_on and \
        trainer.current_epoch % self.per_n_epoch == 0:
            pl_module.loss.train_uncertainty = True
            CONSOLE.print(f"[yellow]Allowed Unc loss in epoch {trainer.current_epoch}.[/yellow]")
        else:
            pl_module.loss.train_uncertainty = False
            
class LongSequenceModeCallback(pl.Callback):
    '''Open Long Sequence Mode for training
    applied after a model can recognize which memory is more important.
    '''
    def __init__(self, from_epoch_n_on):
        super().__init__()
        self.from_epoch_n_on = from_epoch_n_on

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.from_epoch_n_on:
            pl_module.matcher.LONG_SEQUENCE_MODE = True
            CONSOLE.print(f"[yellow]set LONG_SEQUENCE_MODE from epoch {trainer.current_epoch} on.[/yellow]")
        else:
            pl_module.matcher.LONG_SEQUENCE_MODE = False
 
class DumpModelParametersCallback(pl.Callback):
    def __init__(self, per_n_step):
        super().__init__()
        self.per_n_step = per_n_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step % self.per_n_step == 0:
            # pl_module.last_normal_parameters['model'] = [p.clone() for p in pl_module.matcher.parameters()]
            # NOTE: avoid nan exp_avg in optimizer! 
            # save optimizer state_dict
            # pl_module.last_normal_parameters['optimizer'] = copy.deepcopy(pl_module.optimizers().state_dict())
            
            CONSOLE.print("[yellow][CallBack][/yellow]: backup matcher parameters")


class SaveSharedCheckpointCallback(pl.Callback):
    def __init__(self):
        """Save version independent last checkpoints for elastic training"""
        super().__init__()

    # def on_train_epoch_end(self, trainer, pl_module):
    def on_validation_epoch_end(self, trainer, pl_module):
        # save version independent checkpoints
        # DDP save, rank=0
        # skip save during sanity check
        save_path = os.environ.get('ELASTIC_RESUME_CKPT_DIR')
        if trainer.global_rank == 0 and save_path is not None:
            os.makedirs(Path(save_path), exist_ok=True)
            # copy checkpoint to save_path by os asynchronusly
            last_ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, "last.ckpt")
            if os.path.exists(last_ckpt_path):
                target = os.path.join("..", *last_ckpt_path.split("/")[-3:])
                link_path = os.path.join(save_path, "last.ckpt")
                if os.path.islink(link_path):
                    os.unlink(link_path)
                os.symlink(target, link_path)
                CONSOLE.print(f"[yellow][CallBack][/yellow]: Save checkpoint at epoch {trainer.current_epoch}")

class PredOcclusionCallback(pl.Callback):
    def __init__(self, from_epoch_n_on):
        super().__init__()
        self.from_epoch_n_on = from_epoch_n_on

    def on_epoch_start(self, trainer, pl_module):
        # Freezing feature backbone and coarse matching bkb
        if trainer.current_epoch == self.from_epoch_n_on:
            trainer.loss.pred_occlusion = True
            CONSOLE.print(f"[yellow]Allow pred occlusion from epoch {self.from_epoch_n_on} on.[/yellow]")



class MemoryUsageCallback(pl.Callback):
    def __init__(self, print_every_n_steps: int = 50):
        super().__init__()
        self.print_every_n_steps = print_every_n_steps
        self.memory_usage = defaultdict(float)
        self.hooks = []

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not torch.cuda.is_available():
            CONSOLE.print("[yellow]CUDA not available. MemoryUsageCallback will not run.[/yellow]")
            return

        for name, module in pl_module.named_modules():
            pre_hook = self._create_pre_hook()
            post_hook = self._create_post_hook(name)
            self.hooks.append(module.register_forward_pre_hook(pre_hook))
            self.hooks.append(module.register_forward_hook(post_hook))

    def _create_pre_hook(self):
        def pre_hook(module, inputs):
            torch.cuda.synchronize()
            module._memory_before_forward = torch.cuda.memory_allocated()
        return pre_hook

    def _create_post_hook(self, name):
        def post_hook(module, inputs, outputs):
            torch.cuda.synchronize()
            mem_before = module._memory_before_forward
            mem_after = torch.cuda.memory_allocated()
            self.memory_usage[name] += (mem_after - mem_before) / (1024 ** 2)  # Convert to MB
            del module._memory_before_forward
        return post_hook

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        if not self.hooks or (trainer.global_step + 1) % self.print_every_n_steps != 0:
            return

        CONSOLE.print(f"\n--- [Memory Usage Report at Step {trainer.global_step + 1}] ---")
        sorted_usage = sorted(self.memory_usage.items(), key=lambda item: item[1], reverse=True)
        
        total_activation_memory = sum(self.memory_usage.values())

        CONSOLE.print(f"{'Module Name':<70} | {'Activation Memory (MB)'}")
        CONSOLE.print("-" * 90)

        for name, usage_mb in sorted_usage:
            if usage_mb > 0.01:
                CONSOLE.print(f"{name:<70} | {usage_mb:.4f}")

        CONSOLE.print("-" * 90)
        CONSOLE.print(f"{'Total Activation Memory in this Interval':<70} | {total_activation_memory:.4f}")
        CONSOLE.print(
            f"{'Peak Memory Allocated (Total)':<70} | {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f}"
        )
        CONSOLE.print("--- [End of Report] ---\n")

        self.memory_usage.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class EMACallback(pl.Callback):
    """
    Implements Exponential Moving Average (EMA) for model parameters.
    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    
    Why this version?
    - It does NOT wrap the optimizer, ensuring compatibility with other callbacks 
      that modify LR or param_groups (like ReduceBackboneLearningRateCallback).
    - It maintains a separate copy of weights and swaps them in during validation/testing.
    """
    def __init__(self, decay: float = 0.999, use_ema_for_validation: bool = True):
        super().__init__()
        self.decay = decay
        self.use_ema_for_validation = use_ema_for_validation
        
        # Storage for EMA weights and temporary backup of original weights
        self.ema_state_dict = {}
        self.original_state_dict = {}
        self._ema_initialized = False


    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA weights when training starts."""
        # Only initialize if not already loaded from checkpoint
        if not self._ema_initialized:
            self.ema_state_dict = {
                name: param.clone().detach().to(pl_module.device)
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }
            self._ema_initialized = True
            CONSOLE.print(f"[dim]EMA initialized with decay {self.decay}[/dim]")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA weights after each training step."""
        # Update EMA weights
        CONSOLE.print("[dim]EMA averaging[/dim]")
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.ema_state_dict:
                    # formula: shadow = decay * shadow + (1 - decay) * variable
                    # equivalent to: shadow -= (1 - decay) * (shadow - variable)
                    # using in-place operations to save memory
                    ema_param = self.ema_state_dict[name].to(param.device)
                    ema_param.sub_((1.0 - self.decay) * (ema_param - param))
                    self.ema_state_dict[name] = ema_param

    def on_validation_start(self, trainer, pl_module):
        """Swap original weights with EMA weights for validation."""
        if not self._ema_initialized or not self.use_ema_for_validation:
            return

        CONSOLE.print(f"[dim]Swapping to EMA weights for validation (Step {trainer.global_step})[/dim]")
        self._swap_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        """Restore original weights after validation."""
        if not self._ema_initialized or not self.use_ema_for_validation:
            return

        CONSOLE.print("[dim]Restoring training weights after validation[/dim]")
        self._restore_weights(pl_module)

    def on_test_start(self, trainer, pl_module):
        if not self._ema_initialized:
            return
        CONSOLE.print("[dim]Swapping to EMA weights for testing[/dim]")
        self._swap_weights(pl_module)

    def on_test_end(self, trainer, pl_module):
        if not self._ema_initialized:
            return
        CONSOLE.print("[dim]Restoring training weights after testing[/dim]")
        self._restore_weights(pl_module)

    def _swap_weights(self, pl_module):
        """Helper to save current weights and load EMA weights."""
        self.original_state_dict = {
            name: param.clone().detach()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        
        # Load EMA weights into the model
        # We manually copy to ensure gradient chains are not messed up, 
        # though we are in validation so no grads anyway.
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.ema_state_dict:
                    param.data.copy_(self.ema_state_dict[name])

    def _restore_weights(self, pl_module):
        """Helper to restore original weights."""
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.original_state_dict:
                    param.data.copy_(self.original_state_dict[name])
        
        # Clear backup to save memory
        self.original_state_dict = {}

    def state_dict(self):
        """Save EMA state to checkpoint."""
        return {
            'ema_state_dict': self.ema_state_dict,
            'decay': self.decay,
            '_ema_initialized': self._ema_initialized
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.ema_state_dict = state_dict.get('ema_state_dict', {})
        self.decay = state_dict.get('decay', self.decay)
        self._ema_initialized = state_dict.get('_ema_initialized', False)
        CONSOLE.print("[dim]EMA state loaded from checkpoint.[/dim]")

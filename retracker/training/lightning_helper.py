import torch

from retracker.utils.rich_utils import CONSOLE

def remove_nan_from_optimizer_state(optimizer):
    for group in optimizer.state.values():
        for k, v in group.items():
            if isinstance(v, float) and torch.isnan(v):
                group[k] = 0.0
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, float) and torch.isnan(vv):
                        group[k][kk] = 0.0
            elif isinstance(v, torch.Tensor) and torch.isnan(v).any():
                group[k] = torch.where(torch.isnan(v), torch.zeros_like(v), v)


def check_nan_grad_during_training(pl_module):
    nan_num = sum([torch.isnan(p).sum() for p in pl_module.matcher.parameters() if torch.isnan(p).any()])
    if nan_num > 0:
        # recover parameters from last normal parameter state
        for p, last_p in zip(pl_module.matcher.parameters(), pl_module.last_normal_parameters['model']):
            p.data = last_p.data.clone()
        # remove all Nan in state:
        # remove_nan_from_optimizer_state(pl_module.optimizers())
        CONSOLE.print(f"[yellow]recover from {nan_num} nans in model&optimizer parameters[/yellow]")
        pl_module.abort_current_branch_flag = True

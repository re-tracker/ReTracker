import torch
import torch.nn as nn


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T):
        super().__init__()
        self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

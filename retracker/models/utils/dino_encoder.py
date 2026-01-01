import copy

import torch.nn as nn
from torch import Tensor


class DinoEncoder(nn.Module):
    """encode dino features for memory bank and queries"""

    def __init__(self, config):
        super().__init__()
        dim = config.dim
        blocks_num = config.blocks_num
        block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(blocks_num)])

    def forward(self, feats: Tensor):
        """encode dino features in MemoryBank and queries by tiny encoder
        Args:
            feats (Tensor) : [..., C]

        Returns:
            feats (Tensor): [..., C]
        """
        feat_shape = feats.shape

        feats = feats.reshape(-1, feat_shape[-1])
        for block in self.blocks:
            feats = feats + block(feats)

        feats = feats.view(*feat_shape)
        return feats

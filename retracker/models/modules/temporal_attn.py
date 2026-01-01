import torch.nn as nn
from torch import Tensor

from retracker.models.modules import LocalFeatureTransformer
from retracker.models.utils.mem_manager.memorymanager import MemoryManager


class TemporalAttn(nn.Module):
    def __init__(self, config, module_id):
        super().__init__()
        self.config = config
        self.module_id = module_id
        self.use_detached_memory = config["use_detached_memory"]
        self.bank_attn = LocalFeatureTransformer(config["bank_attn"])
        self.new_coarse_matching = LocalFeatureTransformer(config["new_coarse"])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        mask_c0: Tensor = None,
        mask_c1: Tensor = None,
        mem_manager: MemoryManager = None,
    ):
        memory_c0 = mem_manager.memory[f"feat_0_{self.module_id}"]
        memory_last = mem_manager.memory[f"feat_last_{self.module_id}"]

        feat_c0, _c0 = self.bank_attn(feat_c0, memory_c0)
        feat_c1, _c1 = self.bank_attn(feat_c1, memory_last)
        feat_c0, feat_c1 = self.new_coarse_matching(feat_c0, feat_c1)

        mem_manager.set_memory(f"feat_0_{self.module_id}", _c0, detach=self.use_detached_memory)
        mem_manager.set_memory(f"feat_last_{self.module_id}", _c1, detach=self.use_detached_memory)
        return feat_c0, feat_c1

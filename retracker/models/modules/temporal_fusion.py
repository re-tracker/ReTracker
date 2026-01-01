import copy

import torch.nn as nn
from torch import Tensor

from ..utils.mem_manager.memorymanager import MemoryManager
from .queries_attn import QueriesAttn
from .temporal_attn import TemporalAttn


class TemporalFusion(nn.Module):
    def __init__(self, config: dict, d_model: int, module_id: str):
        super().__init__()
        self.config = config
        self.module_id = module_id
        self.layers_name = config["layers_name"]
        layers_dict = {
            "temporal_attn": copy.deepcopy(TemporalAttn(config["temporal_attn"], self.module_id)),
            "queries_attn": copy.deepcopy(
                QueriesAttn(config["queries_attn"], d_model, self.module_id)
            ),
        }
        self.temporal_layer = nn.ModuleList(
            [copy.deepcopy(layers_dict[name]) for name in self.layers_name]
        )

    def forward(
        self,
        b_ids: Tensor,
        i_ids: Tensor,
        feat_c0: Tensor,
        feat_c1: Tensor,
        mask_c0: Tensor = None,
        mask_c1: Tensor = None,
        mem_manager: MemoryManager = None,
    ):
        """"""

        is_first_frame = False
        if not mem_manager.exists(f"feat_0_{self.module_id}"):
            is_first_frame = True
            mem_manager.set_memory(f"feat_0_{self.module_id}", feat_c0)
            mem_manager.set_memory(f"feat_last_{self.module_id}", feat_c1)

        for layer, layer_name in zip(self.temporal_layer, self.layers_name):
            if layer_name == "temporal_attn":
                feat_c0, feat_c1 = layer(feat_c0, feat_c1, mask_c0, mask_c1, mem_manager)  # sample
            elif layer_name == "queries_attn":
                if is_first_frame:  # first frame has no memory for queries_attn
                    continue
                feat_c0, feat_c1 = layer(b_ids, i_ids, feat_c0, feat_c1, mem_manager)
            else:
                raise NotImplementedError

        return feat_c0, feat_c1

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

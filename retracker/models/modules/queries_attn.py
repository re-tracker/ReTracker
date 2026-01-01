import copy

import torch
import torch.nn as nn
from torch import Tensor

from retracker.models.modules import LocalFeatureTransformer, Mem_Queries_FeatureTransformer

from retracker.models.utils.mem_manager.memorymanager import MemoryManager

from ..utils.position_encoding import TemporalEncodingSine


class QueriesAttn(nn.Module):
    def __init__(self, config, d_model, module_id):
        super().__init__()
        self.config = config
        self.module_id = module_id
        self.replace_rate: float = config["replace_rate"]
        self.sampled_num: int = config["sampled_num"]
        self.QK_pe = TemporalEncodingSine(d_model, config["max_pe_length"])
        self.queries_attn_layers = nn.ModuleList(
            [
                copy.deepcopy(Mem_Queries_FeatureTransformer(config["queries_attn"]))
                for _ in range(config["layer_num"])
            ]
        )
        self.final_attn_layers = nn.ModuleList(
            [copy.deepcopy(LocalFeatureTransformer(config["final_attn"])) for _ in range(1)]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        b_ids: Tensor,
        i_ids: Tensor,
        feat_c0: Tensor,
        feat_c1: Tensor,
        mem_manager: MemoryManager = None,
    ):
        """Temporal matching pipeline, the operations are followed:
        Args:
          feat_c0(torch.Tensor): [B, N, C]
          feat_c1(torch.Tensor): [B, M, C]
        Returns:
        """

        # crop features at assigned positions
        # queries_feat: B, n(n=768 of N=4096), C(256)
        B, N, C = feat_c0.shape
        n = i_ids.shape[0] // B

        # preds_feat_in_memory: B, n, ?(frames in memory), C | max ?=8 during Training
        preds_feat_in_memory = mem_manager.sample_memory(
            f"preds_feat_{self.module_id}", stack_dim=-2, samples_length=self.sampled_num
        )  # Bn ? C
        queries_feat = feat_c0[b_ids, i_ids].reshape(B * n, 1, C)  # Bn 1 C  n out of N feat
        preds_feat_in_memory = torch.cat([queries_feat, preds_feat_in_memory], dim=-2)
        K = preds_feat_in_memory.shape[-2]

        # replace some places in pred_attn_matrix
        preds_feat_in_memory = self._random_replace(
            preds_feat_in_memory, queries_feat, self.replace_rate
        )

        # build attn matrix by preds_feat_in_memory:
        pred_attn_matrix = torch.zeros((B, N, K, C), device=feat_c0.device)  # B N ? C
        pred_attn_matrix[b_ids, i_ids] = preds_feat_in_memory.view(B * n, K, C)

        # queries_feat: [B, n, C]; pred_feat_in_memory: [B, n, K, C]
        # feat_c0: [B, N, C]; pred_attn_matrix: [B, N, K, C]

        # build mask: only i_ids feat are involved
        mask0 = torch.zeros((B, N, 1), device=feat_c0.device).bool()  # B N
        mask1 = torch.zeros((B, N, K), device=feat_c0.device).bool()  # B N
        mask0[b_ids, i_ids] = True
        mask1[b_ids, i_ids] = True

        for idx in range(self.config["layer_num"]):
            # 1. queries_attn layer, update queries feat
            feat_c0 = self.queries_attn_layers[idx](
                feat_c0, pred_attn_matrix, mask0, mask1, self.QK_pe
            )  # [B H'W' C], [B N K C]

        # 3. update feat_c0, feat_c1
        # Autocast is only beneficial/available on CUDA. Keep CPU paths simple.
        with torch.amp.autocast("cuda", enabled=feat_c0.is_cuda):
            feat_c0, feat_c1 = self.final_attn_layers[0](feat_c0, feat_c1)
        return feat_c0, feat_c1

    def _random_replace(
        self, feats_in_memory: Tensor, feat: Tensor, replace_ratio: float
    ) -> Tensor:
        """random replace some `feats_in_memory` in `dim` channel with `feat` during training\n
        Args:\n
          feats_in_memory(torch.Tensor): [Bn, K, C]\n
          feat(torch.Tensor): [Bn, 1, C]\n
          replace_ratio(float): ratio of replacing\n
        Returns:\n
          feats_in_memory(torch.Tensor): [Bn, K, C]\n
        """
        # create a random mask in `dim` channel
        replace_ratio = replace_ratio if self.training else 0
        replace_mask = (
            torch.rand(feats_in_memory.shape[-2], device=feats_in_memory.device) < replace_ratio
        )
        # replace_mask K, replace some places in K channel by replace_mask
        feats_in_memory[:, replace_mask, :] = feat

        return feats_in_memory

import torch
import torch.nn as nn


class CoarseMatching_CLS_selected(nn.Module):
    """Coarse matching module by classification token"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, data):
        """provide coarse matches: mkpts1_c
            1. set necessary dict for coarse level dense supervision
            2. get fine level matches
        Input:
            cls_onehot (torch.Tensor): [B, M, num_classes+1]
        Output:
            provide coarse matches: mkpts1_c
        """
        cls_onehot = data["pred_cls_queries"]
        queries = data["queries"]  # [B, N, 2]
        _device = queries.device
        B, N = queries.shape[0], queries.shape[1]
        h0i, w0i = data["hw_i"]
        h0_8x, w0_8x = h0i // 8, w0i // 8
        scale_8x = h0i // h0_8x

        def clip_kpts(kpts, hw):
            torch.clamp_(kpts[..., 0], min=0, max=hw[1] - 1)
            torch.clamp_(kpts[..., 1], min=0, max=hw[0] - 1)
            return kpts

        b_ids = torch.arange(B, device=_device)[..., None].expand(B, N).reshape(-1)

        assert cls_onehot.shape[1] == data["queries"].shape[1]
        # with explicit queries appended, use it directly;
        i_ids = torch.arange(N, device=_device)[None].expand(B, N).reshape(-1)  # [B*N]

        # get the most possible class from cls_onehot:
        mconf_logits_map, cls_pred_map = cls_onehot[..., :-1].max(-1)  # [B, M]
        all_j_ids = cls_pred_map  # [B,M]
        j_ids = all_j_ids[b_ids, i_ids]
        mconf_logits = mconf_logits_map[b_ids, i_ids]

        mkpts1_c = (
            torch.stack([j_ids % w0_8x, torch.div(j_ids, w0_8x, rounding_mode="trunc")], dim=1)
            * scale_8x
        ) * 1.0
        coarse_matches = {
            "b_ids": b_ids,
            "j_ids": j_ids,
            "m_bids": b_ids,
            "queries": data["queries"],
            "mkpts1_c": mkpts1_c,
            "mconf_logits_coarse": mconf_logits.reshape(B, N, 1),
            "updated_pos": mkpts1_c.reshape(B, N, 2),  # to B N 2
            "updated_occlusion": data[
                "pred_occlusion_queries"
            ],  # .reshape(-1,1,1), # B N 1 to BN 1 1
            "updated_certainty": data[
                "pred_certainty_queries"
            ],  # .reshape(-1,1,1), # B N 1 to BN 1 1
            "pred_cls_queries": data["pred_cls_queries"],
        }
        return coarse_matches

"""given queries feat, coarse matches, refine matches by PIPs;

v1.0 change forward api
"""

import copy
import logging
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from retracker.models.backbone.layers import Mlp
from retracker.models.modules.roma_decoder import (
    F0RefinementBlock,
    ROMATransformerDecoder3,
)
from retracker.models.utils.kernel_functions import CosKernel
from retracker.models.utils.local_crop import LocalCrop
from retracker.models.utils.mem_manager.memorymanager import MemoryManager
from retracker.models.utils.misc import (
    get_pos_enc,
    random_erase_patch,
)
from retracker.utils.profiler import global_profiler


logger = logging.getLogger(__name__)


class MatchesRefinement(nn.Module):
    def __init__(
        self,
        mem_manager: MemoryManager,
        mem_dict_name: str,
        use_last_mem: bool,
        config: dict,
    ):
        super().__init__()
        # cfg
        self.config = config
        self.use_randomwalk = config["use_randomwalk"]
        self.pass_info_between_lvls = config["pass_info_between_lvls"]
        self.pips_iter_num = config["pips_iter_num"]
        self.causal_context_num = config["causal_context_num"]
        self.causal_memory_size = config.get("causal_memory_size", 32)  # Memory size for inference
        self.corr_radius = config["corr_radius"]  # 3
        self.corr_num_levels = config["corr_num_levels"]
        self.fine_dims = config["fine_dims"]
        self.mem_manager = mem_manager
        self.mem_dict_name = mem_dict_name
        self.use_last_mem = use_last_mem

        self.hidden_dim = config["hidden_dims"]
        self.window_size = config["window_size"]
        self.jitter_aug = config["jitter_aug"]
        self.jitter_list = config["jitter_list"]
        self.chn_c = 256
        self.chn_f = 128

        self.K = CosKernel(T=0.2)
        self.W = self.corr_radius * 2 + 1
        self.use_abs_pos_pe = config["use_abs_pos_pe"]
        self.pe_dim = (
            config["pe_dim"] * 1 if not self.use_abs_pos_pe else config["pe_dim"] * 2
        )  # 64
        self.flow_dim = config["flow_dim"]  # 64

        # Project features to lower dimension for efficiency
        self.input_feature_dims = [384, 256, 128]
        self.target_feature_dims = self.fine_dims  # [128, 128, 128]
        if self.fine_dims is not None:
            self.feature_projectors = nn.ModuleList(
                [
                    nn.Conv2d(in_dim, out_dim, kernel_size=1)
                    if in_dim != out_dim
                    else nn.Identity()
                    for in_dim, out_dim in zip(self.input_feature_dims, self.target_feature_dims)
                ]
            )

            # Update level dimensions to reflect projected features
            # All levels now use the target dimension (e.g., 128) + PE + Flow
            lvls_dim = [dim + self.pe_dim + self.flow_dim for dim in self.target_feature_dims]
        else:
            lvls_dim = [384 + self.pe_dim, 256 + self.pe_dim, 128 + self.pe_dim]

        self.pos_conv_local = torch.nn.Conv2d(2, config["pe_dim"], 1, 1)
        self.pos_conv_global = torch.nn.Conv2d(2, config["pe_dim"], 1, 1)
        self.local_crop = LocalCrop(
            corr_num_levels=self.corr_num_levels, corr_radius=self.corr_radius
        )

        self.augment_config = config.get("augment_config", {})
        # patch augmentation
        self.p_augment = self.augment_config.get("p_augment", 0)
        self.p_zero_mask = self.augment_config.get("p_zero_mask", 0)
        self.p_replace_mask = self.augment_config.get("p_replace_mask", 0)
        # patch erase augmentation
        self.use_patch_erase_aug = self.augment_config.get("use_patch_erase_aug", False)
        self.patch_erase_p = self.augment_config.get("patch_erase_p", 0)
        self.use_prior_jitter_aug = self.augment_config.get("use_prior_jitter_aug", False)
        self._build_troma_blocks(config, lvls_dim)
        if config["use_patch_cross_attn"]:
            self.cross_attn = nn.ModuleList(
                [
                    copy.deepcopy(F0RefinementBlock(dim=lvls_dim[i] - self.pe_dim))
                    for i in range(self.corr_num_levels)
                ]
            )

    def forward(
        self,
        data,
        feat0_pym_list,
        feat1_pym_list,
        warped_feat1_pym_list,
        is_get_causal_context=True,
        always_use_warped=False,
    ):
        ## 0. get variables
        bz = data["queries"].shape[0]
        hw_i, hw_d, hw_f = data["hw_i"], data["hw0_d"], data["hw0_f"]
        hw_c = data["hw0_c"] if "hw0_c" in data else None
        B = bz
        WW = self.W * self.W
        N = data["updated_pos"].shape[0] // B  # BxN, F=1, C=2

        # crop Patches;
        with global_profiler.record("feat_prepare"):
            if len(feat0_pym_list) == 3:
                feat0_pym_dino, feat0_pym_coarse, feat0_pym_fine = feat0_pym_list
                feat1_pym_dino, feat1_pym_coarse, feat1_pym_fine = feat1_pym_list

                feat0_pym_dino = rearrange(feat0_pym_dino, "B F (H W) C-> B F C H W", H=hw_d[0])
                feat1_pym_dino = rearrange(feat1_pym_dino, "B F (H W) C-> B F C H W", H=hw_d[0])
                feat0_pym_coarse = rearrange(feat0_pym_coarse, "B F (H W) C-> B F C H W", H=hw_c[0])
                feat1_pym_coarse = rearrange(feat1_pym_coarse, "B F (H W) C-> B F C H W", H=hw_c[0])
                feat0_pym_fine = rearrange(feat0_pym_fine, "B F (H W) C-> B F C H W", H=hw_f[0])
                feat1_pym_fine = rearrange(feat1_pym_fine, "B F (H W) C-> B F C H W", H=hw_f[0])

                source_fmaps_pyramid = [feat0_pym_dino, feat0_pym_coarse, feat0_pym_fine]
                target_fmaps_pyramid = [feat1_pym_dino, feat1_pym_coarse, feat1_pym_fine]

                # Project features to reduce dimensionality
                if self.fine_dims is not None:
                    with global_profiler.record("feat_project"):
                        source_fmaps_pyramid = self._project_features(source_fmaps_pyramid)
                        target_fmaps_pyramid = self._project_features(target_fmaps_pyramid)
                # target_fmaps_pyramid = self._warp_strategy(warped_feat1_pym_list, hw_d, hw_c, hw_f)

        queries_pos_feat = data["queries"]  # B N 2
        _data = {
            "occlusion_logits": data["updated_occlusion"],  # [BxN, F, 1]
            "certainty_logits": data["updated_certainty"],  # [BxN, F, 1]
        }
        _data["updated_pos"] = data["updated_pos"]  # BxN, F=1, C=2

        if not self.mem_manager.exists("updated_pos"):
            first_pos = data["queries"].reshape(-1, 1, 2)
            # Memory size: training uses smaller size for efficiency, inference uses configured size
            train_mem_size = 12
            infer_mem_size = self.causal_memory_size
            self.mem_manager.push_memory(
                "updated_pos",
                value=first_pos,
                detach=True,
                stack_dim=1,
                auto_fill=True,
                custom_size=(train_mem_size if self.training else infer_mem_size),
            )

        with global_profiler.record("source_local_crop"):
            track_feat_patch_nlvl_list = self.local_crop.get_track_feat_3lvl(
                source_fmaps_pyramid, queries_pos_feat, scales=[16, 8, 2]
            )
            track_feat_patch_nlvl_list = [
                rearrange((track_feat_patch_ilvl), "BN F C W V -> BN F (W V) C")
                for track_feat_patch_ilvl in track_feat_patch_nlvl_list
            ]

        updated_pos_nlvl, updated_occ_nlvl, updated_exp_nlvl = [], [], []
        updated_pos_nlvl_flow, updated_occ_nlvl_flow, updated_exp_nlvl_flow = [], [], []
        is_masked_list = []
        past_context_tokens = []

        for iter_idx in range(3):
            level_name = ["16x", "8x", "2x"][iter_idx]
            with global_profiler.record(f"iter_{level_name}"):
                _data["updated_pos"] = _data["updated_pos"].detach()  # BxN, F=1, C=2
                # add jitter aug if training
                if self.training and self.jitter_aug:
                    _data["updated_pos"] = self._add_jitter(_data["updated_pos"], iter_idx, hw_i[0])

                _data["occlusion_logits"] = _data["occlusion_logits"].detach()
                _data["certainty_logits"] = _data["certainty_logits"].detach()
                current_pos = _data["updated_pos"].clone()
                current_occ = _data["occlusion_logits"].clone()
                current_exp = _data["certainty_logits"].clone()

                updated_pos_fine_scale = rearrange(
                    _data["updated_pos"].clone(), "(B N) F C -> (B F) N C", B=B
                )

                # Only extract features for the current level (not all 3 levels)
                with global_profiler.record("target_local_crop"):
                    scale = [16, 8, 2][iter_idx]
                    _corr_feat_patch = self.local_crop.get_track_feat_single_level(
                        target_fmaps_pyramid[iter_idx], updated_pos_fine_scale, scale=scale
                    )
                    _corr_feat_patch_nlvl = rearrange(
                        _corr_feat_patch, "BN F C W V -> BN F (W V) C"
                    )

                _track_feat_patch_nlvl = track_feat_patch_nlvl_list[iter_idx]

                if iter_idx > 0:
                    _corr_feat_patch_nlvl, is_masked_this_iter = self.patch_aug(
                        _corr_feat_patch_nlvl, B, N
                    )
                else:
                    is_masked_this_iter = False
                is_masked_list.append(is_masked_this_iter)

                # ===  Prepare context tokens for the current level ===
                projected_context_tokens = []
                if iter_idx > 0:
                    for i, token in enumerate(past_context_tokens):
                        # Step 1: Add PE at the source level, where dimensions match perfectly.
                        token_with_pe = token + self.context_token_pes[i]

                        # Step 2: Now, project the combined token to the target level's dimension.
                        proj_token = token_with_pe
                        for j in range(i, iter_idx):
                            proj_token = self.context_token_projs[j](proj_token)

                        projected_context_tokens.append(proj_token)

                # Pass context tokens and get the new one back ===
                # extract last walking context from memory
                last_walking_context = self.mem_manager.get_memory(
                    key=f"walking_context_{iter_idx}", default=None
                )
                with global_profiler.record("troma_block"):
                    res, new_context_token, next_walking_context = self.random_walk_troma_process(
                        data,
                        current_pos,
                        _track_feat_patch_nlvl,
                        _corr_feat_patch_nlvl,
                        level=iter_idx,
                        B=B,
                        N=N,
                        prior_pos_delta=None,
                        context_tokens=projected_context_tokens,
                        last_walking_context=last_walking_context,
                    )
                past_context_tokens.append(new_context_token)
                # ====================================================================

                last_walking_context = self.mem_manager.set_memory(
                    key=f"walking_context_{iter_idx}", value=next_walking_context, detach=False
                )

                if res.dim() == 4:  # 4d dense head
                    _WW = res.shape[2]
                    pos_delta, occ_delta, exp_delta = (
                        res[..., _WW // 2, :2],
                        res[..., _WW // 2, 2:3],
                        res[..., _WW // 2, 3:4],
                    )
                else:
                    pos_delta, occ_delta, exp_delta = res[..., :2], res[..., 2:3], res[..., 3:4]

                pos_delta = pos_delta * [16.0, 8.0, 2.0][iter_idx]

                # if iter_idx > 0:
                #     pos_delta = occ_delta = exp_delta = 0
                updated_pos = current_pos + pos_delta
                updated_occ = current_occ + occ_delta
                updated_exp = current_exp + exp_delta

                _data["updated_pos"] = updated_pos
                _data["occlusion_logits"] = updated_occ
                _data["certainty_logits"] = updated_exp

                updated_pos_nlvl.append(updated_pos)
                updated_occ_nlvl.append(updated_occ)
                updated_exp_nlvl.append(updated_exp)

                if res.dim() == 4:  # 4d dense
                    # res: BN F WW C
                    _WW = res.shape[2]
                    pos_delta_dense, occ_delta_dense, exp_delta_dense = (
                        res[..., :, :2],
                        res[..., :, 2:3],
                        res[..., :, 3:4],
                    )

                    # Apply the same scale as sparse delta
                    scale = [16.0, 8.0, 2.0][iter_idx]
                    pos_delta_dense_scaled = pos_delta_dense * scale

                    updated_pos_dense = current_pos.detach()[..., None, :] + pos_delta_dense_scaled
                    updated_occ_dense = current_occ.detach()[..., None, :] + occ_delta_dense
                    updated_exp_dense = current_exp.detach()[..., None, :] + exp_delta_dense
                    updated_pos_nlvl_flow.append(updated_pos_dense)
                    updated_occ_nlvl_flow.append(updated_occ_dense)
                    updated_exp_nlvl_flow.append(updated_exp_dense)

        _data.update(
            {
                "updated_pos_nlvl": torch.stack(updated_pos_nlvl, dim=-2),
                "updated_occ_nlvl": torch.stack(updated_occ_nlvl, dim=-2),
                "updated_exp_nlvl": torch.stack(updated_exp_nlvl, dim=-2),
                "is_masked_list": is_masked_list,
            }
        )

        if res.dim() == 4:  # 4d dense head
            _data.update(
                {
                    "updated_pos_nlvl_flow": torch.stack(
                        updated_pos_nlvl_flow, dim=1
                    ),  # BN [F=1, stack here] WW 2
                    "updated_occ_nlvl_flow": torch.stack(updated_occ_nlvl_flow, dim=1),
                    "updated_exp_nlvl_flow": torch.stack(updated_exp_nlvl_flow, dim=1),
                }
            )

        # Compute features at final position for memory (all 3 levels needed)
        with global_profiler.record("save_state"):
            self.save_current_state(
                _data,
                B,
                N,
                WW,
                track_feat_patch_nlvl_list,
                target_fmaps_pyramid,
                corr_feat_patch_nlvl_list=None,
            )  # Will compute fresh
        return _data

    def _project_features(self, fmaps_pyramid):
        """
        Projects feature maps to a lower dimension using 1x1 convolutions.

        Args:
            fmaps_pyramid (list[torch.Tensor]): List of feature maps with shape [B, F, C, H, W].

        Returns:
            list[torch.Tensor]: Projected feature maps with target dimension.
        """
        projected_fmaps = []
        for i, fmap in enumerate(fmaps_pyramid):
            if i >= len(self.feature_projectors):
                projected_fmaps.append(fmap)
                continue

            B, F, C, H, W = fmap.shape
            # Collapse B and F dimensions for Conv2d
            fmap_reshaped = fmap.view(B * F, C, H, W)

            # Apply projection (Linear/Conv1x1)
            fmap_proj = self.feature_projectors[i](fmap_reshaped)

            # Restore dimensions
            fmap_proj = fmap_proj.view(B, F, -1, H, W)
            projected_fmaps.append(fmap_proj)

        return projected_fmaps

    def save_current_state(
        self,
        _data,
        B,
        N,
        WW,
        source_feat_patch_nlvl_list,
        target_fmaps_pyramid,
        corr_feat_patch_nlvl_list=None,
    ):
        """
        dump current frames infos, only confident predictions are saved;

        Args:
            corr_feat_patch_nlvl_list: Pre-computed correlation features from last iteration.
                                       If provided, skip redundant feature extraction.
        """
        updated_mask = torch.sigmoid(_data["certainty_logits"]) > (0 if self.training else 0.2)

        # Memory size: training uses smaller size for efficiency, inference uses configured size
        train_mem_size = 12
        infer_mem_size = self.causal_memory_size
        mem_size = train_mem_size if self.training else infer_mem_size

        # Reuse pre-computed features if available, otherwise compute
        if corr_feat_patch_nlvl_list is None:
            updated_pos_fine_scale = rearrange(
                _data["updated_pos"].clone().detach(), "(B N) F C -> (B F) N C", B=B
            )
            corr_feat_patch_nlvl_list = self.local_crop.get_track_feat_3lvl(
                target_fmaps_pyramid, updated_pos_fine_scale, scales=[16, 8, 2]
            )
            corr_feat_patch_nlvl_list = [
                rearrange((corr_feat_patch_ilvl), "BN F C W V -> BN F (W V) C")
                for corr_feat_patch_ilvl in corr_feat_patch_nlvl_list
            ]

        for idx, (_source_feat_patch_ilvl, corr_feat_patch_ilvl) in enumerate(
            zip(source_feat_patch_nlvl_list, corr_feat_patch_nlvl_list)
        ):
            _C = corr_feat_patch_ilvl.shape[-1]
            _updated_mask = repeat(updated_mask, f"BN F 1 -> BN F {WW} {_C}")
            # corr_feat_patch_ilvl=self.cross_attn[idx](corr_feat_patch_ilvl, source_feat_patch_ilvl)
            self.mem_manager.push_memory(
                f"causal_corr_{idx}",
                value=corr_feat_patch_ilvl,
                detach=False,
                stack_dim=1,
                auto_fill=True,
                custom_size=mem_size,
                updated_mask=_updated_mask,
            )

        _updated_mask = repeat(updated_mask, "BN F 1 -> BN F 2")
        updated_pos = _data["updated_pos"]
        self.mem_manager.push_memory(
            key="updated_pos",
            value=updated_pos,
            stack_dim=1,
            detach=True,
            auto_fill=True,
            custom_size=mem_size,
            updated_mask=_updated_mask,
        )
        return

    def random_walk_troma_process(
        self,
        data,
        pos_curr,
        patch_t0,
        patch_B,
        level,
        B,
        N,
        prior_pos_delta=None,
        context_tokens=None,
        last_walking_context=None,
    ):
        """ """
        WW = patch_B.shape[2]

        # 10
        F = self.causal_context_num + 1
        # patch_B = self.cross_attn[level](patch_B, patch_t0)
        patch_random_mem = self.get_transpose_matrix(level, default=patch_B)
        patch_random_i = torch.cat([patch_t0, patch_random_mem], dim=1)
        # patch_random_i = torch.cat([patch_t0, patch_random_mem], dim=1)

        patch_random_i = rearrange(patch_random_i, "BN F WW C -> (BN F) WW C")

        if self.training and self.use_patch_erase_aug:
            patch_B = random_erase_patch(patch_B[:, 0], p=self.patch_erase_p)[:, None]
        patch_B_repeated = repeat(patch_B, f"BN 1 WW C -> BN {F} WW C")
        patch_B_repeated = rearrange(patch_B_repeated, "BN F WW C -> (BN F) WW C")

        affinity_matrix_iB = self.K(patch_random_i, patch_B_repeated)

        patch_B_reshaped = rearrange(
            patch_B_repeated, "BNF (W V) C -> BNF C W V", W=int(math.sqrt(WW))
        )
        dino_pe_patch = get_pos_enc(
            patch_B_reshaped, self.new_patch_pe_lvln[level], freq=[1, 8, 8][level]
        )
        dino_pe_patch = rearrange(dino_pe_patch, "n c h w -> n (h w) c")
        gps = torch.einsum("bnm,bmd->bnd", affinity_matrix_iB, dino_pe_patch)
        tokens = torch.cat([gps, patch_random_i], dim=-1)

        tokens = rearrange(tokens, "(BN F) WW C -> BN F WW C", F=F)
        troma_block = getattr(self, f"troma_flow_{['16x', '8x', '2x'][level]}")

        # # query position
        coords_init = data["queries"]  # B, N, 2
        # # previous frame position - reshape from [BN, F, 2] to [B, N, 2]
        coords_curr = rearrange(pos_curr[:, 0, :], "(B N) C -> B N C", B=B)
        # # current iteration position - reshape from [BN, F, 2] to [B, N, 2]
        coords_prev = rearrange(data["updated_pos"][:, 0, :], "(B N) C -> B N C", B=B)
        if self.training:
            # Use checkpointing only during training
            # We use a lambda function because checkpoint doesn't support keyword arguments directly
            res, final_tokens_for_decoding, next_walking_context = checkpoint(
                lambda t, ci, cc, cp, ct, lt: troma_block(
                    t,
                    ci,
                    cc,
                    cp,
                    context_tokens=ct,
                    last_walking_context=lt,
                    use_walking_memory=True,
                ),
                tokens,
                coords_init,
                coords_curr,
                coords_prev,
                context_tokens,
                last_walking_context,
                # prior_pos_delta,
                use_reentrant=False,  # Recommended for modern PyTorch versions
            )
        else:
            # Use the standard forward pass during evaluation
            res, final_tokens_for_decoding, next_walking_context = troma_block(
                tokens,
                coords_init,
                coords_curr,
                coords_prev,
                context_tokens=context_tokens,
                last_walking_context=last_walking_context,
                use_walking_memory=True,
            )

        # Clone outputs outside compiled function to avoid CUDA graph tensor overwrite
        # Required when using torch.compile with reduce-overhead mode
        return res.clone(), final_tokens_for_decoding.clone(), next_walking_context.clone()

    def _build_troma_blocks(self, config, lvls_dim):
        """Build TROMA blocks for multi-scale refinement."""
        troma_config = config["troma_blocks"]
        patch_size = self.W**2

        # Build TROMA blocks for each level (16x, 8x, 2x)
        level_names = ["troma_16x", "troma_8x", "troma_2x"]
        for i, level_name in enumerate(level_names):
            level_cfg = troma_config[level_name]
            block_config = {
                "d_model": lvls_dim[i],
                "nhead": level_cfg["nhead"],
                "layer_num": level_cfg["layer_num"],
                "patch_size": patch_size,
                "block_type": level_cfg["block_type"],
                "shared_kv": level_cfg.get("shared_kv", True),
            }
            setattr(
                self,
                f"troma_flow_{level_name.split('_')[1]}",
                ROMATransformerDecoder3(block_config),
            )

        # Get decoder configurations from config
        flow_decoder_cfg = troma_config.get("flow_decoder", {"out_features": 4})
        affinity_decoder_cfg = troma_config.get("affinity_decoder", {"hidden_features": 256})
        patch_pe_cfg = troma_config.get("patch_pe", {"new_pe_dim": 64})

        # Flow decoder and affinity decoder
        self.troma_flow_decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    Mlp(
                        in_features=lvls_dim[i],
                        hidden_features=lvls_dim[i],
                        out_features=flow_decoder_cfg["out_features"],
                    )
                )
                for i in range(self.corr_num_levels)
            ]
        )
        self._zero_init_troma_flow_decoder()
        self.troma_flow_affinity_decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    Mlp(
                        in_features=lvls_dim[i],
                        hidden_features=affinity_decoder_cfg["hidden_features"],
                        out_features=self.W**4,
                    )
                )
                for i in range(self.corr_num_levels)
            ]
        )

        # Positional encoding layers
        self.patch_pe_lvln = nn.ModuleList(
            [torch.nn.Conv2d(2, lvls_dim[i], 1, 1) for i in range(self.corr_num_levels)]
        )
        self.new_patch_pe_lvln = nn.ModuleList(
            [
                torch.nn.Conv2d(2, patch_pe_cfg["new_pe_dim"], 1, 1)
                for i in range(self.corr_num_levels)
            ]
        )

        # Projection layers to match channel dimensions between levels
        self.context_token_projs = nn.ModuleList()
        # Learnable positional embeddings for each context token to signify its level
        self.context_token_pes = nn.ParameterList()

        for i in range(self.corr_num_levels):
            # Add a learnable positional embedding for the context token from each level i
            self.context_token_pes.append(nn.Parameter(torch.randn(1, 1, lvls_dim[i])))
            # Add a projection layer from level i to level i+1
            if i < self.corr_num_levels - 1:
                self.context_token_projs.append(nn.Linear(lvls_dim[i], lvls_dim[i + 1]))

    def get_transpose_matrix(self, level, default=None):
        # Memory size: training uses smaller size for efficiency, inference uses configured size
        train_mem_size = 12
        infer_mem_size = self.causal_memory_size

        if not self.mem_manager.exists(f"causal_corr_{level}"):
            self.mem_manager.push_memory(
                f"causal_corr_{level}",
                value=default,
                detach=False,
                stack_dim=1,
                auto_fill=True,
                custom_size=(train_mem_size if self.training else infer_mem_size),
            )

        # Use 'balanced' strategy for inference: first half recent, second half distant
        # Use 'foremost' for training: consecutive frames for stability
        sample_method = "foremost" if self.training else "balanced"
        patch_random = self.mem_manager.sample_memory(
            key=f"causal_corr_{level}",
            stack_dim=1,
            samples_length=self.causal_context_num,
            default=None,
            sample_method=sample_method,
            drop_out=(0.1 if self.training else 0),
        )
        return patch_random

    def get_affinity_matrix(self, patch_A, patch_B):
        affinity_matrix = self.K(patch_A, patch_B)
        return affinity_matrix

    def _add_jitter(self, coords: torch.Tensor, level: int, image_size: int) -> torch.Tensor:
        """Add Gaussian jitter noise to coordinates for training augmentation.

        Args:
            coords: Coordinates tensor in pixel space, shape [BN, F, 2]
            level: Current refinement level (0=16x, 1=8x, 2=2x)
            image_size: Image size in pixels (height or width, assumed square)

        Returns:
            Jittered coordinates tensor
        """
        # jitter_list contains normalized jitter factors, e.g., [0.03, 0.015, 0.0075]
        # Multiply by image_size to get jitter in pixel space
        scale_factor = self.jitter_list[level] * image_size
        noise = torch.randn_like(coords) * scale_factor
        return coords + noise

    def patch_aug(self, patches_to_aug, B, N):
        """
        apply adversarial augmentation to the input feature patches;
        only activate in training mode, with a certain total probability;

        Args:
            patches_to_aug (torch.Tensor): feature patches to augment, shape [B*N, F, WW, C].
            B (int): original batch size.
            N (int): number of tracking points per sample.

        Returns:
            torch.Tensor: augmented feature patches.
            bool: a boolean value, True if augmentation is performed, False otherwise.
        """
        if not self.training or torch.rand(1) >= self.p_augment:
            return patches_to_aug, False

        rand_val = torch.rand(1)

        if rand_val < self.p_zero_mask:
            augmented_patches = torch.zeros_like(patches_to_aug)

        elif rand_val < self.p_zero_mask + self.p_replace_mask:
            augmented_patches = self.get_distractor_patch(
                patches_to_aug.shape, patches_to_aug.device
            )
        else:
            augmented_patches = self.shuffle_patches_across_points(patches_to_aug, B, N)

        return augmented_patches, True

    def get_distractor_patch(self, shape, device):
        # return random gaussian noise as a distractor patch;
        return torch.randn(shape, device=device) * 0.2

    def shuffle_patches_across_points(self, patches, B, N):
        """
        shuffle the patches across N points;
        """
        if N <= 1:
            return patches

        BN, F, WW, C = patches.shape
        patches_reshaped = patches.view(B, N, F, WW, C)

        indices = torch.stack([torch.randperm(N) for _ in range(B)]).to(patches.device)

        indices_expanded = indices.view(B, N, 1, 1, 1).expand_as(patches_reshaped)
        shuffled_patches_reshaped = torch.gather(patches_reshaped, 1, indices_expanded)

        # (B*N, F, WW, C)
        return shuffled_patches_reshaped.view_as(patches)

    def _zero_init_troma_flow_decoder(self):
        for decoder_mlp in self.troma_flow_decoder:
            logger.debug("Zero-initializing the last layer (fc2) of a flow decoder MLP.")
            nn.init.constant_(decoder_mlp.fc2.weight, 0)
            if decoder_mlp.fc2.bias is not None:
                nn.init.constant_(decoder_mlp.fc2.bias, 0)

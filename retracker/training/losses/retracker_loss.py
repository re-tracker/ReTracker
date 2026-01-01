import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from torch.nn.functional import kl_div
from torch.nn.functional import huber_loss

from .multiclass_focal_loss import build_focal_loss
from retracker.models.utils.local_crop import LocalCrop
from retracker.utils.rich_utils import CONSOLE


class ReTrackerLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config
        self.sparse_spvs = self.config['sparse_spvs']
        self.train_uncertainty = False
        self.model = [kwargs['model']]
        self.device = next(self.model[0].parameters()).device
        self.task_type = config['task_type']
        self.sliding_wz = config['sliding_wz']
        self.local_crop = LocalCrop(corr_num_levels=config['corr_num_levels'], corr_radius=config['corr_radius'])
        self.flow_loss_weights = config['flow_loss_weights']
        self.wo_safe_mask = config['wo_safe_mask']
        
        # keep last consistency loss for logging
        self.last_consistency_loss = torch.tensor(0.0, device=self.device)
        
        # build loss weights
        _cfg = config['multi_focal_loss']
        alpha_list = [_cfg.alpha_class] * _cfg.num_classes + [_cfg.alpha_bin]

        self._init_loss(loss_type='bce', alpha_list=alpha_list, _cfg=_cfg)

        
        # coarse-level
        # self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        self.video_loss_thresh = self.loss_config['video_loss_thresh']
        self.pred_occlusion = False
        # verbose for debug
        self.verbose = False

    def _init_loss(self, loss_type='bce', alpha_list=None, _cfg=None):
        if loss_type == 'bce':
            # Register as buffer so it moves to device automatically, but not saved in state_dict
            self.register_buffer('ce_weights', torch.tensor(alpha_list, dtype=torch.float32), persistent=False)
            self.ignore_index = 4096 # if self.task_type == 'video_matching' else -100
            self.cls_loss_func = self._vanilla_cls_loss

        elif loss_type == 'focal':
            ignore_index = 4096 # if self.task_type == 'video_matching' else -100
            self.cls_loss_func = build_focal_loss(
                alpha=alpha_list,
                gamma=_cfg.gamma,
                reduction='mean',
                ignore_index=ignore_index,
                device=self.device)

    def _vanilla_cls_loss(self, pred_cls_onehot, gt_cls_ids):
        # Ensure weights are on the correct device
        weight = self.ce_weights
        if weight.device != pred_cls_onehot.device:
            weight = weight.to(pred_cls_onehot.device)

        cls_loss = F.cross_entropy(
            pred_cls_onehot, 
            gt_cls_ids, 
            weight=weight, 
            ignore_index=self.ignore_index,
            reduction='none'  # Change to none to inspect individual losses
        )
        cls_loss = cls_loss.mean()
        return cls_loss
 

    def _compute_coarse_loss(self, data, preds):
        coarse_loss_list = []
        loss_c_mconf, loss_c_certainty = self._compute_coarse_cls_and_certainty_loss(
            preds['pred_cls_queries'],
            preds['pred_certainty_queries'],
            preds['gt_cls_map_i_16x_j_8x'],
            preds['gt_cls_ids'],
            preds,
            weight=None)
        coarse_loss_list.append(loss_c_mconf)

        loss_c = (sum(coarse_loss_list) / len(coarse_loss_list)) if len(coarse_loss_list)>0 else 0
        loss_c = loss_c * self.loss_config['coarse_weight']

        return loss_c, loss_c_mconf, loss_c_certainty

    def forward(self, data, preds):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        loss = 0

        if 'gt_cls_map_i_16x_j_8x' in preds.keys() and 'pred_cls_queries' in preds.keys():
            loss_c, loss_c_mconf, loss_c_certainty = self._compute_coarse_loss(data, preds)
            # Guard against numerical issues in coarse loss
            if torch.isnan(loss_c) or torch.isinf(loss_c):
                CONSOLE.print(f"[yellow]Warning: loss_c is NaN or inf: {loss_c}[/yellow]")
                loss_c = loss_c.nan_to_num(0.0)
            if torch.isnan(loss_c_certainty) or torch.isinf(loss_c_certainty):
                CONSOLE.print(
                    f"[yellow]Warning: loss_c_certainty is NaN or inf: {loss_c_certainty}[/yellow]"
                )
                loss_c_certainty = loss_c_certainty.nan_to_num(0.0)

            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
            loss = loss + loss_c
            # count certainty 
            loss = loss + loss_c_certainty

        # 2. fine-level loss
        # loss_f now is for 
        if self.config['train_fine']:
            loss_f = self._compute_fine_loss(data, preds)
        else:
            loss_f = torch.tensor(0.00001).to(self.device)

        # 3. kl loss
        loss_kl = None
       
        # hack: solve inf/nan loss temporarily
        if torch.isnan(loss_f) or torch.isinf(loss_f):
            params = list(self.model[0].parameters())
            _sum = torch.sum(torch.stack([torch.sum(param) for param in params]))
            loss_f = _sum * 0.
            CONSOLE.print("[yellow]solve nan loss by multiplying 0.[/yellow]")
 
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
            # log consistency loss
            loss_scalars.update({"loss_consistency": self.last_consistency_loss.clone().detach().cpu()})

            if 'flow_gt' in data.keys():
                loss_flow = self._compute_flow_loss(data, preds) * self.flow_loss_weights
                loss += loss_flow * self.loss_config['fine_weight']
                loss_scalars.update({"loss_flow":  loss_flow.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        if loss_kl is not None:
            loss += loss_kl * self.loss_config['kl_weight']
            loss_scalars.update({'loss_kl': loss_kl.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

    def causal_video_forward(self, data, preds):
        '''loss function for video tracking,
        sliding_res is a list of results from sliding window;
        '''
        loss_scalars = {}
        loss = 0

        if 'gt_cls_map_i_16x_j_8x' in preds.keys() and 'pred_cls_queries' in preds.keys():
            loss_c, loss_c_mconf, loss_c_certainty = self._compute_coarse_loss(data, preds)
            # Guard against numerical issues in coarse loss
            if torch.isnan(loss_c) or torch.isinf(loss_c):
                CONSOLE.print(f"[yellow]Warning: loss_c (video) is NaN or inf: {loss_c}[/yellow]")
                loss_c = loss_c.nan_to_num(0.0)
            if torch.isnan(loss_c_certainty) or torch.isinf(loss_c_certainty):
                CONSOLE.print(
                    f"[yellow]Warning: loss_c_certainty (video) is NaN or inf: {loss_c_certainty}[/yellow]"
                )
                loss_c_certainty = loss_c_certainty.nan_to_num(0.0)

            loss = loss + loss_c
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
            loss = loss + loss_c_certainty
            loss_scalars.update({"loss_c_certainty": loss_c_certainty.clone().detach().cpu()})

        
        if preds['updated_pos_nlvl'] is not None:
            loss_f = self._compute_fine_loss(data, preds)
            if loss_f is not None:
                loss += loss_f * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
                # log consistency loss
                loss_scalars.update({"loss_consistency": self.last_consistency_loss.clone().detach().cpu()})
                if 'flow_gt' in data.keys() and self.loss_config['compute_flow_loss']:
                    loss_flow = self._compute_flow_loss(data, preds) * self.flow_loss_weights
                    loss += loss_flow * self.loss_config['fine_weight']
                    loss_scalars.update({"loss_flow":  loss_flow.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound


        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss_scalars": loss_scalars})

        # metrics: PCK
        if preds['updated_pos_nlvl'] is not None:
            metrics = self._compute_video_metrics_pck(data, preds)
            data.update({"metrics": metrics})
        return {"loss": loss}

    def _compute_fine_loss(self, data, preds):
        loss_f = self._compute_video_loss_pos_conf(data, preds)
        loss_occ = self._compute_video_loss_occ(data, preds)
        if 'flow_gt' in data.keys():
            loss_f = loss_f * self.loss_config['matching_fine_weight']
        loss_occ = self._compute_video_loss_occ(data, preds)
        loss_consist = self._compute_consistency_loss(preds)
        # store for logging
        self.last_consistency_loss = loss_consist.detach()
        loss_f += loss_occ + loss_consist * 0

        return loss_f

    def _compute_consistency_loss(self, preds):
        mask_lists = preds.get('is_masked_list', None)
        if mask_lists is None:
            return torch.tensor(0.0, device=self.device)

        # Normalize mask_lists to a list of lists of bools
        if isinstance(mask_lists, list) and len(mask_lists) > 0 and isinstance(mask_lists[0], list):
            per_triplet_masks = mask_lists
        elif isinstance(mask_lists, list):
            per_triplet_masks = [mask_lists]
        else:
            return torch.tensor(0.0, device=self.device)

        # Determine number of refinement iters from preds
        all_updated_pos = preds.get('updated_pos_nlvl', None)  # BN, F, n_iters, C
        if all_updated_pos is None or all_updated_pos.dim() < 3:
            return torch.tensor(0.0, device=self.device)
        n_iters = all_updated_pos.shape[2]

        # OR-reduce across triplets for each iter index
        reduced_mask = [False] * n_iters
        for masks in per_triplet_masks:
            for i in range(min(len(masks), n_iters)):
                reduced_mask[i] = reduced_mask[i] or bool(masks[i])

        total_consistency_loss = 0.0
        num_losses = 0
        for iter_idx in range(1, n_iters):
            if reduced_mask[iter_idx]:
                teacher_prediction = all_updated_pos[:, :, iter_idx - 1, :].detach()
                student_prediction = all_updated_pos[:, :, iter_idx, :]
                loss = F.l1_loss(student_prediction, teacher_prediction)
                total_consistency_loss += loss
                num_losses += 1

        if num_losses > 0:
            return total_consistency_loss / num_losses
        else:
            return torch.tensor(0.0, device=self.device)

    def _compute_coarse_cls_and_certainty_loss(self, pred_cls_onehot, pred_certainty, gt_cls_map, gt_cls_ids, preds, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Input:
            pred_cls_onehot (torch.Tensor): (B, F, N, C), logits
            pred_certainty (torch.Tensor): (B, F, N, 1),
            gt_cls_map (torch.Tensor): (B, F, H, W)
            gt_cls_ids (torch.Tensor | None): (BN, F)
        """
        if gt_cls_map is not None: # dense supervise (H*W)
            # transfer gt_cls_map to one-hot (N, HW0, C)
            gt_cls_map = F.one_hot(gt_cls_map, num_classes=pred_cls_onehot.shape[-1]).float() # (N, H0, W0, C=HW1+1)
            gt_cls_map = rearrange(gt_cls_map, 'b f h w c -> (b f) (h w) c')

            # `b_ids/i_ids` come in two shapes depending on the caller:
            # - video path: (BN, F) after stacking over time
            # - image matching path: (BN,) for a single pair
            b_ids = preds.get("b_ids", None)
            i_ids = preds.get("i_ids", None)
            if b_ids is None or i_ids is None:
                raise ValueError("Dense coarse supervision requires preds['b_ids'] and preds['i_ids']")
            if b_ids.dim() == 1:
                b_ids = b_ids[:, None]
            if i_ids.dim() == 1:
                i_ids = i_ids[:, None]

            preds["b_ids"] = rearrange(b_ids, "bn f -> (bn f)")
            preds["i_ids"] = rearrange(i_ids, "bn f -> (bn f)")
            gt_cls_map = gt_cls_map[preds['b_ids'], preds['i_ids']] # BFN, C
            # gt_cls_map = gt_cls_map.reshape(-1, gt_cls_map.shape[-1])
            gt_cls_ids = gt_cls_map.argmax(dim=-1) # BFN

        elif gt_cls_ids is not None: # sparse supervision, (BN, F)
            # 2. sparse supervise (M supervise)
            gt_cls_ids = rearrange(gt_cls_ids, '(b n) f -> (b f n)', b=pred_cls_onehot.shape[0])
            assert gt_cls_ids.dim() == 1
        else:
            ...
            raise ValueError("gt_cls_map and gt_cls_ids cannot be both None")

        # select queries with supervision
        pred_cls_onehot = pred_cls_onehot.reshape(-1, pred_cls_onehot.shape[-1])
        cls_loss = self.cls_loss_func(pred_cls_onehot, gt_cls_ids) 
        
        pred_cls_ids = pred_cls_onehot.argmax(dim=-1)
        # ! too strict for tracking task
        # gt_certainty = (gt_cls_ids == pred_cls_ids).float()
        # loss_certainty = F.binary_cross_entropy_with_logits(pred_certainty.reshape(-1), gt_certainty)
        
        # so we use the following:
        def dist_between_cls(pred_cls_ids, gt_cls_ids, h, w):
            pred_cls_ids_xy = torch.stack([pred_cls_ids%w, pred_cls_ids//w], dim=-1)
            gt_cls_ids_xy = torch.stack([gt_cls_ids%w, gt_cls_ids//w], dim=-1)
            dist = (pred_cls_ids_xy*1. - gt_cls_ids_xy*1.).norm(dim=-1)
            return dist
        dist = dist_between_cls(pred_cls_ids, gt_cls_ids, h=64, w=64)
        gt_certainty = (dist < 6).float() # around 4 * 8 pixels
        loss_certainty = F.binary_cross_entropy_with_logits(pred_certainty.reshape(-1), gt_certainty)
        return cls_loss.mean(), loss_certainty

    def _compute_flow_loss(self, data, preds):
        assert 'updated_pos_nlvl_flow' in preds.keys()
        assert 'flow_gt' in data.keys()

        B = data['images'].shape[0] if 'images' in data else data['image0'].shape[0]
        queries = data['queries']
        flow_gt = data['flow_gt']
        # data['flow_valid_mask'] 通常是 [B, N, H, W] 或类似的，需要确保维度对齐
        flow_valid_mask = data['flow_valid_mask'] * 1.

        # 这里的 flow_gt_with_mask 拼接是为了在 get_crop_patches 里一起被 crop 出来
        flow_gt_with_mask = torch.cat([data['flow_gt'], data['flow_valid_mask']], dim=1)

        # [BN, L, n_iters, HW, 2]
        flow_pred_patches = preds['updated_pos_nlvl_flow']
        # flow_pred_patches has shape [B*N, ...] where B*N is flattened
        # We need N (points per sample) to slice queries which has shape [B, N_total, 2]
        fine_spvs_num = flow_pred_patches.shape[0]  # B*N flattened
        N = fine_spvs_num // B  # N points per sample

        queries = queries[:, :N]

        # GT flows to multilvl flows, sample 7x7 map
        # [B N C H W] -> Crop -> [BN, L, 1, HW, C]
        flow_gt_patches_list = self.local_crop.get_crop_patches(fmaps=flow_gt_with_mask, queries=queries)
        flow_gt_patches_nlvl = torch.stack(flow_gt_patches_list, dim=1) 

        # === 确定当前监督的是哪一层 ===
        # 原代码只取了最后一层 [:,-1:]。
        # 假设你的模型层级顺序是 [Coarse(1/16), Mid(1/8), Fine(1/2)]
        # 那么 -1 对应的是 Fine 层，Stride = 2.0
        # 为了通用性，你可以根据 shape[1] 来推断，或者手动指定
        
        # 原始切片操作
        flow_gt_patches_nlvl = flow_gt_patches_nlvl[:,-1:]
        flow_pred_patches = flow_pred_patches[:,-1:]

        # [BN, L, 1, HW, C] -> [BN, L, 1, HW, C]
        flow_gt_patches = rearrange(flow_gt_patches_nlvl, 'B L N C H W -> (B N) L 1 (H W) C')
        
        # 提取 GT 里的 mask (假设在最后一个通道)
        # flow_gt_patches: [..., :2] 是 flow, [..., 2] 是 mask
        flow_valid_gt_patches = flow_gt_patches[...,2] == 1. 

        n_iters = flow_pred_patches.shape[2]
        gamma = 0.3
        decay_factors = torch.pow(gamma, torch.arange(n_iters - 1, -1, -1)).to(queries)

        # ======================= 计算 Huber Loss =======================
        # [BN, L, n_iters, HW]
        flow_loss_tensor = huber_loss(
            flow_gt_patches[..., :2],
            flow_pred_patches[..., :2],
            delta=4.0,
            reduction='none',
        ) * decay_factors[None, :, None, None] # 注意维度广播

        # Handle NaN
        if torch.isnan(flow_loss_tensor).any() or torch.isinf(flow_loss_tensor).any():
            CONSOLE.print("[yellow]Warning: flow_loss_tensor contains NaN or inf[/yellow]")
            flow_loss_tensor = flow_loss_tensor.nan_to_num(0.0)

        # ============================================================
        # [新增] 基于 Patch 视野范围的 Mask (Anti-Hallucination)
        # ============================================================
        
        # 1. 计算每个像素的预测误差 (Pixel-wise Error)
        # 这里的 flow 都是像素单位的相对位移
        # dist: [BN, L, n_iters, HW]
        dist_pred_gt = torch.norm(flow_pred_patches[..., :2] - flow_gt_patches[..., :2], dim=-1)

        # 2. 定义阈值
        # 假设我们只监督最后一层 (Fine, Stride=2.0)
        # 如果你的代码可能会监督其他层，需要根据 layer index 动态设置 stride
        # 这里按你只取 -1 层 (Fine) 处理：
        current_stride = 2.0  # Fine 层
        
        # 宽容度因子 (Tolerance):
        # Tracking 时我们设为 4.5。Matching 时虽然数据质量高，但因为Coarse预测可能偏，
        # 为了防止 MLP 强行拟合 Patch 外的 GT，我们依然需要这个保护。
        # 建议设为 4.5 ~ 5.0 个 Grid
        tolerance_grid = 5.0 
        valid_threshold = current_stride * tolerance_grid # e.g., 2.0 * 5.0 = 10.0 px

        # 3. 生成 Spatial Mask
        # 如果误差 > 10px，说明 GT 很可能已经跑出 Patch (7*2=14px) 的有效范围了
        # 或者说明当前像素是背景/干扰物，且距离中心太远
        patch_spatial_mask = dist_pred_gt < valid_threshold

        # 4. 合并 Mask
        # 原有的 flow_valid (数据集提供的) AND 我们的 spatial_mask (防漂移的)
        # 注意广播维度: [BN, L, 1, HW] & [BN, L, n_iters, HW]
        final_valid_mask = flow_valid_gt_patches & patch_spatial_mask

        # ============================================================

        if final_valid_mask.any():
            flow_loss = flow_loss_tensor[final_valid_mask].mean()
        else:
            flow_loss = torch.tensor(0.0, device=flow_loss_tensor.device)

        if torch.isnan(flow_loss) or torch.isinf(flow_loss):
            CONSOLE.print(f"[yellow]Warning: flow_loss is NaN or inf: {flow_loss}[/yellow]")
            flow_loss = flow_loss.nan_to_num(0.0)

        return flow_loss * 0.05

    def old_compute_flow_loss(self, data, preds):
        assert 'updated_pos_nlvl_flow' in preds.keys()
        assert 'flow_gt' in data.keys()

        B = data['images'].shape[0] if 'images' in data else data['image0'].shape[0]
        queries = data['queries']
        flow_gt = data['flow_gt']
        flow_valid_mask = data['flow_valid_mask'] * 1. # to float
        flow_gt_with_mask = torch.cat([data['flow_gt'], data['flow_valid_mask']], dim=1)
        flow_pred_patches = preds['updated_pos_nlvl_flow']
        # flow_pred_patches has shape [B*N, ...] where B*N is flattened
        # We need N (points per sample) to slice queries which has shape [B, N_total, 2]
        fine_spvs_num = flow_pred_patches.shape[0]  # B*N flattened
        N = fine_spvs_num // B  # N points per sample

        # same with fine loss, we only supervision a subset of queries due to memory limit
        queries = queries[:, :N]

        # GT flows to multilvl flows, sample 7x7 map
        flow_gt_patches_list = self.local_crop.get_crop_patches(fmaps=flow_gt_with_mask, queries=queries) # [B N C H W]
        flow_gt_patches_nlvl = torch.stack(flow_gt_patches_list, dim=1) # L = len(patch_list)

        # only supervise the last level
        flow_gt_patches_nlvl = flow_gt_patches_nlvl[:,-1:]
        flow_pred_patches = flow_pred_patches[:,-1:]

        flow_gt_patches = rearrange(flow_gt_patches_nlvl, 'B L N C H W -> (B N) L 1 (H W) C')
        flow_valid_gt_patches = flow_gt_patches[...,2] == 1.
                

        n_iters = flow_pred_patches.shape[2]
        gamma = 0.3
        decay_factors = torch.pow(gamma, torch.arange(n_iters - 1, -1, -1)).to(queries)

        # debug: 
        debug = False
        if debug:
            # only supervise fine level
            flow_valid_gt_patches = flow_valid_gt_patches[:,2]
            # only supervise finest lvl
            flow_loss = (huber_loss(flow_gt_patches[...,:2], flow_pred_patches[...,:2], delta=4.0, reduction='none'))[:,2,][...,24,:][flow_valid_gt_patches.bool()[...,24]].mean()

            # flow_loss = (huber_loss(flow_gt_patches[...,:2], flow_pred_patches[...,:2], delta=4.0, reduction='none'))[:,2,][flow_valid_gt_patches.bool()].mean()
            # prediction similarity:
            # assert (flow_pred_patches[:,2,:,24] - data['updated_pos_nlvl'][:,:,2])[flow_valid_gt_patches.bool()[...,24]].max()<1., "debug prediction consistency"

            # GT similarity:
            assert (flow_gt_patches[:,2,...,24,:2][:,0,1] - data['trajs'][0,0][...][...,1] ).max() < 10
        else:
            # Compute per-sample flow loss with temporal decay
            flow_loss_tensor = huber_loss(
                flow_gt_patches[..., :2],
                flow_pred_patches[..., :2],
                delta=4.0,
                reduction='none',
            ) * decay_factors[None, :, None]

            # Replace NaN/Inf in the tensor to avoid propagating invalid values
            if torch.isnan(flow_loss_tensor).any() or torch.isinf(flow_loss_tensor).any():
                CONSOLE.print("[yellow]Warning: flow_loss_tensor contains NaN or inf[/yellow]")
                flow_loss_tensor = flow_loss_tensor.nan_to_num(0.0)

            valid_mask = flow_valid_gt_patches.bool()
            if valid_mask.any():
                flow_loss = flow_loss_tensor[valid_mask].mean()
            else:
                # No valid flow supervision in this batch
                flow_loss = torch.tensor(0.0, device=flow_loss_tensor.device)

            if torch.isnan(flow_loss) or torch.isinf(flow_loss):
                CONSOLE.print(f"[yellow]Warning: flow_loss is NaN or inf: {flow_loss}[/yellow]")
                flow_loss = flow_loss.nan_to_num(0.0)

        return flow_loss * 0.05

    '''loss: post iccv version'''
    def _new_compute_video_loss_pos_conf(self, data, preds):
        '''
        loss: add gating weight to position loss
        '''
        if data['trajs'].shape[1] == 3 and preds['updated_pos_nlvl'].shape[1] == 1:
            # tmp code for matching fine
            data['trajs'] = data['trajs'][:, -1:]
        # devide data['trajs'] to same chunks as sliding_res does;

        B, T, _, H, W = data['images'].shape

        # sample some points for fine training
        # preds['updated_pos_nlvl'] has shape [B*N, F, nlvl, C] where B*N is flattened
        # We need N (points per sample) to slice data['trajs'] which has shape [B, F, N_total, C]
        fine_supervision_BN = preds['updated_pos_nlvl'].shape[0]  # B*N flattened
        N = fine_supervision_BN // B  # N points per sample
        data['trajs'] = data['trajs'][:, :, :N]

        gt_traj = data['trajs'] # B F N C
        gt_vis = (gt_traj[...,0]>-W*0.1) & (gt_traj[...,0]<W*1.1) & (gt_traj[...,1]>-H*0.1) & (gt_traj[...,1]<H*1.1) # B F N
        
        # Predictions
        pred_occ = preds['updated_occ_nlvl'] # BN F nlvl C
        pred_vis = 1 - torch.sigmoid(pred_occ) # Predicted visibility (1 - occlusion)
        # pred_vis = 1-torch.sigmoid(preds['fine_occlusion_logits']) # BN F 1
        res = preds['updated_pos_nlvl'] # BN F nlvl C
        res_exp = preds['updated_exp_nlvl'] # BN F nlvl C
        if gt_traj.shape[1] != res.shape[1]: # for new version we omit the loss of first frame;
            gt_traj = gt_traj[:,1:]
            gt_vis = gt_vis[:,1:]
        n_iters = res.shape[2]
        gt_traj = repeat(gt_traj, f'B F N C -> (B N F) {n_iters} C') # C = 2
        gt_vis = repeat(gt_vis, f'B F N -> (B N F) {n_iters}') # C = 2
        # pred_vis = repeat(pred_vis, f'BN F 1 -> (BN F) {n_iters}')
        res = rearrange(res, 'BN F i C -> (BN F) i C')
        res_exp = rearrange(res_exp, 'BN F i C -> (BN F) i C')
        ## print 4 scales loss
        # errs = (res-gt_traj).norm(dim=-1).mean(0)
        # print(errs)
        # reweight loss:
        gamma = 0.8
        decay_factors = torch.pow(gamma, torch.arange(n_iters - 1, -1, -1)).to(res)

        # ======================= POSITION LOSS =======================
        # 1. Approaching L1 behavior for errors > 0.5)
        raw_huber_loss = huber_loss(gt_traj, res, delta=0.5, reduction='none') 
        
        # 2. Confidence Gating: 
        # clamp(min=0.01) 
        pred_conf = torch.sigmoid(res_exp).squeeze(-1).detach() # [(BN F), n_iters]
        gating_weight = torch.clamp(pred_conf, min=0.01) + 0.5
        
        # Huber * Decay * Gating
        huber_loss_tensor = raw_huber_loss * decay_factors[None,:,None] * gating_weight[..., None]
        
        if torch.isnan(huber_loss_tensor).any() or torch.isinf(huber_loss_tensor).any():
            CONSOLE.print("[yellow]Warning: huber_loss_tensor contains NaN or inf[/yellow]")
            huber_loss_tensor = huber_loss_tensor.nan_to_num(0.0)
        
        in_frame_mask = gt_vis.bool()
        if in_frame_mask.sum() > 0:
            position_loss = huber_loss_tensor[in_frame_mask].mean()
        else:
            position_loss = torch.tensor(0.0).to(res.device)

        if torch.isnan(position_loss) or torch.isinf(position_loss):
            CONSOLE.print(f"[yellow]Warning: position_loss is NaN or inf: {position_loss}[/yellow]")
            position_loss = position_loss.nan_to_num(0.0)
        
        # ======================= CERTAINTY LOSS (Modified) =======================
        # Certainty loss (Optimized Version)
        in_frame_mask_flat = in_frame_mask.reshape(-1)

        if in_frame_mask_flat.sum() > 0:
            dist_gt_in_frame = (res.reshape(-1, 2)[in_frame_mask_flat] - 
                                gt_traj.reshape(-1, 2)[in_frame_mask_flat]).norm(dim=-1)
            
            # 3. Soft Label: 
            # 0px -> 1.0; 2px -> 0.36; 4px -> 0.13
            gt_certainty_in_frame = torch.exp(-0.2 * dist_gt_in_frame)
            res_exp_in_frame = res_exp.reshape(-1)[in_frame_mask_flat]
            
            exp_loss = F.binary_cross_entropy_with_logits(res_exp_in_frame, gt_certainty_in_frame)
        else:
            exp_loss = torch.tensor(0.0).to(res.device)
            
        return position_loss * 1. + exp_loss

    def _compute_video_loss_pos_conf(self, data, preds):
        ''' loss: iccv version'''
        if data['trajs'].shape[1] == 3 and preds['updated_pos_nlvl'].shape[1] == 1:
            # tmp code for matching fine
            data['trajs'] = data['trajs'][:, -1:]
        # devide data['trajs'] to same chunks as sliding_res does;

        B, T, _, H, W = data['images'].shape

        # sample some points for fine training
        # preds['updated_pos_nlvl'] has shape [B*N, F, nlvl, C] where B*N is flattened
        # We need N (points per sample) to slice data['trajs'] which has shape [B, F, N_total, C]
        fine_supervision_BN = preds['updated_pos_nlvl'].shape[0]  # B*N flattened
        N = fine_supervision_BN // B  # N points per sample
        data['trajs'] = data['trajs'][:, :, :N]
        gt_traj = data['trajs'] # B F N C
        gt_vis = (gt_traj[...,0]>-W*0.1) & (gt_traj[...,0]<W*1.1) & (gt_traj[...,1]>-H*0.1) & (gt_traj[...,1]<H*1.1) # B F N
        
        # Predictions
        pred_occ = preds['updated_occ_nlvl'] # BN F nlvl C
        pred_vis = 1 - torch.sigmoid(pred_occ) # Predicted visibility (1 - occlusion)
        # pred_vis = 1-torch.sigmoid(preds['fine_occlusion_logits']) # BN F 1
        res = preds['updated_pos_nlvl'] # BN F nlvl C
        res_exp = preds['updated_exp_nlvl'] # BN F nlvl C
        if gt_traj.shape[1] != res.shape[1]: # for new version we omit the loss of first frame;
            gt_traj = gt_traj[:,1:]
            gt_vis = gt_vis[:,1:]
        n_iters = res.shape[2]
        gt_traj = repeat(gt_traj, f'B F N C -> (B N F) {n_iters} C') # C = 2
        gt_vis = repeat(gt_vis, f'B F N -> (B N F) {n_iters}') # C = 2
        # pred_vis = repeat(pred_vis, f'BN F 1 -> (BN F) {n_iters}')
        res = rearrange(res, 'BN F i C -> (BN F) i C')
        res_exp = rearrange(res_exp, 'BN F i C -> (BN F) i C')
        ## print 4 scales loss
        # errs = (res-gt_traj).norm(dim=-1).mean(0)
        # print(errs)
        # reweight loss:
        gamma = 0.8
        decay_factors = torch.pow(gamma, torch.arange(n_iters - 1, -1, -1)).to(res)

        # ======================= POSITION LOSS =======================
        huber_loss_tensor = huber_loss(gt_traj, res, delta=4.0, reduction='none') * decay_factors[None,:,None]
        
        if self.wo_safe_mask:
            # changed huber loss:
            # 1. 计算 Coarse 部分 (前 n-1 层) -> 使用 delta=4.0
            # 切片 [:, :-1, :]
            loss_coarse = huber_loss(
                gt_traj[:, :-1], 
                res[:, :-1], 
                delta=4.0, 
                reduction='none'
            )

            # 2. 计算 Fine 部分 (最后一层) -> 使用 delta=0.5
            # 切片 [:, -1:, :] (保持维度以便 concat)
            loss_fine = huber_loss(
                gt_traj[:, -1:], 
                res[:, -1:], 
                delta=0.5, 
                reduction='none'
            )

            # 3. 拼接回原始形状
            # [BN F, n_iters-1, 2] + [BN F, 1, 2] -> [BN F, n_iters, 2]
            huber_loss_tensor = torch.cat([loss_coarse, loss_fine], dim=1)

            # 4. 乘以衰减系数
            huber_loss_tensor = huber_loss_tensor * decay_factors[None, :, None]
            # =============================================================

            if torch.isnan(huber_loss_tensor).any() or torch.isinf(huber_loss_tensor).any():
                CONSOLE.print("[yellow]Warning: huber_loss_tensor contains NaN or inf[/yellow]")
                huber_loss_tensor = huber_loss_tensor.nan_to_num(0.0)
            
            in_frame_mask = gt_vis.bool()
            if in_frame_mask.sum() > 0:
                position_loss = huber_loss_tensor[in_frame_mask].mean()
            else:
                position_loss = torch.tensor(0.0).to(res.device)

            if torch.isnan(position_loss) or torch.isinf(position_loss):
                CONSOLE.print(f"[yellow]Warning: position_loss is NaN or inf: {position_loss}[/yellow]")
                position_loss = position_loss.nan_to_num(0.0)
            
            # =============================================================
            # Certainty loss (Optimized Version)
            in_frame_mask_flat = in_frame_mask.reshape(-1)

            if in_frame_mask_flat.sum() > 0:
                dist_gt_in_frame = (res.reshape(-1, 2)[in_frame_mask_flat] - 
                                    gt_traj.reshape(-1, 2)[in_frame_mask_flat]).norm(dim=-1)
                
                mask_in_frame = dist_gt_in_frame < 6
                gt_certainty_in_frame = mask_in_frame.float()
                
                res_exp_in_frame = res_exp.reshape(-1)[in_frame_mask_flat]
                
                exp_loss = F.binary_cross_entropy_with_logits(res_exp_in_frame, gt_certainty_in_frame)
            else:
                exp_loss = torch.tensor(0.0).to(res.device)
                
            return position_loss * 0.05 + exp_loss
        else:
            # === 新增代码: 基于 Patch 范围的有效性过滤 ===
            # 1. 计算预测值与 GT 的欧氏距离
            # res: [(BN F), n_iters, 2], gt_traj: [(BN F), n_iters, 2]
            dist_pred_gt = torch.norm(res - gt_traj, p=2, dim=-1) # Shape: [(BN F), n_iters]

            # 2. 定义各层级的 Stride (下采样倍率)
            # 假设 n_iters 分别对应 [1/16, 1/8, 1/2] 的输出
            # 如果你的 n_iters 数量多于3次（例如每层迭代多次），你需要扩展这个列表以匹配 res.shape[1]
            strides_list = [16.0, 8.0, 2.0] 
            
            # 简单兼容性处理：如果迭代次数不匹配，默认使用最后一种或者循环使用（根据你的实际模型结构调整）
            if n_iters != len(strides_list):
                # 如果迭代次数是层级的倍数，例如每层迭代2次共6次: [16,16, 8,8, 2,2]
                # 这里做一个简单的容错，如果长度不一致，需要你手动根据模型配置修改 strides_list
                if n_iters > len(strides_list): 
                    # 假设最后一层迭代多次，或者简单的填充
                    strides_list = strides_list + [strides_list[-1]] * (n_iters - len(strides_list))
                else:
                    strides_list = strides_list[:n_iters]

            strides_tensor = torch.tensor(strides_list, device=res.device).view(1, n_iters)

            # 3. 计算阈值
            # Patch 半径 = 7 // 2 = 3.5。
            # 我们稍微放宽一点到 4.0，避免在边缘处截断正常的梯度
            tolerance_factors = torch.tensor([4.0, 4.0, 4.0], device=res.device).view(1, n_iters)
            valid_thresholds = strides_tensor * tolerance_factors # [1, n_iters]
            patch_valid_mask = dist_pred_gt < valid_thresholds

            # patch_radius_grid = 4.0 
            # valid_thresholds = strides_tensor * patch_radius_grid # Shape: [1, n_iters] (e.g., [64, 32, 8])

            # # 4. 生成 Mask: 只有距离小于阈值的点才参与计算
            # # 如果距离太远，说明 GT 不在 Patch 内，甚至不在 Patch 边缘，这时的 Attention 是无效的
            # patch_valid_mask = dist_pred_gt < valid_thresholds

            # 5. 合并 Mask: 既要在帧内 (gt_vis)，又要 Patch 有效
            final_loss_mask = gt_vis.bool() & patch_valid_mask
            # ========================================================

            if final_loss_mask.sum() > 0:
                # 使用新的 mask 计算平均 Loss
                position_loss = huber_loss_tensor[final_loss_mask].mean()
            else:
                position_loss = torch.tensor(0.0).to(res.device)

            if torch.isnan(position_loss) or torch.isinf(position_loss):
                CONSOLE.print(f"[yellow]Warning: position_loss is NaN or inf: {position_loss}[/yellow]")
                position_loss = position_loss.nan_to_num(0.0)
            
            # =============================================================
            # Certainty loss (Optimized Version)
            # 注意：这里是否应用 patch_valid_mask 取决于你的需求。
            # 通常 Certainty 也应该只在 Patch 有效时才有意义，或者说离得远本身就应该预测低置信度。
            # 原代码使用了 dist_gt_in_frame < 6 的硬阈值，这本身就是一种保护，所以这里可以保持原样
            # 或者如果你希望更加一致，也可以用 final_loss_mask
            
            in_frame_mask_flat = gt_vis.bool().reshape(-1) # 保持原逻辑，或者改成 final_loss_mask.reshape(-1)

            if in_frame_mask_flat.sum() > 0:
                # 原有逻辑保持不变
                dist_gt_in_frame = (res.reshape(-1, 2)[in_frame_mask_flat] - 
                                    gt_traj.reshape(-1, 2)[in_frame_mask_flat]).norm(dim=-1)
                
                mask_in_frame = dist_gt_in_frame < 6
                gt_certainty_in_frame = mask_in_frame.float()
                
                res_exp_in_frame = res_exp.reshape(-1)[in_frame_mask_flat]
                
                exp_loss = F.binary_cross_entropy_with_logits(res_exp_in_frame, gt_certainty_in_frame)
            else:
                exp_loss = torch.tensor(0.0).to(res.device)
                
            return position_loss * 0.05 + exp_loss
    
    def _compute_video_loss_occ(self, data, preds):
        if data['visibles'].shape[1] == 3 and preds['updated_occ_nlvl'].shape[1] == 1:
            # tmp code for matching fine
            data['visibles'] = data['visibles'][:, -1:]

        B, T = data['images'].shape[:2]

        # sample some points for fine training
        # preds['updated_pos_nlvl'] has shape [B*N, F, nlvl, C] where B*N is flattened
        # We need N (points per sample) to slice data which has shape [B, F, N_total]
        fine_supervision_BN = preds['updated_pos_nlvl'].shape[0]  # B*N flattened
        N = fine_supervision_BN // B  # N points per sample
        data['visibles'] = data['visibles'][:, :, :N]
        data['valids'] = data['valids'][:, :, :N]
        gt_occ = 1-data['visibles']*1.
        # pred_occ = preds['fine_occlusion_logits'][...,0] # BN F
        pred_occ = preds['updated_occ_nlvl'] # BN F nlvl C

        if gt_occ.shape[1] != pred_occ.shape[1]: # for new version we omit the loss of first frame;
            gt_occ = gt_occ[:,1:]
        n_iters = pred_occ.shape[2]
        gt_occ = repeat(gt_occ, f'B F N-> (B N F) {n_iters} {1}') # B F N -> BNF 1
        # gt_occ = repeat(gt_occ, f'B F N C -> (B N F) {n_iters} C') # C = 2
        pred_occ = rearrange(pred_occ, 'BN F i C -> (BN F) i C')
        occ_loss = F.binary_cross_entropy_with_logits(pred_occ.reshape(-1), gt_occ.reshape(-1))
        return occ_loss

    def _compute_video_metrics_pck(self, data, preds, verbose=False):
        B, T = data['images'].shape[:2]

        # preds['updated_pos_nlvl'] has shape [B*N, F, nlvl, C] where B*N is flattened
        # We need N (points per sample) to slice data which has shape [B, F, N_total, C]
        fine_supervision_BN = preds['updated_pos_nlvl'].shape[0]  # B*N flattened
        N = fine_supervision_BN // B  # N points per sample

        gt_traj = data['trajs'][:, :, :N]  # B F N C
        gt_valids = data['valids'][:, :, :N]  # B F N
        res = preds['updated_pos_nlvl'] # BN F nlvl C
        if gt_traj.shape[1] != res.shape[1]: # for new version we omit the loss of first frame;
            gt_traj = gt_traj[:,1:]
            gt_valids = gt_valids[:,1:]
        n_iters = res.shape[2]
        gt_traj = repeat(gt_traj, f'B F N C -> (B N F) {n_iters} C') # C = 2
        gt_valids = rearrange(gt_valids, f'B F N -> (B N F)') 
        res = rearrange(res, 'BN F i C -> (BN F) i C')
        
        # cacualte epes 
        if verbose:
            l2_dist = (res - gt_traj).norm(dim=-1)
            CONSOLE.print(f"[dim]{l2_dist.max()=}, {l2_dist.min()=}[/dim]")

        epes = (res-gt_traj).norm(dim=-1)
        # only cacualte valid epes:
        epes_all = epes[gt_valids.bool()] # BNF, n_iters
        epes = epes_all.mean(0) # n_iters
        metric_dict = {'Fine/epe': epes[-1]}
        for iter_i in range(n_iters):
            metric_dict.update({f'Fine/epe@iter_{iter_i}': epes[iter_i]})
        
        # caculate PCKs
        thresh = [1,2,4,8,16]
        for th in thresh:
            metric_dict.update({f'Fine/pck@{th}': (epes_all[:,-1]<th).sum() / len(epes_all[:,-1])})
        return metric_dict

import torch
from einops import repeat, rearrange
from torch.cuda.amp import autocast

from retracker.training.geometry import warp_kpts, homo_warp_kpts, homo_warp_kpts_glue, warp_grid
from retracker.utils.rich_utils import CONSOLE

##############  ↓  Coarse-Level supervision  ↓  ##############
@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


def add_items_for_matching_task(data):
    # image_m
    if 'image_m' not in data.keys():
        data['image_m'] = data['image0']

    if 'is_first_frame' not in data.keys():
        data['is_first_frame'] = torch.ones(data['image0'].shape[0], dtype=torch.bool, device=data['image0'].device)
        data['is_last_frame'] = data['is_first_frame']
        data['frame_num'] = [2] #s


@torch.no_grad()
def spvs_coarse_trajs(data, scale, suffix=''):
    """Build coarse GT matrix and remove unvisible points from supervision;
    Input: 
        data(dict):{
            'image';
            'trajs': [B, S ,N, 2]   (x,y) order
            'occs': [B, S, 1, H, W]
            'masks': [B, S, 1, H, W]
            'visibles': [B, S, N]
            'valids': [B, S, N]
        },
        scale(int): scale = HW_img / HW_spvs_matrix,
        
    
    1. build GT corresponding matrix ()
    2. return all queries with 
    
    Update:
        data(dict):{
            'conf_matrix_gt_01': [N, hw0, hw1], 
        }
    """
    # 1. misc
    device = data['image0'].device
    B, _, H0, W0 = data['image0'].shape
    _, _, HM, WM = data['image_m'].shape # useless here
    _, _, H1, W1 = data['image1'].shape

    h0, w0, hm, wm, h1, w1 = map(lambda x: x // scale, [H0, W0, HM, WM, H1, W1])

    # 2. 
    def clip_kpts(kpts, hw):
        torch.clamp_(kpts[...,0], min=0, max=hw[1]-1)
        torch.clamp_(kpts[...,1], min=0, max=hw[0]-1)
        return kpts
    pt0_c = torch.div(data['trajs'][:, 0]+scale/2, scale, rounding_mode='trunc').long() # B,N,2
    ptm_c = torch.div(data['trajs'][:, 1]+scale/2, scale, rounding_mode='trunc').long() # 
    pt1_c = torch.div(data['trajs'][:, 2]+scale/2, scale, rounding_mode='trunc').long() # 
    pt0_c = clip_kpts(pt0_c, (h0, w0))
    ptm_c = clip_kpts(ptm_c, (hm, wm))
    pt1_c = clip_kpts(pt1_c, (h1, w1))

    # y*w + x
    ids_BN = pt0_c[...,1] * w0 + pt0_c[...,0]  # B,N
    
    b_ids = torch.arange(B,device=device).expand(ids_BN.shape[1], B).T.long().reshape(-1) # B N
    ids_0 = (pt0_c[...,1] * w0 + pt0_c[...,0]).reshape(-1)
    ids_m = (ptm_c[...,1] * wm + ptm_c[...,0]).reshape(-1)
    ids_1 = (pt1_c[...,1] * w1 + pt1_c[...,0]).reshape(-1)

    # 4. construct a gt conf_matrix
    visibles_mask_0 = data['valids'][:,0].reshape(-1).bool() # 
    visibles_mask_m = data['valids'][:,1].reshape(-1).bool() # 
    visibles_mask_1 = data['valids'][:,2].reshape(-1).bool() # 
    visibles_mask_0m = torch.logical_and(visibles_mask_0, visibles_mask_m)
    visibles_mask_m1 = torch.logical_and(visibles_mask_m, visibles_mask_1)
    visibles_mask_01 = torch.logical_and(visibles_mask_0, visibles_mask_1)

    b_ids_0m, i_ids_0m, j_ids_0m = b_ids[visibles_mask_0m], ids_0[visibles_mask_0m], ids_m[visibles_mask_0m]
    b_ids_m1, i_ids_m1, j_ids_m1 = b_ids[visibles_mask_m1], ids_m[visibles_mask_m1], ids_1[visibles_mask_m1]
    b_ids_01, i_ids_01, j_ids_01 = b_ids[visibles_mask_01], ids_0[visibles_mask_01], ids_1[visibles_mask_01]

    conf_matrix_gt_0m = torch.zeros(B, h0*w0, h1*w1+1, device=device)
    conf_matrix_gt_0m[..., h1*w1] = 1
    conf_matrix_gt_0m[b_ids_0m, i_ids_0m, j_ids_0m] = 1

    # assert m and 1 are not occluded either;
    conf_matrix_gt_m1 = torch.zeros(B, h0*w0, h1*w1+1, device=device)
    conf_matrix_gt_m1[..., h1*w1] = 1
    conf_matrix_gt_m1[b_ids_m1, i_ids_m1, j_ids_m1] = 1
    
    conf_matrix_gt_01 = torch.zeros(B, h0*w0, h1*w1+1, device=device)
    conf_matrix_gt_01[..., h1*w1] = 1
    conf_matrix_gt_01[b_ids_01, i_ids_01, j_ids_01] = 1
    data.update({
        f'conf_matrix_gt_0m{suffix}':conf_matrix_gt_0m, 
        f'conf_matrix_gt_m1{suffix}':conf_matrix_gt_m1, 
        f'conf_matrix_gt_01{suffix}': conf_matrix_gt_01})

    if scale == 8:
        data.update({
            f'gt_cls_map_i_16x_j_8x': None,
            f'gt_cls_map_8x_2D': None,
            f'gt_cls_ids': ids_1,
            f'gt_cls_ids_vis': visibles_mask_01,
        })

        # gt_cls_map = conf_matrix_gt_01.float().argmax(dim=-1) # [N, hw0]
        # data.update({
        #     # f'gt_mconf_map{suffix}': gt_mconf_map,
        #     f'gt_cls_map': gt_cls_map,
        # })
        # gt_cls_map_8x_2D = rearrange(gt_cls_map, 'B (h0 w0) -> B h0 w0', h0=H0//8, w0=W0//8)
        # gt_cls_map_i_16x_j_8x = gt_cls_map_8x_2D[:, ::2, ::2] # B, 1024
        # data.update({
        #     f'gt_cls_map_i_16x_j_8x': gt_cls_map_i_16x_j_8x,
        #     f'gt_cls_map_8x_2D': gt_cls_map_8x_2D,
        # })

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids_0m) == 0:
        CONSOLE.print("[yellow]No groundtruth 0m coarse match found.[/yellow]")
    if len(b_ids_m1) == 0:
        CONSOLE.print("[yellow]No groundtruth m1 coarse match found.[/yellow]")
    if len(b_ids_01) == 0:
        CONSOLE.print("[yellow]No groundtruth 01 coarse match found.[/yellow]")

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'gt_position_m': data['trajs'][:, 1],
        'gt_position': data['trajs'][:, 2],
        'gt_occlusion_m': ~data['valids'][:, 1],
        'gt_occlusion': ~data['valids'][:, 2],
    })


def compute_supervision_coarse(data, config):
    # assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    scale_dino = config.retracker_config.conf_matrix_gt_dino_scale # coarse_classification_scale
    scale = config.retracker_config.conf_matrix_gt_scale # coarse_classification_scale
    if data_source.lower() in ['scannet', 'flyingthings', 'movi-e', 'pointodyssey', 'k-epic']:
        spvs_coarse_trajs(data, scale_dino, '_dino')
        spvs_coarse_trajs(data, scale, '')
    elif data_source.lower() in ['megadepth']:
        spvs_trajs_from_matching_dataset(data, [16, 8], ['_16x',''], config) # provide more trajs than dino first
        # product dense flow for supervision
        spvs_flow(data, on_reference=True) # reference/target frame
    else:
        raise ValueError(f'Unknown data source: {data_source}')

@torch.no_grad()
@autocast(enabled=False)
def spvs_trajs_from_matching_dataset(data, scales_list, suffix_list, config, trajs_sample_grid=8):
    """ caculate and update coarse and fine supervision:
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1], for coarse supervision
            "trajs": [N, hw0, 2], for fine supervision
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    M = config['fixed_coarse_spvs_num']
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    for scale, suffix in zip(scales_list, suffix_list):
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

        # 2. warp grids
        # create kpts in meshgrid and resize them to image resolution
        grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
        grid_pt0_i = scale0 * grid_pt0_c

        correct_0to1 = torch.zeros((grid_pt0_i.shape[0], grid_pt0_i.shape[1]), dtype=torch.bool, device=grid_pt0_i.device)
        w_pt0_i = torch.zeros_like(grid_pt0_i)
        invalid_dpt_b_mask = data['T_0to1'].sum(dim=-1).sum(dim=-1) == 0
        if invalid_dpt_b_mask.sum() != 0:
            if data['homography'].sum()==0:
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts(grid_pt0_i[invalid_dpt_b_mask], data['norm_pixel_mat'][invalid_dpt_b_mask], data['homo_sample_normed'][invalid_dpt_b_mask], original_size1=data['origin_img_size1'][invalid_dpt_b_mask])
            else:
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_glue(grid_pt0_i[invalid_dpt_b_mask], data['homography'][invalid_dpt_b_mask], original_size1=data['origin_img_size1'][invalid_dpt_b_mask])
            correct_0to1[invalid_dpt_b_mask] = correct_0to1_homo
            w_pt0_i[invalid_dpt_b_mask] = w_pt0_i_homo
        if (~invalid_dpt_b_mask).sum() != 0:
            correct_0to1_dpt, w_pt0_i_dpt = warp_kpts(grid_pt0_i[~invalid_dpt_b_mask], data['depth0'][~invalid_dpt_b_mask], data['depth1'][~invalid_dpt_b_mask], data['T_0to1'][~invalid_dpt_b_mask], data['K0'][~invalid_dpt_b_mask], data['K1'][~invalid_dpt_b_mask])
            correct_0to1[~invalid_dpt_b_mask] = correct_0to1_dpt
            w_pt0_i[~invalid_dpt_b_mask] = w_pt0_i_dpt

        w_pt0_c = w_pt0_i / scale1

        # 3. check if mutual nearest neighbor
        w_pt0_c_round = w_pt0_c[:, :, :].round() # [N, hw, 2]
        w_pt0_c_round = w_pt0_c_round.long() # [N, hw, 2]
        nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1 # [N, hw]

        def out_bound_mask(pt, w, h):
            return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
        # w_pt1_c_round = w_pt1_c[:, :, :].round().long()
        # nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0
        # nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0
        # loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
        # correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
        correct_0to1[:, 0] = False  # ignore the top-left corner
    
        # 4. construct a gt conf_matrix
        # mask1 = torch.stack([data[f'mask1{suffix}'].reshape(-1, h1*w1)[_b, _i] for _b, _i in enumerate(nearest_index1)], dim=0)
        # correct_0to1 = correct_0to1 * data[f'mask0{suffix}'].reshape(-1, h0*w0) * mask1
        correct_0to1 = correct_0to1
        conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1+1, device=device)

        # positive pairs
        b_ids, i_ids = torch.where(correct_0to1 != 0)
        j_ids = nearest_index1[b_ids, i_ids]
        conf_matrix_gt[b_ids, i_ids, j_ids] = 1
        
        # negative pairs
        b_ids_neg, i_ids_neg = torch.where(correct_0to1 == 0)
        conf_matrix_gt[b_ids_neg, i_ids_neg, -1] = 1
        # data.update({
        #     f'conf_matrix_gt_01{suffix}': conf_matrix_gt})
    
        # build gt_cls_map and gt_mconf_map
        gt_cls_map = conf_matrix_gt.float().argmax(dim=-1) # [N, hw0]
        data.update({
            # f'gt_mconf_map{suffix}': gt_mconf_map,
            # f'conf_matrix_gt_01{suffix}': conf_matrix_gt,
            f'gt_cls_map{suffix}': gt_cls_map,
        })

        # 5. save coarse matches(gt) for training fine level
        if len(b_ids) == 0:
            CONSOLE.print(f"[yellow]No groundtruth coarse match found for: {data['pair_names']}[/yellow]")
        
        # 6. add trajs to dict if scale = trajs_sample_grid
        if scale == trajs_sample_grid:
            # caculate the precise matches for resized image(H, W)
            grid_pt0_i_resized = grid_pt0_i / data['scale0'][:, None] # [B, m, 2]
            w_pt0_i_resized = w_pt0_i / data['scale1'][:, None] #

            # random select m points for each member of Batch, if m < M, padding to M:
            trajs_0, trajs_1 = [], []
            for _b in range(N):
                pt0 = grid_pt0_i_resized[_b]
                pt1 = w_pt0_i_resized[_b]
                trajs_0.append(pt0)
                trajs_1.append(pt1)

            trajs_0, trajs_1 = torch.stack(trajs_0, dim=0), torch.stack(trajs_1, dim=0) # [B, hw, 2]
            valid = correct_0to1

            # rerank M kpts, valids first;
            sort_index = torch.sort(valid.float(), descending=True).indices # B, M
            # rerank trajs [B, M, 2] with sort_index [B, M]
            trajs_0 = trajs_0[torch.arange(N)[:, None], sort_index]
            trajs_1 = trajs_1[torch.arange(N)[:, None], sort_index]
            valid = valid[torch.arange(N)[:, None], sort_index]
            
            trajs_all = torch.stack([trajs_0, trajs_0, trajs_1], dim=1) # [B, F=3, hw, 2]
            valids_all = torch.stack([valid, valid, valid], dim=1) # [B, F, hw]
            B, F, hw = valids_all.shape
            trajs_M = torch.zeros((B, F, M, 2), device=valid.device)
            valids_M = torch.zeros(B, F, M, device=valid.device, dtype=torch.bool)

            # random select M kpts from hw
            for i in range(trajs_all.shape[0]):
                # 1 select M kpts from trajs, valid matches has higher priority
                random_mask = torch.multinomial(valid[i].float()+0.05, num_samples=M, replacement=True)
                trajs_M[i] = trajs_all[i, :, random_mask]
                valids_M[i] = valids_all[i, :, random_mask]
                
            data.update({f'trajs': trajs_M}) # B, F, M, 2
            data.update({f'visibles': valids_M}) # B, F, M
            data.update({f'valids': valids_M}) # B, F, M
            data.update({f'occs': ~valids_M}) # B, F, M

            if 'gt_position' not in data.keys():
                data.update({
                    'gt_position_m': data['trajs'][:, 1],
                    'gt_position': data['trajs'][:, 2],
                    'gt_occlusion_m': ~data['valids'][:, 1],
                    'gt_occlusion': ~data['valids'][:, 2],
                    'gt_valid_m': data['valids'][:, 1],
                    'gt_valid': data['valids'][:, 2],
                })
    # build 1024x4096 gt_cls_map
    gt_cls_map_8x = data['gt_cls_map'] # [B, 4096], value in (0, 4096+1)
    gt_cls_map_8x_2D = rearrange(gt_cls_map_8x, 'B (h0 w0) -> B h0 w0', h0=H0//8, w0=W0//8)
    gt_cls_map_i_16x_j_8x = gt_cls_map_8x_2D[:, ::2, ::2] # B, 1024
    data.update({
        # f'gt_mconf_map{suffix}': gt_mconf_map,
        f'gt_cls_map_i_16x_j_8x': gt_cls_map_i_16x_j_8x[:,None],
        f'gt_cls_ids': None,
        f'gt_cls_map_8x_2D': gt_cls_map_8x_2D,
    })


def create_meshgrid(h, w, normalized_coordinates=False, device='cpu'):
    """Create a meshgrid of coordinates.
    """
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack([xs, ys], dim=-1).to(device)
    if normalized_coordinates:
        grid = grid.float()
        grid[..., 0] = 2.0 * grid[..., 0] / (w - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (h - 1) - 1.0
    return grid.float()


@torch.no_grad()
@autocast(enabled=False)
def spvs_flow(data, on_reference=True):
    """Generate dense optical flow supervision.

    This function calculates the dense flow field from a source to a target image
    using camera geometry and depth maps. It handles cases without valid depth
    by falling back to a homography transformation. The resulting flow and
    validity mask are added to the input `data` dictionary.

    Args:
        data (dict): A dictionary containing image data, including:
            'image0', 'image1', 'depth0', 'depth1', 'K0', 'K1', 'T_0to1'.
            It may also contain 'homography' for fallback cases.
        on_reference (bool): If True, computes flow from image0 to image1.
            If False, computes flow from image1 to image0.

    Updates:
        data (dict): Adds the following keys:
            'flow_gt' (torch.Tensor): The calculated flow field [B, 2, H, W].
            'flow_valid_mask' (torch.Tensor): A boolean mask indicating valid
                flow vectors [B, 1, H, W].
    """
    device = data['image0'].device
    B, _, H, W = data['image0'].shape
    
    # 1. Select source and target data based on 'on_reference' flag
    if on_reference:
        img_src, img_tgt = data['image0'], data['image1']
        depth_src, depth_tgt = data['depth0'], data['depth1']
        K_src, K_tgt = data['K0'], data['K1']
        T_src_to_tgt = data['T_0to1']
        homography = data.get('homography')
        original_size_tgt = data.get('origin_img_size1')
        scale_src = data.get('scale0', 1.0)
        scale_tgt = data.get('scale1', 1.0)
    else:
        img_src, img_tgt = data['image1'], data['image0']
        depth_src, depth_tgt = data['depth1'], data['depth0']
        K_src, K_tgt = data['K1'], data['K0']
        # Invert the transformation to go from target to source
        T_src_to_tgt = torch.inverse(data['T_0to1'])
        if 'homography' in data and data['homography'] is not None:
            homography = torch.inverse(data['homography'])
        else:
            homography = None
        original_size_tgt = data.get('origin_img_size0')
        scale_src = data.get('scale1', 1.0)
        scale_tgt = data.get('scale0', 1.0)

    _, _, H_src, W_src = img_src.shape
    _, _, H_tgt, W_tgt = img_tgt.shape

    # 2. Create a dense grid of pixel coordinates for the source image
    grid_src = create_meshgrid(H_src, W_src, False, device).view(1, H_src * W_src, 2).repeat(B, 1, 1)

    # Scale grid to original image size for warping
    if isinstance(scale_src, torch.Tensor) and scale_src.dim() == 2:
        # scale_src is (B, 2), for (scale_w, scale_h)
        grid_src_orig = grid_src * scale_src.unsqueeze(1)
    else:
        # scalar multiplication
        grid_src_orig = grid_src * scale_src

    # 3. Initialize output tensors
    warped_grid_orig = torch.zeros_like(grid_src_orig)
    valid_mask = torch.zeros((B, H_src * W_src), dtype=torch.bool, device=device)

    # 4. Warp points using depth for batches with valid transformations
    # A valid transformation is assumed if the matrix is not all zeros.
    valid_T_mask = T_src_to_tgt.abs().sum(dim=(1, 2)) > 0
    if valid_T_mask.any():
        valid_indices = torch.where(valid_T_mask)[0]
        valid_mask_dpt, warped_grid_dpt = warp_grid(
            grid_src_orig[valid_indices],
            depth_src[valid_indices],
            depth_tgt[valid_indices],
            T_src_to_tgt[valid_indices],
            K_src[valid_indices],
            K_tgt[valid_indices]
        )
        warped_grid_orig[valid_indices] = warped_grid_dpt
        valid_mask[valid_indices] = valid_mask_dpt
    
    # 5. Handle fallback cases (e.g., using homography)
    invalid_T_mask = ~valid_T_mask
    if invalid_T_mask.any() and homography is not None:
        invalid_indices = torch.where(invalid_T_mask)[0]
        
        # Use clone to avoid in-place modification issues if homography is used elsewhere
        homography_fallback = homography[invalid_indices].clone()
        
        valid_mask_homo, warped_grid_homo = homo_warp_kpts_glue(
            grid_src_orig[invalid_indices],
            homography_fallback,
            original_size1=original_size_tgt[invalid_indices] if original_size_tgt is not None else None
        )
        warped_grid_orig[invalid_indices] = warped_grid_homo
        valid_mask[invalid_indices] = valid_mask_homo

    # 6. Calculate the flow vector field
    # Scale warped grid back to network-input resolution
    if isinstance(scale_tgt, torch.Tensor) and scale_tgt.dim() == 2:
        warped_grid = warped_grid_orig / scale_tgt.unsqueeze(1)
    else:
        warped_grid = warped_grid_orig / scale_tgt
    # flow = warped coordinates - original coordinates
    # flow = warped_grid - grid_src
    flow = warped_grid
    
    # 7. Reshape to image-like dimensions [B, C, H, W]
    flow_gt = rearrange(flow, 'b (h w) c -> b c h w', h=H_src, w=W_src)
    flow_valid_mask = rearrange(valid_mask, 'b (h w) -> b 1 h w', h=H_src, w=W_src)

    # 8. Update the data dictionary
    data.update({'flow_gt': flow_gt, 'flow_valid_mask': flow_valid_mask})

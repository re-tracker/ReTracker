import cv2
import torch
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def _compute_dist_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'megadepth':
        thr = 5
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

def _compute_traj_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'flyingthings':
        thr = 8
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        # fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
        #                                     (fkpts0[i, 1], fkpts1[i, 1]),
        #                                     transform=fig.transFigure, c=color[i], linewidth=1)
        #                                 for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def make_matching_figure_flow(
        img0, img1, mkpts0, mkpts1, color, mkpts0_flow=None, mkpts1_flow=None,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        # fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
        #                                     (fkpts0[i, 1], fkpts1[i, 1]),
        #                                     transform=fig.transFigure, c=color[i], linewidth=1)
        #                                 for i in range(len(mkpts0))]
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)
    
    if mkpts1_flow is not None:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts1_flow = transFigure.transform(axes[1].transData.transform(mkpts1_flow))
        if mkpts1.shape[0] != 0:
            WW = mkpts1_flow.shape[0] // mkpts1.shape[0]
            _color = np.array(color).repeat(WW, axis=0)
            axes[1].scatter(mkpts1_flow[:, 0], mkpts1_flow[:, 1], c=_color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

def make_matching_figure_withGT(
        img0, img1, mkpts0, mkpts1, gt_mkpts1, mkpts1_c, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 4, figsize=(12, 8), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    axes[2].imshow(img1, cmap='gray')
    axes[3].imshow(img1, cmap='gray')
    for i in range(4):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        # fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
        #                                     (fkpts0[i, 1], fkpts1[i, 1]),
        #                                     transform=fig.transFigure, c=color[i], linewidth=1)
        #                                 for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1_c[:, 0], mkpts1_c[:, 1], c=color, s=4)
        axes[2].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)
        axes[3].scatter(gt_mkpts1[:, 0], gt_mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_confidence_figure(data, b_id, alpha=True):
    b_mask = data['b_ids'] == b_id # check m_bids
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].clone().detach().cpu().float().numpy()
    mconf = data['mconf'][b_mask].detach().cpu().float().numpy()
    conf_thr = 0.9999
    
    # matching info
    if alpha == 'dynamic':
        # Use number of matches to scale alpha (confidence visualization has no GT mask).
        alpha = dynamic_alpha(len(kpts0))
    color = confidence_colormap(mconf, thr=conf_thr)
    
    text = [
        f'#Matches {len(kpts0)}',
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_dist_evaluation_figure(data, b_id, alpha='dynamic', plot_flow=False):
    b_mask_skeptical = data['b_ids_all'] == b_id # keep low confidence predictions as well
    b_mask = data['b_ids'] == b_id # keep low confidence predictions as well
    # caculate PCK@ 1,3,5 if data['mkpts1_err'] is existed
    pck_list={1: 0, 3: 0, 5: 0, 10: 0}
    mkpts1_errs = data['mkpts1_err'][b_mask_skeptical].cpu().float().numpy()
    for dist_thr in pck_list:
        if len(mkpts1_errs) > 0:
            pck_list[dist_thr] = (mkpts1_errs < dist_thr).sum() / len(mkpts1_errs)
        else:
            pck_list[dist_thr] = 0.0

    dist_thr = _compute_dist_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f_all'][b_mask_skeptical].cpu().numpy()
    kpts1 = data['mkpts1_f_all'][b_mask_skeptical].clone().detach().cpu().float().numpy()
    
    correct_mask = mkpts1_errs < dist_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_traj_colormap(mkpts1_errs, dist_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({dist_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R_err{data["R_errs"][b_id]:.2e}',
        f't_err{data["t_errs"][b_id]:.2e}',
        f'PCK@1: {pck_list[1]:.2f}',
        f'PCK@3: {pck_list[3]:.2f}',
        f'PCK@5: {pck_list[5]:.2f}',
        f'PCK@10: {pck_list[10]:.2f}',
        # f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    if plot_flow:
        kpts1_flow_raw = data['updated_pos_nlvl_flow'].clone().detach().cpu().float().numpy()
        valid_points = data['valids'][0,0].cpu() # B=1 F=1
        kpts1_flow = kpts1_flow_raw[valid_points][:,-1].reshape(-1,2) # BN, nlvl=3, 1, WW, 2 -> N*WW, 2
        figure = make_matching_figure_flow(img0, img1, kpts0, kpts1,
                                           color=color, mkpts0_flow=None, mkpts1_flow=kpts1_flow,
                                           text=text)
    else:
        figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                    color, text=text)
    return figure



def _make_evaluation_figure(data, b_id, alpha='dynamic', use_m_bids_f=False, plot_flow=False):
    if use_m_bids_f:
        b_mask = (data['m_bids_f'] == b_id) if 'm_bids_f' in data else (data['m_bids'] == b_id)
    else:
        b_mask = data['b_ids'] == b_id
        b_mask_skeptical = data['b_ids_all'] == b_id # we don't remove predcitons with low confidence
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].clone().detach().cpu().float().numpy()
    
    # caculate PCK@ 1,3,5 if data['mkpts1_err'] is existed
    # - keep predictions with low confidence
    pck_list={1: 0, 3: 0, 5: 0, 10: 0}
    if 'mkpts1_err' in data:
        mkpts1_errs = data['mkpts1_err'][b_mask_skeptical].cpu().float().numpy()
        for dist_thr in pck_list:
            pck_list[dist_thr] = (mkpts1_errs < dist_thr).sum() / len(mkpts1_errs)

    epi_errs = data['epi_errs'][b_mask].cpu().float().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    # n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu()) if 'conf_matrix_gt' in data else data['gt'][1]['gt_prob'].sum()
    # recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R_err{data["R_errs"][b_id]:.2e}',
        f't_err{data["t_errs"][b_id]:.2e}',
        f'PCK@1: {pck_list[1]:.2f}',
        f'PCK@3: {pck_list[3]:.2f}',
        f'PCK@5: {pck_list[5]:.2f}',
        f'PCK@10: {pck_list[10]:.2f}',
        # f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure


def _make_gt_figure(data, b_id, use_m_bids_f: bool = False):
    """Best-effort GT visualization.

    The historical codebase had several variants of "GT figure" helpers that
    depended on internal training tensors. For the refactored repo we keep this
    entry point so callers can request `mode='gt'`, but we fall back to the
    standard evaluation figure unless explicit GT tensors are available.
    """
    return _make_evaluation_figure(data, b_id, alpha=True, use_m_bids_f=use_m_bids_f)


def _make_trajs_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_traj_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1_c = data['mkpts1_c'][b_mask].cpu().float().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    gt_kpts1 = (data['trajs'][:,2]).reshape(-1,2)[b_mask].cpu().float().numpy() # target
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().float().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().float().numpy()[[1, 0]]
        kpts1_c = kpts1_c / data['scale1'][b_id].cpu().float().numpy()[[1, 0]]
        gt_kpts1 = gt_kpts1 / data['scale1'][b_id].cpu().float().numpy()[[1, 0]]

    traj_errs = data['traj_errs'][0][b_mask.cpu().float().numpy()]
    correct_mask = traj_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    if 'conf_matrix_gt_01' in data:
        n_gt_matches = int(data['conf_matrix_gt_01'][b_id].sum().cpu())
    else:
        n_gt_matches = 0
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt_01
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_traj_colormap(traj_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    # figure = make_matching_figure(img0, img1, kpts0, kpts1,
    #                               color, text=text)
    figure = make_matching_figure_withGT(img0, img1, kpts0, kpts1, gt_kpts1, kpts1_c,
                                  color, text=text)
    return figure

def _make_trajs_evaluation_frame(image, trajs, occs, color_val, cmap='Set1'):
    """
    Input: images, Tensor, [C,H,W], C=1
    """
    color_map = cm.get_cmap(cmap)
    np_img = (image.cpu().numpy() * 255).round().astype(np.uint8)
    rgb_np_img = cv2.cvtColor(np_img[0], cv2.COLOR_GRAY2RGB)
    kpts = trajs.cpu().float().numpy()
    occs = occs.cpu().float().numpy()
    # kpts = kpts / scale.cpu().numpy()[[1, 0]]
    # clip gt kpts to avoid out of image
    bound = 100
    kpts[:,0] = np.clip(kpts[:,0], 0 - bound, rgb_np_img.shape[1]-1 + bound)
    kpts[:,1] = np.clip(kpts[:,1], 0 - bound, rgb_np_img.shape[0]-1 + bound)
    S1, D = trajs.shape
    for idx, (kpt, occ) in enumerate(zip(kpts, occs)):
        _color =  np.array(color_map(color_val[idx])[:3]) * 255
        cv2.circle(rgb_np_img, (int(kpt[0]), int(kpt[1])), 2, _color, (-1 if occ else 0), cv2.FILLED)
    return rgb_np_img


def make_matching_figures(data, config, mode='evaluation', plot_flow=False):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_ReTracker.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt', 'dist']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=True,
                use_m_bids_f=False,
                plot_flow=plot_flow)
        elif mode == 'confidence':
            fig = _make_confidence_figure(
                data, b_id,
                alpha=True,
                )
        elif mode == 'gt':
            try:
                fig = _make_gt_figure(data, b_id, use_m_bids_f=False)
            except Exception:
                fig = _make_evaluation_figure(
                    data, b_id,
                    alpha=True,
                    use_m_bids_f=False)
        elif mode == 'dist':
            # try:
                fig = _make_dist_evaluation_figure(
                    data, b_id,
                    alpha=True,
                    plot_flow=plot_flow,
                )
            # except:
            #     pass
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def make_scores_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_ReTracker.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            # hist1 = torch.histc(data['conf_matrix'][b_id].reshape(-1), bins=20, min=0, max=1) # softmax without mn and thr
            # mconf = data['mconf'][data['m_bids_f']==b_id]
            # mconf = mconf[mconf>0.01].reshape(-1)
            # hist2 = torch.histc(mconf, bins=20, min=0, max=1) # softmax with mn and thr
            # del mconf
            if config.LOFTR.MATCH_COARSE.SKIP_SOFTMAX and config.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES:
                plots = [data['histc_skipmn_in_softmax'][b_id].reshape(-1)] # [-30, 70] scores
                if 'histc_skipmn_in_softmax_gt' in data:
                    plots.append(data['histc_skipmn_in_softmax_gt'][b_id].reshape(-1))
            elif config.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES:
                pass
                # plots = [data['histc_skipmn_in_softmax'][b_id].reshape(-1), hist1, hist2]
            else:
                pass
                # plots = [hist1, hist2]
            group = len(plots)
            start, end = 0, 100
            bins=100
            width = (end//bins-1)/group
            fig, ax = plt.subplots()
            for i, hist in enumerate(plots):
                # hist = torch.histc(z.reshape(-1), bins=bins, min=0, max=1)
                ax.set_yscale('log')
                x = range(start, end, end//bins)
                x = [t + i*width for t in x]
                ax.bar(x, hist.cpu(), align='edge', width=width)
                # plt.savefig('./vis/gaussian_b1_1e4x1e4_skipmn_divtop2cmp_in_dualxcross_softmax_t1e1.png')
            
        elif mode == 'confidence':
            raise NotImplementedError()
        elif mode == 'gt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def make_tracking_videos(data, config, mode='evaluation'):
    """Make tracking figures from a batch;
    v1. Mark all trajs in green 
    Args:
        data(Dict) {
            'images': [B, F, C, H, W],
            'trajs':, [B, F, N, 2]
            'pred_trajs':, [B, F, N, 2]
            'pred_occs':, [B, F, N, 1]
            'valids':, [B, F, N]
        }
        
    """
    videos = {mode:[]}
    for b_id in range(data['images'].size(0)):
        if mode == 'evaluation':
            video = []
            color_val = np.random.rand(data['trajs'].shape[2])
            for f_id in range(data['images'].size(1)):
                _fig_pred = _make_trajs_evaluation_frame(
                    data['images'][b_id,f_id],
                    data['pred_trajs'][b_id,f_id],
                    ~data['pred_occs'][b_id,f_id], # 估计的conf
                    color_val,
                )
                _fig_gt = _make_trajs_evaluation_frame(
                    data['images'][b_id,f_id],
                    data['trajs'][b_id,f_id],
                    data['valids'][b_id,f_id], # GT conf
                    color_val,
                )
                video.append(np.concatenate([_fig_pred, _fig_gt], axis=-2))
            video = np.stack(video) # F H W 3
            videos[mode].append(video) #
    
    videos[mode] = np.stack(videos[mode]) # B F H W 3
    # B F C H W
    return videos


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

def confidence_colormap(conf, thr, alpha=1.0):
    # return green if confidence is high, red if confidence is low
    # x = 1 - np.clip(conf / (thr * 2), 0, 1)
    x = np.where(conf < thr, 0, 1)
    return np.stack([1-x, x, np.zeros_like(conf), np.ones_like(conf)], -1)

def error_traj_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

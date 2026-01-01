import pprint
import torch
import cv2
import numpy as np
from collections import OrderedDict
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous

from retracker.utils.rich_utils import CONSOLE

# --- METRICS ---
def compute_traj_errors_video(data, config):
    """ 
        data (dict):{
            "trajs" (Tensor): [..., 2]
            "traj_errs" (Tensor): [..., 2]
            "occ_errs" (Tensor): [...] 
        }
    Update:
        data (dict):{
            "traj_errs" (Tensor): scalar
            "occ_errs" (Tensor): [...]
        }
    """
    data.update({'traj_errs': [], 'occ_errs': []})
    traj_errs, occ_errs = [], []

    gt_trajs = data['trajs'].cpu().numpy()
    pred_trajs = data['pred_trajs'].detach().cpu().numpy()
    
    B = gt_trajs.shape[0]
    traj_errs = np.linalg.norm(
        gt_trajs - pred_trajs ,axis=-1).reshape(B, -1)
    occ_errs = np.zeros(B)    
    data.update({
        'traj_errs':traj_errs.mean(axis=-1),
        'traj_errs_raw':traj_errs,
        'occ_errs':occ_errs,
    })


def traj_error(pred_pos, target_pos):
    # traj_err = (pred_pos - target_pos).mean()
    traj_err = np.sqrt(((pred_pos - target_pos)**2).sum(-1))
    return traj_err, traj_err

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data, config):
    """ 
    Update: data (dict):{"epi_errs": [M]}
    """
    # [修复开始]：处理 T_0to1 可能存在的额外维度 [B, 1, 4, 4] -> [B, 4, 4]
    T_0to1 = data['T_0to1']
    if T_0to1.dim() == 4 and T_0to1.shape[1] == 1:
        T_0to1 = T_0to1.squeeze(1)
    
    # 同理处理 K0, K1，防止形状为 [B, 1, 3, 3] 导致后续计算出错
    K0 = data['K0']
    if K0.dim() == 4 and K0.shape[1] == 1:
        K0 = K0.squeeze(1)
        
    K1 = data['K1']
    if K1.dim() == 4 and K1.shape[1] == 1:
        K1 = K1.squeeze(1)
    # [修复结束]

    B = T_0to1.shape[0]
    
    # 现在 T_0to1 是 [B, 4, 4]，切片 [:, :3, 3] 会得到 [B, 3]，符合预期
    Tx = numeric.cross_product_matrix(T_0to1[:, :3, 3])
    E_mat = Tx @ T_0to1[:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    scale0 = data['scale0']
    scale1 = data['scale1']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        _scale0 = scale0[bs] if scale0.dim() == 2 else scale0[bs, 0]
        _scale1 = scale1[bs] if scale1.dim() == 2 else scale1[bs, 0]
        
        _pts0, _pts1 = pts0[mask], pts1[mask]
        
        if len(_pts0) == 0: # 防止空数据报错
            continue
            
        _unscaled_pts0 = _pts0 * _scale0[None]
        _unscaled_pts1 = _pts1 * _scale1[None]
        
        epi_errs.append(
            symmetric_epipolar_distance(_unscaled_pts0, _unscaled_pts1, E_mat[bs], K0[bs], K1[bs]))
            
    if len(epi_errs) > 0:
        epi_errs = torch.cat(epi_errs, dim=0)
    else:
        epi_errs = torch.tensor([], device=pts0.device)

    data.update({'epi_errs': epi_errs})


def _compute_symmetrical_epipolar_errors(data, config):
    """ 
    Input:
        data (dict):{
            "m_bids" List[int]: [N]
            "mkpts0_f" List[float]: [N, 2]
            "mkpts1_f" List[float]: [N, 2]
            "scale0" List[float]: [M, 2]
            "scale1" List[float]: [M, 2]
            "K0" List[float]: [M, 3, 3]
            "K1" List[float]: [M, 3, 3]
            "T_0to1" List[float]: [M, 4, 4]
        }
    Update:
        data (dict):{"epi_errs": [M]}
    """
    B = data['T_0to1'].shape[0]
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    scale0 = data['scale0']
    scale1 = data['scale1']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        _scale0, _scale1 = scale0[bs], scale1[bs]
        _pts0, _pts1 = pts0[mask], pts1[mask]
        _unscaled_pts0 = _pts0 * _scale0[None]
        _unscaled_pts1 = _pts1 * _scale1[None]
        epi_errs.append(
            symmetric_epipolar_distance(_unscaled_pts0, _unscaled_pts1, E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def compute_pose_errors(data, config):
    """ 
    Input:
        data (dict):{
            "m_bids" List[int]: [N]
            "mkpts0_f" List[float]: [N, 2]
            "mkpts1_f" List[float]: [N, 2]
            "scale0" List[float]: [M, 2]
            "scale1" List[float]: [M, 2]
            "K0" List[float]: [M, 3, 3]
            "K1" List[float]: [M, 3, 3]
            "T_0to1" List[float]: [M, 4, 4]
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = 0.5 # config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = 0.99999 # config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})
    
    m_bids = data['m_bids'].cpu().numpy()
    pts0 = (data['mkpts0_f']).cpu().float().numpy()
    pts1 = (data['mkpts1_f']).cpu().float().numpy()
    scale0 = data['scale0'].cpu().numpy()
    scale1 = data['scale1'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        _scale0, _scale1 = scale0[bs], scale1[bs]
        _pts0, _pts1 = pts0[mask], pts1[mask]
        _unscaled_pts0 = _pts0 * _scale0[None]
        _unscaled_pts1 = _pts1 * _scale1[None]
        
        ret = estimate_pose(_unscaled_pts0, _unscaled_pts1, K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        CONSOLE.print("[yellow]E is None while trying to recover pose.[/yellow]")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def trajs_pck(dist, thresholds):
    """
    dist: 1D array
    thresholds: List
    """
    pcks = []
    dist.sort()
    for th in thresholds:
        idx = np.searchsorted(dist, th, side='right')
        pcks.append(idx/dist.shape[0])
    
    return {f'PCK@{t}': pck for t, pck in zip(thresholds, pcks)}

def aggregate_metrics(metrics):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    CONSOLE.print(f"[cyan]Aggregating metrics over {len(unq_ids)} unique items...[/cyan]")

    # traj acc
    pck_thresholds = [2, 4, 8, 16]
    dist = np.array(metrics['traj_errs_raw']).reshape(-1) # convert to 1D array
    pcks = trajs_pck(dist, pck_thresholds)
    return {**pcks}

    # # pose auc
    # angular_thresholds = [5, 10, 20]
    # pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    # aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # # matching precision
    # dist_thresholds = [epi_err_thr]
    # precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    # return {**aucs, **precs}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def error_auc(errors, thresholds, method='exact_auc'):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    if method == 'exact_auc':
        errors = [0] + sorted(list(errors))
        recall = list(np.linspace(0, 1, len(errors)))

        aucs = []
        for thr in thresholds:
            last_index = np.searchsorted(errors, thr)
            y = recall[:last_index] + [recall[last_index-1]]
            x = errors[:last_index] + [thr]
            aucs.append(np.trapz(y, x) / thr)

    elif method == 'fire_paper':
        aucs = []
        for threshold in thresholds:
            accum_error = 0
            percent_error_below = np.zeros(threshold + 1)
            for i in range(1, threshold + 1):
                percent_error_below[i] = np.sum(errors < i) * 100 / len(errors)
                accum_error += percent_error_below[i]
            
            aucs.append(accum_error / (threshold * 100))
    else:
        raise NotImplementedError

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def aggregate_matching_metrics(metrics, epi_err_thr=5e-4, eval_n_time=1, threshold=[5, 10, 20], auc_method='exact_auc'):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    CONSOLE.print(f"[cyan]Aggregating metrics over {len(unq_ids)} unique items...[/cyan]")

    # pose auc
    angular_thresholds = threshold
    if eval_n_time >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, eval_n_time)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    # pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)
    CONSOLE.print(f"[cyan]num of pose_errors: {pose_errors.shape}[/cyan]")
    aucs = error_auc(pose_errors, angular_thresholds, method=auc_method)  # (auc@5, auc@10, auc@20)

    if eval_n_time >= 1:
        for i in range(eval_n_time):
            aucs_i = error_auc(pose_errors.reshape(-1, eval_n_time)[:,i], angular_thresholds, method=auc_method)
            CONSOLE.print(f"[cyan]results of {i}-th RANSAC[/cyan]\n{pprint.pformat(aucs_i)}")
    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
    
    u_num_mathces = np.array(metrics['num_matches'], dtype=object)[unq_ids]
    u_percent_inliers = np.array(metrics['percent_inliers'], dtype=object)[unq_ids]
    num_matches = {f'num_matches': u_num_mathces.mean() }
    percent_inliers = {f'percent_inliers': u_percent_inliers.mean()}
    return {**aucs, **precs, **num_matches, **percent_inliers}

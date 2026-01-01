from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Optional
import torch
from tqdm import tqdm
import numpy as np
import json

from retracker.data.datasets.utils import dataclass_to_cuda_
from retracker.visualization.visualizer import Visualizer
from retracker.evaluation.model_utils import reduce_masked_mean
from retracker.evaluation.eval_utils import compute_tapvid_metrics
from retracker.io.results import dump_results
from retracker.utils.rich_utils import CONSOLE

import logging

class PublicEvaluator:
    """
    Generalized evaluator for TrackingEngine on EvaluationDataset
    Output logs and .
    """

    def __init__(
        self,
        exp_dir: str,
        *,
        load_dump: bool = True,
        save_dump: bool = False,
        skip_if_done: bool = False,
        save_video: bool = True,
    ) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.load_dump = bool(load_dump)
        self.save_dump = bool(save_dump)
        self.skip_if_done = bool(skip_if_done)
        self.save_video = bool(save_video)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualizations")
        self.dumps_dir = os.path.join(exp_dir, "dumps")
        self.metrics_dir = os.path.join(exp_dir, "metrics_per_seq")
        self.done_dir = os.path.join(exp_dir, "done")
        os.makedirs(self.dumps_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.done_dir, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        # Make a filesystem-friendly name (keep it stable for resume).
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        if "tapvid" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.9

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1)).cpu().numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

            # TAP-Vid paper metrics assume tracks are in 256x256 raster coordinates.
            # When evaluation runs at a different input resolution (e.g. 512/768/1024),
            # rescale both GT and predictions back to the 256 coordinate system so that
            # the fixed pixel thresholds [1, 2, 4, 8, 16] remain comparable.
            if hasattr(sample, "video") and sample.video is not None:
                _, _, _, H, W = sample.video.shape
                if H > 1 and W > 1 and (H != 256 or W != 256):
                    sx = 255.0 / float(W - 1)
                    sy = 255.0 / float(H - 1)
                    query_points = query_points.copy()
                    query_points[..., 1] *= sy  # y
                    query_points[..., 2] *= sx  # x
                    gt_tracks = gt_tracks.copy()
                    gt_tracks[..., 0] *= sx  # x
                    gt_tracks[..., 1] *= sy  # y
                    pred_tracks = pred_tracks.copy()
                    pred_tracks[..., 0] *= sx  # x
                    pred_tracks[..., 1] *= sy  # y

            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )
            # `compute_tapvid_metrics` returns arrays (shape [B]); for eval we use B=1.
            out_metrics = {
                k: float(v[0]) if isinstance(v, np.ndarray) and v.size == 1 else float(np.mean(v))
                for k, v in out_metrics.items()
            }

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            CONSOLE.print(f"[dim]metrics[/dim] {out_metrics}")
            CONSOLE.print(f"[dim]avg[/dim] {metrics['avg']}")
        elif dataset_name == "dynamic_replica" or dataset_name == "pointodyssey":
            *_, N, _ = sample.trajectory.shape
            B, T, N = sample.visibility.shape
            H, W = sample.video.shape[-2:]
            device = sample.video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(d_, (1 - sample.visibility) * start_tracking_mask).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(d_, sample.visibility * start_tracking_mask).item() * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = reduce_masked_mean(d_, start_tracking_mask).item() * 100.0
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 50
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * sample.visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            CONSOLE.print(f"[dim]metrics[/dim] {out_metrics}")
            CONSOLE.print(f"[dim]avg[/dim] {metrics['avg']}")

    @torch.no_grad()
    def evaluate(
        self,
        engine,
        dataset_name: str,
        dataloader,
        visualize_every: int = 1,

        writer: Any | None = None,
        step: Optional[int] = 0,
        ):
        '''
        engine: TrackingEngine for specific method;
        dataloader: dataloader for specific dataset EvaluationDataset;

        '''
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=30,
            tracks_leave_trace=1,
        )
        
        for ind, sample in enumerate(tqdm(dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    CONSOLE.print("[yellow]batch is None[/yellow]")
                    continue

            # Prefer a stable per-video identifier to support resume and multi-process runs.
            if hasattr(sample, "seq_name") and sample.seq_name is not None:
                seq_name = sample.seq_name[0] if isinstance(sample.seq_name, list) else sample.seq_name
            else:
                seq_name = str(ind)
            seq_name = self._safe_name(seq_name)
            engine_name = getattr(engine, "display_name", engine.__class__.__name__)
            artifact_id = self._safe_name(f"{dataset_name}_{engine_name}_{seq_name}")

            done_path = os.path.join(self.done_dir, f"{artifact_id}.done")
            if self.skip_if_done and os.path.exists(done_path):
                continue

            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                ).to(device)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)

            # dump results;
            dump_npz_path = os.path.join(self.dumps_dir, f"{artifact_id}.npz")

            # check_exist
            if self.load_dump and os.path.exists(dump_npz_path):
                pred_tracks = self._load_pred_npz(dump_npz_path, device=device)
            else:
                pred_tracks = engine.video_forward(sample.video, queries)
                if self.save_dump:
                    self._dump_pred_npz(pred_tracks, dump_npz_path)
            

            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = engine(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if ind % visualize_every == 0:
                vis.visualize(
                    sample.video,
                    pred_tracks[0] if isinstance(pred_tracks, tuple) else pred_tracks,
                    gt_tracks=sample.trajectory,
                    visibility=pred_tracks[1],
                    filename=artifact_id,
                    writer=writer,
                    step=step,
                    save_video=self.save_video,
                )
            # dump_results(self.exp_dir, dataset_name + "_" + seq_name, pred_tracks[0], pred_tracks[1])

            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
            # Persist per-sequence metrics for aggregation and resume.
            try:
                seq_metrics = metrics[sample.seq_name[0]]
                with open(os.path.join(self.metrics_dir, f"{artifact_id}.json"), "w") as f:
                    json.dump(seq_metrics, f, indent=2)
            except Exception:
                # Metrics are best-effort; don't fail the whole evaluation because of IO.
                pass

            # Mark sequence as done for resume.
            try:
                with open(done_path, "w") as f:
                    f.write("ok\n")
            except Exception:
                pass
        return metrics

    def _dump_pred_npz(self, pred_tracks, dump_npz_path):
        pred_dict = {'pred_tracks':pred_tracks[0].cpu().numpy(), 'pred_occs': pred_tracks[1].cpu().numpy()}
        np.savez_compressed(dump_npz_path, **pred_dict)
    
    def _load_pred_npz(self, dump_npz_path, device):
        pred_dict = np.load(dump_npz_path, allow_pickle=True)
        # convert to torch tensor
        pred_tracks = torch.from_numpy(pred_dict['pred_tracks']).to(device)
        pred_occs = torch.from_numpy(pred_dict['pred_occs']).to(device)
        return (pred_tracks, pred_occs)

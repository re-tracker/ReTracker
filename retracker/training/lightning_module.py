from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Literal

import torch
from omegaconf import OmegaConf

from retracker.training.utils.logging import configure_logger
from retracker.training.utils.videodata import construct_pairs
from retracker.utils.checkpoint import safe_torch_load
from retracker.utils.rich_utils import CONSOLE

try:
    import lightning.pytorch as pl  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    pl = None  # type: ignore[assignment]


class _LightningModuleFallback:
    """Minimal fallback for unit tests when Lightning is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.trainer = None
        self.logger = None
        self.loggers = []
        self.global_step = 0
        self.current_epoch = 0

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None


LightningModuleBase = pl.LightningModule if pl is not None else _LightningModuleFallback


@contextmanager
def _passthrough_profile(_name: str):
    yield


def _build_default_profiler():
    """Best-effort profiler to keep training module importable without Lightning."""
    try:
        from retracker.training.profiler import PassThroughProfiler  # type: ignore[import-not-found]

        return PassThroughProfiler()
    except ModuleNotFoundError:
        return type("FallbackProfiler", (), {"profile": staticmethod(_passthrough_profile)})()
    except Exception:
        if pl is not None:
            raise
        return type("FallbackProfiler", (), {"profile": staticmethod(_passthrough_profile)})()


def class_parser(class_and_args, Class):
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=Class)
    parser.add_argument("--data", type=dict)  # to ignore data
    config = class_and_args["init_args"]["config"]
    config_init = parser.instantiate_classes(config)
    return config_init.model


class PL_ReTracker(LightningModuleBase):
    def __init__(
        self, config, profiler, automatic_optimization=True, pretrained_ckpt=None, dump_dir=None
    ):
        super().__init__()
        self.automatic_optimization = automatic_optimization
        self.profiler = profiler or _build_default_profiler()
        self.config = OmegaConf.create(config)
        self.dump_dir = dump_dir
        # task_mode allows overriding runtime task (tracking vs matching)
        self.task_mode = None

        self._initialize_matcher_and_loss()
        self._initialize_flags_and_logs()
        self.debug_mode_init()
        self._load_pretrained_ckpt(pretrained_ckpt)
        self._configure_logging(dump_dir)

    def _initialize_matcher_and_loss(self):
        """Initialize the matcher and loss function based on the config."""
        from retracker.models import ReTracker
        from retracker.training.losses.retracker_loss import ReTrackerLoss

        self.matcher = ReTracker(config=self.config.retracker_config)
        self.loss = ReTrackerLoss(config=self.config.loss_config, model=self.matcher)
        self.matcher.model_task_type = self.config.model_task_type
        CONSOLE.print(f"{self.config.model_task_type}")

    def _load_pretrained_ckpt(self, pretrained_ckpt):
        """Load pretrained checkpoint if provided."""
        if pretrained_ckpt == None:
            return
        ckpt = safe_torch_load(pretrained_ckpt, map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.matcher.load_state_dict(state_dict, strict=False)
        # print which keys are not loaded
        for key in state_dict.keys():
            if key not in self.matcher.state_dict().keys():
                CONSOLE.print(f"[yellow]Key {key} not loaded")
        CONSOLE.print(f"Load '{pretrained_ckpt}' as pretrained checkpoint.")

    def _configure_logging(self, dump_dir):
        """Configure logging and create dump directory if specified."""
        if dump_dir is not None:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            configure_logger(dump_dir)
            CONSOLE.print(f"[dim]Dumped to: {dump_dir}[/dim]")

    def _initialize_flags_and_logs(self):
        """Initialize flags and logging containers."""
        self.abort_current_branch_flag = False
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_sequence_outputs = []
        self.frame_logs = {}
        self.video_logs = {}
        self.video_data_cache = []

    def configure_optimizers(self):
        from retracker.training.optim import build_optimizer, build_scheduler

        optimizer = build_optimizer(self, self.config.optimizer_config)
        scheduler = build_scheduler(self.config.optimizer_config, optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Debug: Comprehensive data logging for distributed training verification
        if self.debug_mode and batch_idx < self.debug_batches:
            self._log_debug_info(batch, batch_idx, mode="train")

        loss_dict = {}
        loss_dict = self._unified_training_step(batch, batch_idx)
        return loss_dict

    def on_train_epoch_start(self):
        """
        在每个 Epoch 开始时，强制更新 Dataloader 中 Sampler 的 epoch。
        """
        current_epoch = self.current_epoch

        loaders = self.trainer.train_dataloader

        def _set_epoch(loader):
            # 优先处理 batch_sampler (TaskSynchronizedBatchSampler)
            if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(current_epoch)
                # print(f"[ReTracker] Set batch_sampler epoch to {current_epoch}")
            # 其次处理 sampler (RandomConcatSampler)
            elif hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(current_epoch)

        if loaders is None:
            return

        # 情况 A: CombinedLoader (PL 2.0+ 常见)
        if hasattr(loaders, "loaders"):
            loaders = loaders.loaders  # 可能是 dict 或 list

        # 情况 B: List of Loaders
        if isinstance(loaders, list):
            for loader in loaders:
                _set_epoch(loader)

        # 情况 C: Dict of Loaders
        elif isinstance(loaders, dict):
            for loader in loaders.values():
                _set_epoch(loader)

        # 情况 D: 单个 Loader
        else:
            _set_epoch(loaders)

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) == 0:
            return

        # Debug: Detailed analysis of training_step_outputs
        loss_tensor = torch.stack(self.training_step_outputs)
        avg_loss = loss_tensor.mean()
        max_loss = loss_tensor.max()
        min_loss = loss_tensor.min()
        std_loss = loss_tensor.std()

        # Check for problematic values
        has_nan = torch.isnan(loss_tensor).any()
        has_inf = torch.isinf(loss_tensor).any()
        large_values = (loss_tensor > 1000).sum()

        CONSOLE.print(f"\n[bold][Loss Analysis][/bold] @ Epoch {self.current_epoch:03}:")
        CONSOLE.print(f"  avg_loss={avg_loss:.6f}")
        CONSOLE.print(f"  max_loss={max_loss:.6f}")
        CONSOLE.print(f"  min_loss={min_loss:.6f}")
        CONSOLE.print(f"  std_loss={std_loss:.6f}")
        CONSOLE.print(f"  num_steps={len(self.training_step_outputs)}")
        CONSOLE.print(f"  has_nan={has_nan}, has_inf={has_inf}")
        CONSOLE.print(f"  large_values(>1000)={large_values}")

        # Show some sample values
        if len(loss_tensor) > 0:
            sample_values = loss_tensor[: min(10, len(loss_tensor))].tolist()
            CONSOLE.print(f"  sample_values={[f'{v:.3f}' for v in sample_values]}")

        # Check if this epoch has abnormal values
        if avg_loss > 100 or has_nan or has_inf:
            CONSOLE.print(
                f"[yellow]WARNING: abnormal epoch detected (epoch={self.current_epoch})[/yellow]"
            )
            CONSOLE.print(f"   All loss values: {loss_tensor.tolist()}")
            CONSOLE.print(f"   Rank: {self.trainer.global_rank}")

        self.log("training_epoch_average", avg_loss)
        self.log("training_epoch_max_loss", max_loss)
        self.log("training_epoch_min_loss", min_loss)
        self.log("training_epoch_std_loss", std_loss)
        self.log("training_epoch_has_nan", float(has_nan))
        self.log("training_epoch_has_inf", float(has_inf))
        self.log("training_epoch_large_values", float(large_values))

        self.training_step_outputs.clear()  # free memory

        if self.trainer.global_rank == 0 and self.loggers:
            exp = getattr(self.loggers[0], "experiment", None)
            if exp is not None and hasattr(exp, "add_scalar"):
                exp.add_scalar(  # type: ignore[attr-defined]
                    "Train/avg_loss_on_epoch",
                    avg_loss,
                    global_step=self.current_epoch,
                )

    def validation_step(self, batch, batch_idx):
        # Debug: Log validation data for verification
        if self.debug_mode and batch_idx < self.debug_batches:
            self._log_debug_info(batch, batch_idx, mode="val")

        ret = self._unified_validation_step(batch, batch_idx)
        return ret

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    ############################################
    ######### IMPLEMENTED PIPELINE #############
    ############################################
    def _unified_validation_step(self, batch, batch_idx):
        task_mode = self._infer_task_mode_from_batch(batch)
        if hasattr(self, "matcher"):
            self.matcher.set_task_mode(task_mode)

        if task_mode == "matching":
            loss_dict = self._trainval_image_matching(batch, type="validation")
            if loss_dict is None or batch is None:
                self.zero_grad()
                return None
            return {"loss": batch["loss_scalars"]["loss"]}

        # Tracking / video path (legacy).
        loss_dict = self._causal_trainval_video(batch, type="validation")
        if batch is None:
            self.zero_grad()
            return None
        ret = {"loss": batch["loss_scalars"]["loss"]}

        prepared_data = self._prepare_data_for_viz_and_eval(batch)
        plot_interval = self.config.viz_config.matches_plot_interval

        if self.config.viz_config.enable_plotting and batch_idx % plot_interval == 0:
            metric_dict, _ = self._compute_metrics(prepared_data)
            prepared_data.update(metric_dict["metrics"])
            self._log_visualization(prepared_data, mode="Validation")

        with self.profiler.profile("Compute metrics"):
            compute_traj_errors_video(prepared_data, self.config)

        metrics = {
            "traj_errs": prepared_data["traj_errs"],
            "traj_errs_raw": prepared_data["traj_errs_raw"],
            "occ_errs": prepared_data["occ_errs"],
            "identifiers": prepared_data["identifiers"],
        }
        self.validation_step_outputs.append(metrics)
        return ret

    def _unified_training_step(self, batch, batch_idx):
        rank = self.trainer.global_rank
        task_mode = self._infer_task_mode_from_batch(batch)

        if "images" in batch:
            seq_len = batch["images"].shape[1]
            task_label = "MATCHING" if task_mode == "matching" else f"TRACKING(T={seq_len})"
            if hasattr(self, "matcher"):
                self.matcher.set_task_mode(task_mode)
            CONSOLE.print(f"[dim]>>> [Rank {rank}] Step {batch_idx}: {task_label}[/dim]")

        if task_mode == "matching":
            loss_dict = self._trainval_image_matching(batch, type="training")
        else:
            loss_dict = self._causal_trainval_video(batch, type="training")
        if loss_dict is None:
            self.zero_grad()
            return None
        ret = {"loss": batch["loss_scalars"]["loss"]} | (
            {"metrics": batch["metrics"]} if "metrics" in batch.keys() else {"metrics": {}}
        )

        for key, value in ret["metrics"].items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False)
        for key, value in batch["loss_scalars"].items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)

        if self.config.viz_config.enable_plotting:
            plot_interval = self.config.viz_config.matches_plot_interval
            if batch_idx % plot_interval == 0:
                if (
                    self.matcher.model_task_type == "causal_video_matching"
                    and batch_idx % plot_interval == 0
                ):
                    prepared_data = self._prepare_data_for_viz_and_eval(batch)
                    metric_dict, _ = self._compute_metrics(prepared_data, mode="Train")
                    prepared_data.update(metric_dict["metrics"])
                    self._log_visualization(prepared_data, mode="Train")
                    # self._log_video_for_causal_tracking(batch, ret, self.global_step, 'Train')
        return loss_dict

    @staticmethod
    def _infer_task_mode_from_batch(batch: Dict[str, Any]) -> Literal["matching", "tracking"]:
        """Infer runtime task mode from the batch structure.

        UnifiedDataModule sets `task_type` to a list of strings ('matching'/'tracking').
        As a fallback, we also infer from the frame dimension of `images`.
        """
        if batch is None:
            return "tracking"

        task_type = batch.get("task_type", None)
        if isinstance(task_type, (list, tuple)) and len(task_type) > 0:
            if str(task_type[0]).lower() == "matching":
                return "matching"
            return "tracking"

        if "images" in batch and hasattr(batch["images"], "shape"):
            # For MegaDepth/ScanNet matching datasets, images are stacked as T=2.
            if int(batch["images"].shape[1]) == 2:
                return "matching"

        return "tracking"

    def _trainval_image_matching(self, batch: Dict[str, Any], type: str):
        """Train/val a single image-pair matching batch (e.g. MegaDepth).

        Matching datasets do not come with tracking-style `occs/trajs/valids`
        tensors by default, so we:
        1) build supervision via depth/pose warping (compute_supervision_coarse),
        2) run the model on the pair,
        3) compute loss with ReTrackerLoss.forward.
        """
        self.video_data = batch  # used by optimizer_closure (kept for compatibility)
        self.matcher.mem_manager.reset_all_memory()
        self.abort_current_branch_flag = False
        self.video_data_cache = []

        # Some codepaths expect image0/image1 even if `images` exists.
        if "image0" not in batch and "images" in batch:
            batch["image0"] = batch["images"][:, 0]
            batch["image1"] = batch["images"][:, 1]

        # Build supervision for matching datasets (MegaDepth, ScanNet).
        from retracker.training.supervision import (
            add_items_for_matching_task,
            compute_supervision_coarse,
        )

        compute_supervision_coarse(batch, self.config)
        add_items_for_matching_task(batch)

        # Forward the pair.
        mode = "train" if type == "training" else "eval"
        ret_state = self.matcher(batch, mode=mode)
        if ret_state:
            return None

        # Compute matching loss (populates batch['loss'] and batch['loss_scalars']).
        self.loss(batch, batch)

        # For visualization/metrics helpers, mirror the keys used by the video path.
        if "mkpts1_f" in batch and "pred_visibles" in batch:
            batch.update(
                {
                    "pred_trajs": batch["mkpts1_f"],  # (B*N), F=1, 2
                    "pred_occs": ~batch["pred_visibles"],  # (B*N), F=1, 1
                }
            )

        # Track loss stats for epoch logging (keep behavior similar to video path).
        self.batch_loss = batch["loss_scalars"]["loss"]
        loss_value = batch["loss_scalars"]["loss"]
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            CONSOLE.print(
                "[yellow]WARNING: NaN/Inf loss detected in matching batch; skipping value.[/yellow]"
            )
            CONSOLE.print(f"   Loss value: {loss_value}")
            CONSOLE.print(f"   Epoch: {self.current_epoch}, Rank: {self.trainer.global_rank}")
            return {"loss": batch["loss"]}

        self.training_step_outputs.append(loss_value)
        return {"loss": batch["loss"]}

    def _causal_trainval_video(self, batch, type):
        """Train or val a pair and dump useful information to cache;
            - apply some strategies and flags;
            - prepare data and supervisions;
            - forward;
            - optimize (if training);
            - cache data(remove later)
        Input:
        Outputs:
        """
        self.video_data = batch  # used in optimizer_closure
        # self.matcher.current_epoch = self.current_epoch
        self.matcher.mem_manager.reset_all_memory()
        self.abort_current_branch_flag = False
        self.video_data_cache = []

        _, loss_dict = self._causal_video_forward_and_compute_loss(self.video_data, self.video_logs)

        batch.update(
            {
                # trajs in frame0 are GT
                "pred_trajs": self.video_logs["video_trajs"],  # BXN, F, 2
                "pred_occs": self.video_logs["video_occs"],  # BXN, F, 1
            }
        )

        self.batch_loss = batch["loss_scalars"]["loss"]

        # Debug: Check for abnormal loss values before appending
        loss_value = batch["loss_scalars"]["loss"]

        # Check for NaN or infinite values
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            CONSOLE.print("[yellow]WARNING: NaN/Inf loss detected; skipping this value.[/yellow]")
            CONSOLE.print(f"   Loss value: {loss_value}")
            CONSOLE.print(f"   Epoch: {self.current_epoch}, Rank: {self.trainer.global_rank}")
            return loss_dict

        # Check for abnormally large values
        if loss_value > 1000:
            CONSOLE.print(f"[yellow]WARNING: large loss detected: {loss_value:.6f}[/yellow]")
            CONSOLE.print(f"   Epoch: {self.current_epoch}, Rank: {self.trainer.global_rank}")

        self.training_step_outputs.append(loss_value)
        return loss_dict

    def _causal_video_forward_and_compute_loss(self, data, video_logs):
        """train/inference a video, and log losses to video_logs
        len(sliding_res) == 1
        """
        video_res_list = []

        with self.profiler.profile("2. ReTracker forwarding"):
            triplet_N = construct_pairs(data)
            for _idx, triplet in enumerate(triplet_N):
                # with self.profiler.profile("Compute coarse supervision"):
                #     compute_supervision_coarse(triplet, self.config)
                ret_state = self.matcher(triplet)
                if ret_state:  # exception
                    return ret_state, None

                video_res_list.append(
                    {
                        "mkpts1_f": triplet["mkpts1_f"],  # BN F 2
                        "pred_visibles": triplet["pred_visibles"],  # BN F 1
                        "fine_occlusion_logits": triplet["fine_occlusion_logits"],
                        "fine_certainty_logits": triplet["fine_certainty_logits"],
                        "updated_pos_nlvl": triplet["updated_pos_nlvl"],  # BN F nlvl 2
                        "updated_occ_nlvl": triplet["updated_occ_nlvl"],  # BN F nlvl 2
                        "updated_exp_nlvl": triplet["updated_exp_nlvl"],  # BN F nlvl 2
                        "is_masked_list": triplet.get("is_masked_list", [False, False, False]),
                        "updated_pos_nlvl_flow": triplet["updated_pos_nlvl_flow"],  # BN F nlvl 2
                        "updated_occ_nlvl_flow": triplet["updated_occ_nlvl_flow"],  # BN F nlvl 2
                        "updated_exp_nlvl_flow": triplet["updated_exp_nlvl_flow"],  # BN F nlvl 2
                    }
                    | (
                        {
                            "b_ids": triplet["b_ids"][:, None],  # BN 1
                            "i_ids": triplet["i_ids"][:, None],  # BN 1
                            "pred_cls_queries": triplet["pred_cls_queries"][:, None],  # B 1 N C
                            "pred_certainty_queries": triplet["pred_certainty_queries"][
                                :, None
                            ],  # B 1 N 1
                        }
                        if "b_ids" in triplet.keys()
                        else {}
                    )
                )

        from retracker.models.utils.misc import _robust_cat

        # Collect all outputs to build res
        video_res = {
            "mkpts1_f": torch.cat([res["mkpts1_f"] for res in video_res_list], dim=1),
            "pred_visibles": torch.cat([res["pred_visibles"] for res in video_res_list], dim=1),
            "fine_certainty_logits": torch.cat(
                [res["fine_certainty_logits"] for res in video_res_list], dim=1
            ),
            "fine_occlusion_logits": torch.cat(
                [res["fine_occlusion_logits"] for res in video_res_list], dim=1
            ),
            "updated_pos_nlvl": _robust_cat(
                [res["updated_pos_nlvl"] for res in video_res_list], dim=1
            ),
            "updated_occ_nlvl": _robust_cat(
                [res["updated_occ_nlvl"] for res in video_res_list], dim=1
            ),
            "updated_exp_nlvl": _robust_cat(
                [res["updated_exp_nlvl"] for res in video_res_list], dim=1
            ),
            "is_masked_list": [
                res.get("is_masked_list", [False, False, False]) for res in video_res_list
            ],
            "updated_pos_nlvl_flow": _robust_cat(
                [res["updated_pos_nlvl_flow"] for res in video_res_list], dim=1
            ),
            "updated_occ_nlvl_flow": _robust_cat(
                [res["updated_occ_nlvl_flow"] for res in video_res_list], dim=1
            ),
            "updated_exp_nlvl_flow": _robust_cat(
                [res["updated_exp_nlvl_flow"] for res in video_res_list], dim=1
            ),
        }
        video_res |= (
            {
                "b_ids": torch.cat([res["b_ids"] for res in video_res_list], dim=1),
                "i_ids": torch.cat([res["i_ids"] for res in video_res_list], dim=1),
                "pred_cls_queries": torch.cat(
                    [res["pred_cls_queries"] for res in video_res_list], dim=1
                ),  # B F N C
                "pred_certainty_queries": torch.cat(
                    [res["pred_certainty_queries"] for res in video_res_list], dim=1
                ),  # B F N 1
            }
            if "b_ids" in video_res_list[0].keys()
            else {}
        )
        video_res |= {"gt_cls_map_i_16x_j_8x": None}

        from einops import rearrange

        # B F N to BN F
        gt_cls_ids = rearrange(data["gt_cls_ids_S"][:, 1:], "B F N -> (B N) F")
        gt_cls_ids_vis = rearrange(data["gt_cls_ids_vis_S"][:, 1:], "B F N -> (B N) F")
        video_res |= {
            "gt_cls_ids": gt_cls_ids,  #
            "gt_cls_ids_vis": gt_cls_ids_vis,
        }

        if "flow_gt" in data.keys():
            video_res.update(
                {"flow_gt": data["flow_gt"], "flow_valid_mask": data["flow_valid_mask"]}
            )
            video_res.update({"queries": data["queries"]})  # B, N, 2

        # Calculate loss
        with self.profiler.profile("3. Compute losses"):
            loss_dict = self.loss.causal_video_forward(data, video_res)

        with self.profiler.profile("4. Summarization tracking results"):
            B = data["images"].shape[0]

            if "loss_f" not in data["loss_scalars"]:
                data["loss_scalars"]["loss_f"] = torch.tensor(0)

            video_logs.update(
                {
                    "avg_loss_scalar": data["loss_scalars"]["loss"],
                    "avg_c_loss_scalar": 0,
                    "avg_f_loss_scalar": data["loss_scalars"]["loss_f"],
                    "video_trajs": video_res["mkpts1_f"],
                    "b_ids": video_res["b_ids"] if "b_ids" in video_res.keys() else None,
                    "fine_certainty_logits": video_res["fine_certainty_logits"],
                    "fine_occlusion_logits": video_res["fine_occlusion_logits"],
                    "video_occs": ~video_res["pred_visibles"],
                    "updated_pos_nlvl_flow": video_res["updated_pos_nlvl_flow"],
                }
            )
        return video_res, loss_dict

    def _prepare_data_for_viz_and_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares and gathers all necessary data for visualization and evaluation.
        This function is pure and has no side effects.
        """
        B = batch["images"].shape[0]

        from einops import rearrange

        pred_trajs = rearrange(batch["pred_trajs"].detach(), "(B N) F C -> B F N C", B=B)
        pred_occs = rearrange(batch["pred_occs"].detach(), "(B N) F C -> B F N C", B=B)[
            ..., 0
        ]  # C=1

        # Pad the query coords to pred_trajs to keep the same length
        pred_trajs = torch.cat([batch["trajs"][:, 0:1], pred_trajs], dim=1)
        pred_occs = torch.cat([~batch["valids"][:, 0:1], pred_occs], dim=1)

        return {
            "trajs": batch["trajs"],
            "pred_trajs": pred_trajs,
            "identifiers": batch["scene_name"],
            "images": batch["images"],
            "pred_occs": pred_occs,
            "valids": batch["valids"],
            # for matching plotting
            "is_matching_plotting": "T_0to1" in batch.keys(),
            "T_0to1": batch["T_0to1"] if "T_0to1" in batch.keys() else None,
            "K0": batch["K0"] if "K0" in batch.keys() else None,
            "K1": batch["K1"] if "K1" in batch.keys() else None,
            "scale0": batch["scale0"] if "scale0" in batch.keys() else None,
            "scale1": batch["scale1"] if "scale1" in batch.keys() else None,
            "dataset_name": batch["dataset_name"] if "dataset_name" in batch.keys() else None,
            "pair_names": batch["pair_names"] if "pair_names" in batch.keys() else None,
            "image0": batch["images"][:, 0],
            "image1": batch["images"][:, -1],
            "queries": batch["trajs"][:, 0],
            "b_ids": self.video_logs["b_ids"] if "b_ids" in self.video_logs.keys() else None,
            "fine_certainty_logits": self.video_logs["fine_certainty_logits"],
            "fine_occlusion_logits": self.video_logs["fine_occlusion_logits"],
            "updated_pos_nlvl_flow": self.video_logs["updated_pos_nlvl_flow"],
        }

    ############################################
    ############## debug logging ###############
    ############################################
    def debug_mode_init(self):
        # Debug configuration - check both top-level and model-level
        self.debug_mode = getattr(self.config.retracker_config, "debug_mode", False)
        self.debug_batches = getattr(self.config.retracker_config, "debug_batches", 5)

        # If not found at top-level, check in model config
        if not hasattr(self.config.retracker_config, "debug_mode"):
            self.debug_mode = getattr(self.config.retracker_config, "debug_mode", False)
        if not hasattr(self.config.retracker_config, "debug_batches"):
            self.debug_batches = getattr(self.config.retracker_config, "debug_batches", 5)

    def _log_debug_info(self, batch, batch_idx, mode="train"):
        """Comprehensive debug logging for distributed training verification"""
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        step = self.global_step

        # Basic rank and batch info
        CONSOLE.print(
            f"[dim][Rank {rank}/{world_size}] {mode.upper()} Batch {batch_idx} (Step {step})[/dim]"
        )

        # Log basic info to tensorboard
        self.log(f"debug/{mode}/rank", rank, on_step=True)
        self.log(f"debug/{mode}/batch_idx", batch_idx, on_step=True)
        self.log(f"debug/{mode}/global_step", step, on_step=True)

        # Image data analysis
        if "images" in batch:
            images = batch["images"]
            img_shape = images.shape
            img_mean = images.mean().item()
            img_std = images.std().item()
            img_min = images.min().item()
            img_max = images.max().item()

            # Use first image's statistics as unique identifier
            first_img_mean = images[0, 0].mean().item()
            first_img_std = images[0, 0].std().item()

            CONSOLE.print(
                f"[dim][Rank {rank}] Images - Shape: {img_shape}, Mean: {img_mean:.6f}, Std: {img_std:.6f}[/dim]"
            )
            CONSOLE.print(
                f"[dim][Rank {rank}] First Image - Mean: {first_img_mean:.6f}, Std: {first_img_std:.6f}[/dim]"
            )

            # Log to tensorboard
            self.log(f"debug/{mode}/images/mean", img_mean, on_step=True)
            self.log(f"debug/{mode}/images/std", img_std, on_step=True)
            self.log(f"debug/{mode}/images/min", img_min, on_step=True)
            self.log(f"debug/{mode}/images/max", img_max, on_step=True)
            self.log(f"debug/{mode}/images/first_img_mean", first_img_mean, on_step=True)
            self.log(f"debug/{mode}/images/first_img_std", first_img_std, on_step=True)

            # Log image dimensions
            self.log(f"debug/{mode}/images/batch_size", img_shape[0], on_step=True)
            self.log(f"debug/{mode}/images/num_frames", img_shape[1], on_step=True)
            self.log(f"debug/{mode}/images/height", img_shape[2], on_step=True)
            self.log(f"debug/{mode}/images/width", img_shape[3], on_step=True)

        # Scene/sequence information
        if "scene_name" in batch:
            scene_name = (
                batch["scene_name"][0]
                if isinstance(batch["scene_name"], list)
                else batch["scene_name"]
            )
            CONSOLE.print(f"[dim][Rank {rank}] Scene: {scene_name}[/dim]")
            # Create a simple hash of scene name for tensorboard
            scene_hash = hash(scene_name) % 10000  # Keep it reasonable for tensorboard
            self.log(f"debug/{mode}/scene_hash", scene_hash, on_step=True)

        # Trajectory data analysis
        if "trajs" in batch:
            trajs = batch["trajs"]
            traj_shape = trajs.shape
            traj_mean = trajs.mean().item()
            traj_std = trajs.std().item()

            CONSOLE.print(
                f"[dim][Rank {rank}] Trajectories - Shape: {traj_shape}, Mean: {traj_mean:.6f}, Std: {traj_std:.6f}[/dim]"
            )

            self.log(f"debug/{mode}/trajs/mean", traj_mean, on_step=True)
            self.log(f"debug/{mode}/trajs/std", traj_std, on_step=True)
            self.log(f"debug/{mode}/trajs/num_points", traj_shape[2], on_step=True)

        # Validity mask analysis
        if "valids" in batch:
            valids = batch["valids"]
            valid_ratio = valids.float().mean().item()
            CONSOLE.print(f"[dim][Rank {rank}] Valid points ratio: {valid_ratio:.4f}[/dim]")

            self.log(f"debug/{mode}/valids/ratio", valid_ratio, on_step=True)

        # Loss information (if available)
        if "loss_scalars" in batch:
            loss_value = batch["loss_scalars"].get("loss", 0)
            if hasattr(loss_value, "item"):
                loss_value = loss_value.item()
            CONSOLE.print(f"[dim][Rank {rank}] Loss: {loss_value:.6f}[/dim]")

            self.log(f"debug/{mode}/loss", loss_value, on_step=True)

        # Dataset information
        if "dataset_name" in batch:
            dataset_name = (
                batch["dataset_name"][0]
                if isinstance(batch["dataset_name"], list)
                else batch["dataset_name"]
            )
            CONSOLE.print(f"[dim][Rank {rank}] Dataset: {dataset_name}[/dim]")
            dataset_hash = hash(dataset_name) % 1000
            self.log(f"debug/{mode}/dataset_hash", dataset_hash, on_step=True)

        # Create a unique data signature for this batch
        data_signature = self._create_data_signature(batch)
        CONSOLE.print(f"[dim][Rank {rank}] Data signature: {data_signature}[/dim]")

        # Convert data signature to a numeric hash for logging
        data_signature_hash = hash(data_signature) % 1000000  # Keep it reasonable for tensorboard
        self.log(f"debug/{mode}/data_signature_hash", data_signature_hash, on_step=True)

        # Log some sample images to tensorboard (only for first batch to avoid clutter)
        if batch_idx == 0 and "images" in batch and self.trainer.global_rank == 0:
            self._log_sample_images(batch, mode)

    def _create_data_signature(self, batch):
        """Create a unique signature for the batch data"""
        signature_parts = []

        if "images" in batch:
            # Use first image's statistics
            first_img = batch["images"][0, 0]
            signature_parts.append(f"img_mean_{first_img.mean().item():.6f}")
            signature_parts.append(f"img_std_{first_img.std().item():.6f}")

        if "scene_name" in batch:
            scene_name = (
                batch["scene_name"][0]
                if isinstance(batch["scene_name"], list)
                else batch["scene_name"]
            )
            signature_parts.append(f"scene_{scene_name}")

        if "trajs" in batch:
            traj_mean = batch["trajs"].mean().item()
            signature_parts.append(f"traj_mean_{traj_mean:.6f}")

        return "_".join(signature_parts)

    def _log_sample_images(self, batch, mode):
        """Log sample images to tensorboard for visual inspection"""
        if "images" not in batch:
            return

        images = batch["images"]
        # Log first frame of first batch
        first_frame = images[0, 0]  # [H, W, C] or [C, H, W]

        # Normalize to [0, 1] for tensorboard
        if first_frame.max() > 1.0:
            first_frame = first_frame / 255.0

        # Ensure the image is in [C, H, W] format for tensorboard
        if first_frame.dim() == 3:
            if first_frame.shape[0] == 3 or first_frame.shape[0] == 1:  # Already [C, H, W]
                pass  # No need to permute
            else:  # [H, W, C] format
                first_frame = first_frame.permute(2, 0, 1)  # [C, H, W]

        # TensorBoard expects [C, H, W] format, not [1, C, H, W]
        # Remove batch dimension if it exists
        if first_frame.dim() == 4 and first_frame.shape[0] == 1:
            first_frame = first_frame.squeeze(0)  # [C, H, W]

        if not self.loggers:
            return
        exp = getattr(self.loggers[0], "experiment", None)
        if exp is None or not hasattr(exp, "add_image"):
            return

        exp.add_image(
            f"debug/{mode}/sample_image_rank_{self.trainer.global_rank}",
            first_frame,
            global_step=self.global_step,
        )

    ############################################
    ############## log #########################
    ############################################
    def _log_visualization(self, prepared_data: Dict[str, Any], mode: str):
        """
        Generates and logs videos for visualization purposes.
        This function's only side effect is logging to the experiment tracker.
        """
        if not self.loggers:
            return
        exp = getattr(self.loggers[0], "experiment", None)
        if exp is None or not hasattr(exp, "add_video"):
            return

        # For matching dataset visualization
        if prepared_data["is_matching_plotting"]:
            self._log_matching_pairs(
                prepared_data, 0, self.config, mode=mode, has_gt_pos=True, plot_flow=False
            )
        else:
            # for tracking dataset:
            from retracker.training.utils.plotting import make_tracking_videos

            videos = make_tracking_videos(
                prepared_data, self.config, mode=self.config.viz_config.plot_mode
            )

            for k, v in videos.items():
                for plot_idx, vid in enumerate(v):
                    exp.add_video(  # type: ignore[attr-defined]
                        f"{mode}_tracking/{k}/pair-{plot_idx}",
                        torch.from_numpy(vid).permute(0, 3, 1, 2)[None],
                        fps=4,
                        global_step=self.global_step,
                    )

    ################################################
    ############## metrics #########################
    ################################################
    def _compute_metrics(self, batch, mode="Train"):
        # for matching dataset:
        identifiers = batch["identifiers"]
        if batch["is_matching_plotting"]:
            # for matching dataset:
            self.compute_metrics_for_matching_task(
                batch, 0, self.config, mode=mode, has_gt_pos=True
            )
            metrics = {"is_matching_plotting": batch["is_matching_plotting"]}

        else:
            # for tracking dataset:
            with self.profiler.profile("Copmute metrics"):
                from retracker.training.utils.metrics import compute_traj_errors_video

                compute_traj_errors_video(batch, self.config)

                metrics = {
                    # to filter duplicate pairs caused by DistributedSampler
                    "traj_errs": batch["traj_errs"],
                    "traj_errs_raw": batch["traj_errs_raw"],
                    "occ_errs": batch["occ_errs"],
                    "identifiers": identifiers,
                }
            metrics.update({"is_matching_plotting": batch["is_matching_plotting"]})
        ret_dict = {"metrics": metrics}
        return ret_dict, identifiers

    ###########################################################
    ################# matching metrics ########################
    ###########################################################
    def compute_metrics_for_matching_task(
        self, batch, batch_idx, config, mode="Train", has_gt_pos=False
    ):
        if mode.lower() == "train":
            plot_conf_thresh = -1
        else:
            plot_conf_thresh = 0.2

        if "mconf" not in batch.keys():  # tmp fix for matching task
            batch.update({"mconf": torch.sigmoid(batch["fine_certainty_logits"])})

        # get valid mkpts0, mkpts1, mconf from batch
        B = batch["queries"].shape[0]  # batch size
        M = batch["queries"].shape[1]  # number of queries per sample
        mkpts0 = batch["queries"].reshape(-1, 2)  # [B, N, 2] -> [B*N, 2]
        mkpts1 = batch["pred_trajs"][:, 1].reshape(-1, 2)  # [B, F, N, 2] -> [B*N, 2]
        mconf = batch["mconf"].reshape(-1, 1)  # [B*N, 1]

        # Generate b_ids with correct shape [B*M] for multi-batch support
        # b_ids contains batch indices: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
        b_ids = torch.arange(B, device=batch["queries"].device)[:, None].expand(B, M).reshape(-1)

        # (Optional) remove invalid queries if has instruction(occlusion or invalid queries for example)
        mconf_mask = (batch["mconf"] > plot_conf_thresh).reshape(-1)
        if mode.lower() == "train":
            valid = batch["valids"][:, 1].reshape(-1)
        else:
            valid = torch.ones_like(mconf_mask)

        #############
        # compute PCK if has gt position
        #  - keep skeptical matches as well
        #############
        if has_gt_pos:
            # plot M matches after refinement:
            mkpts1_gt = batch["trajs"][:, 1, :M].reshape(-1, 2)
            mkpts1_err = torch.norm(mkpts1 - mkpts1_gt, dim=1)
            valid = batch["valids"][:, 1, :M].reshape(-1)
            mkpts1_err = mkpts1_err[valid]  # keep skeptical queries
            batch.update({"mkpts1_err": mkpts1_err})

        ############
        # remove unreliable matches before compute pose metrics
        ############
        batch.update(
            {
                "mkpts0_f": mkpts0[valid & mconf_mask],
                "mkpts1_f": mkpts1[valid & mconf_mask],
                "mconf": mconf[valid & mconf_mask],
                "b_ids": b_ids[valid & mconf_mask],
                "m_bids": b_ids[valid & mconf_mask],
                "mkpts0_f_all": mkpts0[valid],
                "mkpts1_f_all": mkpts1[valid],
                "b_ids_all": b_ids[valid],
            }
        )

        ret_dict, _ = self._compute_matching_metrics(batch)
        batch.update({**ret_dict})
        return ret_dict, _

    def _compute_matching_metrics(self, batch):
        """Compute epipolar loss, pose error,"""
        from retracker.training.utils.metrics import (
            compute_pose_errors,
            compute_symmetrical_epipolar_errors,
        )

        compute_symmetrical_epipolar_errors(batch, self.config)  # compute epi_errs for each match
        compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(batch["pair_names"])
        bs = batch["image0"].size(0)

        # Compute num_matches and percent_inliers for all batches
        num_matches = []
        percent_inliers = []
        for b in range(bs):
            b_mask = batch["b_ids"] == b
            n_matches = batch["mconf"][b_mask].shape[0]
            num_matches.append(n_matches)
            if n_matches > 0:
                percent_inliers.append(batch["inliers"][b].sum() / n_matches)
            else:
                percent_inliers.append(1.0)

        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
            "epi_errs": [
                batch["epi_errs"].cpu().numpy() for b in range(bs)
            ],  # [batch['b_ids'] == b]
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
            "num_matches": num_matches,
            "percent_inliers": percent_inliers,
        }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def _log_matching_pairs(
        self, batch, batch_idx, config, mode="Train", has_gt_pos=False, plot_flow=False
    ):
        """log matches for matching task
        Input:
            batch: dict,
                'queries': torch.Tensor, [B, F=1, N, 2]
                'pred_trajs': torch.Tensor, [B, F=1, N, 2]
                'mconf': torch.Tensor, [B, F=1, N]
                'b_ids': torch.Tensor, [B, F=1, N]
        Output:
            updated batch: dict,
        """
        from retracker.training.utils.plotting import make_matching_figures

        if mode.lower() == "train":
            if has_gt_pos:
                figures = make_matching_figures(batch, config, mode="dist", plot_flow=plot_flow)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(  # type: ignore
                        f"{mode}/train_match_dist/{k}", v, self.global_step
                    )
        elif mode.lower() == "validation":
            figures = make_matching_figures(batch, config, mode="evaluation")
            for k, v in figures.items():
                self.logger.experiment.add_figure(  # type: ignore
                    f"{mode}/validation_match_pose/{k}", v, self.global_step
                )

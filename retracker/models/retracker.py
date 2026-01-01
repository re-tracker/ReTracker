import logging
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from retracker.models.utils.geo_helper import (
    _rotate,
    _rotate_coords,
    _rotate_coords_back,
)
from retracker.utils.profiler import global_profiler

from .retrackerbase import ReTrackerBase
from .utils.misc import (
    extract_interpolated_features,
    get_pos_enc,
    queries_to_coarse_ids,
)
from .utils.model_utils import (
    _preprocess_data,
)


logger = logging.getLogger(__name__)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")


# Constants for better maintainability
class ReTrackerConfig:
    """Configuration constants for ReTracker"""

    SIGMA_NOISE: float = 0.1
    TRACKING_VISIBILITY_THRESHOLD: float = 0.1
    MATCHING_VISIBILITY_THRESHOLD: float = 0.5
    VISIBILITY_THRESHOLD: float = TRACKING_VISIBILITY_THRESHOLD  # Deprecated alias
    UNCERTAINTY_THRESHOLD: float = 0.02
    MATCHING_PREDICTION_THRESHOLD: float = 0.5  # 0.6
    LOSE_TRACK_THRESHOLD: int = 5
    AUGMENTATION_SCALE_FACTOR: float = 0.05
    GRID_SIZE: tuple[int, int] = (8, 8)
    DEFAULT_CERTAINTY_VALUE: int = 100
    DEFAULT_OCCLUSION_VALUE: int = -100


class ReTracker(ReTrackerBase):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # action config
        self.config = config
        self.train_coarse = config["train_coarse"]
        self.train_fine = config["train_fine"]
        self.model_task_type = config["model_task_type"]
        self.use_temporal_dino = config["use_temporal_dino"]
        self.use_jitter_aug = config["use_jitter_aug"]
        self.using_matching_prior = config["use_matching_prior"]
        self.use_compicated_replace_strategy = False
        self.task_mode: str = "tracking"
        # Task-specific visibility thresholds (can be overridden at runtime)
        if hasattr(config, "get"):
            tracking_thr = config.get(
                "tracking_visibility_threshold", ReTrackerConfig.TRACKING_VISIBILITY_THRESHOLD
            )
            matching_thr = config.get(
                "matching_visibility_threshold", ReTrackerConfig.MATCHING_VISIBILITY_THRESHOLD
            )
        else:
            tracking_thr = ReTrackerConfig.TRACKING_VISIBILITY_THRESHOLD
            matching_thr = ReTrackerConfig.MATCHING_VISIBILITY_THRESHOLD
        self.tracking_visibility_threshold = float(tracking_thr)
        self.matching_visibility_threshold = float(matching_thr)
        # sample some points for fine training
        self.fixed_fine_queries_num = config["fixed_fine_queries_num"]

        # High-resolution inference support
        # coarse_resolution: resolution for global stage (default 512x512)
        # The global stage uses DINO backbone which expects fixed resolution
        # Setting this allows inference on higher resolution images
        self.coarse_resolution = config.get("coarse_resolution", (512, 512))
        self.enable_highres_inference = config.get("enable_highres_inference", False)

        # tmp attr for training
        self.sliding_res = []

    def set_task_mode(self, task_mode: str) -> None:
        """Set runtime task mode: 'tracking' or 'matching'."""
        if task_mode not in ("tracking", "matching"):
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        self.task_mode = task_mode

    def set_visibility_thresholds(
        self, tracking: float | None = None, matching: float | None = None
    ) -> None:
        """Set task-specific visibility thresholds."""
        if tracking is not None:
            self.tracking_visibility_threshold = float(tracking)
        if matching is not None:
            self.matching_visibility_threshold = float(matching)

    def _get_visibility_threshold(self) -> float:
        """Return visibility threshold based on current task mode."""
        if getattr(self, "task_mode", "tracking") == "matching":
            return self.matching_visibility_threshold
        return self.tracking_visibility_threshold

    def set_highres_inference(
        self, enable: bool = True, coarse_resolution: tuple[int, int] = (512, 512)
    ) -> None:
        """Enable or disable high-resolution inference mode.

        In high-resolution inference mode:
        - The global/coarse stage still uses a fixed resolution (default 512x512)
          because the DINO backbone and cls_decoder are designed for that resolution.
        - The refinement stage uses the original high-resolution input images
          for better local feature matching.

        This allows processing images larger than 512x512 while maintaining
        compatibility with the trained model weights.

        Args:
            enable (bool): Whether to enable high-resolution inference.
            coarse_resolution (Tuple[int, int]): Resolution (H, W) for the coarse/global stage.
                                                  Should match the resolution the model was trained on.
                                                  Default is (512, 512).

        Example:
            >>> model.set_highres_inference(True, coarse_resolution=(512, 512))
            >>> # Now you can pass images of any resolution (e.g., 1024x1024)
            >>> data = {'image0': img0_1024, 'image1': img1_1024, 'queries': queries_1024}
            >>> model.forward(data)
            >>> # Outputs will be in the original 1024x1024 coordinate space
        """
        self.enable_highres_inference = enable
        self.coarse_resolution = coarse_resolution
        if enable:
            logger.info(
                f"[INFO] High-resolution inference enabled. Coarse stage resolution: {coarse_resolution}"
            )
        else:
            logger.info("[INFO] High-resolution inference disabled.")

    def set_max_queries(self, max_queries: int = None) -> None:
        """Set the maximum number of query points for inference.

        During training, query points are limited to `fixed_fine_queries_num` (default 500)
        to save memory. For inference, you may want to process more points.

        Args:
            max_queries (int or None): Maximum number of query points.
                                       If None, removes the limit (uses a very large number).

        Example:
            >>> model.set_max_queries(2000)  # Allow up to 2000 query points
            >>> model.set_max_queries(None)  # Remove limit entirely
        """
        if max_queries is None:
            self.fixed_fine_queries_num = 100000  # Effectively no limit
            logger.info("[INFO] Query point limit removed (set to 100000)")
        else:
            self.fixed_fine_queries_num = max_queries
            logger.info(f"[INFO] Max query points set to {max_queries}")

    def forward(self, data: dict[str, Any], mode: str = "train", use_aug: bool = False) -> int:
        """Forward pass with optional augmentation

        Args:
            data (dict): Input data dictionary
            mode (str): Training mode ('train', 'eval', etc.)
            use_aug (bool): Whether to use augmentation

        Returns:
            int: Return code (0 for success)
        """
        if not use_aug:
            return self._forward(data, mode, dump_last_pred=True)

        # Augmentation mode - optimize to reduce memory usage
        return self._forward_with_augmentation(data, mode)

    def _forward_with_augmentation(self, data: dict[str, Any], mode: str) -> int:
        """Optimized forward pass with augmentation"""
        # First forward pass without augmentation
        with torch.no_grad():
            self._forward(data, mode, is_aug=False, dump_last_pred=False)

        # Store original results
        original_results = {
            "pos": data["mkpts1_f"].clone(),
            "vis": data["pred_visibles"].squeeze(-1),
            "exp": data["fine_certainty_logits"].squeeze(-1),
        }

        # Rotate image and forward pass
        original_image = data["image1"]
        data["image1"] = _rotate(original_image)

        with torch.no_grad():
            self._forward(data, mode, is_aug=True, dump_last_pred=False)

        # Process rotated results
        rotated_results = {
            "pos": _rotate_coords_back(data["mkpts1_f"], data["hw_i"][0]),
            "vis": data["pred_visibles"].squeeze(-1),
            "exp": data["fine_certainty_logits"].squeeze(-1),
        }

        # Restore original image
        data["image1"] = original_image

        # Select better predictions based on certainty
        update_mask = rotated_results["exp"] > original_results["exp"]

        # Combine results efficiently
        final_pos = torch.where(
            update_mask[..., None].expand(-1, -1, 2),
            rotated_results["pos"],
            original_results["pos"],
        ).reshape(data["bs"], -1, 2)

        final_vis = torch.where(update_mask, rotated_results["vis"], original_results["vis"])
        final_exp = torch.where(update_mask, rotated_results["exp"], original_results["exp"])

        # Update memory and data
        self._update_memory_and_data(data, final_pos, final_vis, final_exp)

        return 0

    def _compute_velocity(self, current_pos: torch.Tensor) -> torch.Tensor:
        """Compute instantaneous velocity from current and previous positions

        Args:
            current_pos (torch.Tensor): Current frame positions, shape (B, N, 2)

        Returns:
            torch.Tensor: Instantaneous velocity, shape (B, N, 2)
        """
        prev_res = self.mem_manager.get_memory("last_frame_pred_dict", default=None)

        if prev_res is not None and "updated_pos" in prev_res:
            prev_pos = prev_res["updated_pos"]
            # Ensure shapes match for velocity calculation
            if prev_pos.shape == current_pos.shape:
                # Calculate instantaneous velocity: velocity = current_pos - prev_pos
                velocity = current_pos - prev_pos
            else:
                # Handle shape mismatch: initialize velocity as zeros
                velocity = torch.zeros_like(current_pos)
        else:
            # First frame: no previous position, initialize velocity as zeros
            velocity = torch.zeros_like(current_pos)

        # Filter out abnormal velocities: set to zero if magnitude exceeds threshold
        velocity_magnitude = velocity.norm(dim=-1)  # (B, N)
        velocity_threshold = 50
        abnormal_mask = velocity_magnitude > velocity_threshold  # (B, N)
        velocity = torch.where(abnormal_mask.unsqueeze(-1), torch.zeros_like(velocity), velocity)

        return velocity

    def _update_memory_and_data(
        self,
        data: dict[str, Any],
        final_pos: torch.Tensor,
        final_vis: torch.Tensor,
        final_exp: torch.Tensor,
    ) -> None:
        """Update memory and data with final results, including instantaneous velocity"""
        # Compute instantaneous velocity
        velocity = self._compute_velocity(final_pos)

        # Get batch size from final_pos
        B = final_pos.shape[0]

        # Store results including velocity
        self.mem_manager.memory["last_frame_pred_dict"] = {
            "updated_pos": final_pos,
            "updated_occlusion": final_vis.reshape(B, -1, 1),
            "updated_certainty": final_exp.reshape(B, -1, 1),
            "updated_velocity": velocity,
        }

        data.update(
            {
                "mkpts1_f": final_pos.reshape(-1, 1, 2),
                "pred_visibles": (
                    (torch.sigmoid(final_exp)) * (1 - torch.sigmoid(final_vis))
                    > self._get_visibility_threshold()
                ).reshape(-1, 1, 1),
            }
        )

    def _forward(
        self,
        data: dict[str, Any],
        mode: str = "train",
        is_aug: bool = False,
        dump_last_pred: bool = True,
    ) -> int:
        """Main forward pass method

        Args:
            data (dict): Input data dictionary
            mode (str): Training mode
            is_aug (bool): Whether this is an augmentation pass
            dump_last_pred (bool): Whether to save last prediction to memory

        Returns:
            int: Return code (0 for success)
        """
        if True:
            with global_profiler.record("forward"):
                # ====================
                # 0. init data
                # ====================
                with global_profiler.record("preprocess"):
                    _preprocess_data(data)

                # ====================
                # High-resolution inference support
                # ====================
                # Store original resolution for refinement stage
                original_hw = data["hw_i"]

                # Check if we need to resize for coarse stage (high-res inference mode)
                use_highres = self.enable_highres_inference and not self.training
                coarse_h, coarse_w = self.coarse_resolution

                if use_highres and (original_hw[0] != coarse_h or original_hw[1] != coarse_w):
                    # Debug print (only first time)
                    if not hasattr(self, "_highres_logged"):
                        logger.info(
                            f"[HighRes] Enabled: input={original_hw}, coarse={self.coarse_resolution}"
                        )
                        self._highres_logged = True

                    # Store references (not clones) for later restoration
                    image0_highres = data["image0"]
                    image1_highres = data["image1"]
                    images_highres = data.get("images", None)

                    # Resize images to coarse resolution for global stage
                    data["image0"] = F.interpolate(
                        data["image0"],
                        size=(coarse_h, coarse_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    data["image1"] = F.interpolate(
                        data["image1"],
                        size=(coarse_h, coarse_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # Also update data['images'] which is used by _global_stage
                    if images_highres is not None:
                        data["images"] = torch.stack([data["image0"], data["image1"]], dim=1)

                    # Scale queries to coarse resolution (need clone here since we modify values)
                    scale_h = coarse_h / original_hw[0]
                    scale_w = coarse_w / original_hw[1]
                    data["queries"] = data["queries"].clone()
                    data["queries"][..., 0] = data["queries"][..., 0] * scale_w  # x
                    data["queries"][..., 1] = data["queries"][..., 1] * scale_h  # y

                    # Update hw_i for coarse stage
                    data["hw_i"] = (coarse_h, coarse_w)

                    # Recompute normalized queries for coarse resolution
                    from .utils.misc import normalize_keypoints

                    data["queries_norm_pos"] = normalize_keypoints(
                        data["queries"], H=coarse_h, W=coarse_w
                    )

                # ====================
                # 1. global stage:
                # ====================
                with global_profiler.record("coarse_stage"):
                    coarse_res = self._get_coarse_results(data, is_aug)

                # ====================
                # Scale coarse results back to original resolution if high-res mode
                # ====================
                if use_highres and (original_hw[0] != coarse_h or original_hw[1] != coarse_w):
                    scale_h_inv = original_hw[0] / coarse_h
                    scale_w_inv = original_hw[1] / coarse_w

                    # Scale updated_pos back to original resolution
                    coarse_res["updated_pos"][..., 0] = (
                        coarse_res["updated_pos"][..., 0] * scale_w_inv
                    )
                    coarse_res["updated_pos"][..., 1] = (
                        coarse_res["updated_pos"][..., 1] * scale_h_inv
                    )

                    # Scale queries back to original resolution (maintain filtering)
                    coarse_res["queries"][..., 0] = coarse_res["queries"][..., 0] * scale_w_inv
                    coarse_res["queries"][..., 1] = coarse_res["queries"][..., 1] * scale_h_inv

                    # Restore original resolution data for refinement stage
                    data["image0"] = image0_highres
                    data["image1"] = image1_highres
                    data["hw_i"] = original_hw

                    # Restore data['images'] for refinement stage
                    if images_highres is not None:
                        data["images"] = images_highres

                    # Clean up intermediate tensors
                    del image0_highres, image1_highres
                    if images_highres is not None:
                        del images_highres

                data["queries"] = coarse_res["queries"]
                if "pred_cls_queries" in coarse_res.keys():
                    data.update({"pred_cls_queries": coarse_res["pred_cls_queries"]})
                self._causal_video_matching_aug(data, coarse_res)
                # ====================
                # 2. refinement stage:
                # ====================
                with global_profiler.record("fine_stage"):
                    fine_res = self._get_fine_results(data, coarse_res, mode)

                # ==================================
                # 3. get final output
                # ==================================
                with global_profiler.record("postprocess"):
                    self._process_final_output(data, fine_res, coarse_res, dump_last_pred)

            return 0

    def _get_coarse_results(self, data: dict[str, Any], is_aug: bool) -> dict[str, Any]:
        """Get coarse matching results based on task type"""
        # Check if we need to filter queries for coarse matching (streaming scenario)
        first_frame_mask = data.get("first_frame_query_mask", None)

        if self.model_task_type in ["image_matching", "video_matching"]:
            return self._global_stage(data)
        elif self.model_task_type == "causal_video_matching":
            if not self.train_coarse:
                coarse_res = self._follow_previous_preds(data)
                coarse_res["queries"] = data["queries"][:, : self.fixed_fine_queries_num]
                coarse_res["updated_pos"] = coarse_res["updated_pos"][
                    :, : self.fixed_fine_queries_num
                ]
                coarse_res["updated_occlusion"] = coarse_res["updated_occlusion"][
                    :, : self.fixed_fine_queries_num
                ]
                coarse_res["updated_certainty"] = coarse_res["updated_certainty"][
                    :, : self.fixed_fine_queries_num
                ]
                coarse_res["mconf_logits_coarse"] = (
                    torch.ones_like(coarse_res["queries"][..., 0:1]) * 10
                )
                data.update(coarse_res)
                return data
            if self.using_matching_prior:
                # For streaming: only do coarse matching for first-frame queries
                if first_frame_mask is not None:
                    # Filter queries to only include first-frame ones for coarse matching
                    original_queries = data["queries"].clone()
                    first_frame_queries = data["queries"][:, first_frame_mask]  # [B, N_first, 2]

                    # Handle case when there are no first-frame queries
                    if first_frame_queries.shape[1] == 0:
                        # No first-frame queries, skip coarse matching entirely
                        # Just use initial positions for all queries
                        B, N_total = original_queries.shape[:2]
                        device = original_queries.device

                        coarse_res = {
                            "queries": original_queries,
                            "updated_pos": original_queries.clone(),
                            "updated_occlusion": torch.zeros((B, N_total, 1), device=device),
                            "updated_certainty": torch.ones((B, N_total, 1), device=device) * 0.5,
                            "updated_velocity": torch.zeros((B, N_total, 2), device=device),
                            "mconf_logits_coarse": torch.zeros((B, N_total, 1), device=device),
                        }
                        data.update(coarse_res)
                        return coarse_res

                    # Create filtered data for coarse matching
                    filtered_data = data.copy()
                    filtered_data["queries"] = first_frame_queries

                    # Recompute queries_norm_pos for filtered queries only
                    # This ensures feat_d0_en_with_queries only contains first-frame queries
                    from retracker.models.utils.misc import normalize_keypoints

                    filtered_data["queries_norm_pos"] = normalize_keypoints(
                        first_frame_queries, H=data["hw_i"][0], W=data["hw_i"][1]
                    )
                    filtered_data["queries_real_num"] = first_frame_queries.shape[1]

                    # Do coarse matching only on first-frame queries
                    coarse_res = self._global_stage(filtered_data)

                    # Expand coarse results to include all queries
                    # For non-first-frame queries, use initial positions
                    B, N_total = original_queries.shape[:2]
                    device = original_queries.device

                    # Initialize full-size coarse results
                    full_updated_pos = torch.zeros((B, N_total, 2), device=device)
                    full_updated_occlusion = torch.zeros((B, N_total, 1), device=device)
                    full_updated_certainty = torch.ones((B, N_total, 1), device=device)
                    full_updated_velocity = torch.zeros((B, N_total, 2), device=device)
                    full_mconf_logits_coarse = torch.zeros((B, N_total, 1), device=device)

                    # Fill in coarse results for first-frame queries
                    first_frame_indices = torch.where(first_frame_mask)[0]
                    for i, idx in enumerate(first_frame_indices):
                        if i < coarse_res["updated_pos"].shape[1]:
                            full_updated_pos[:, idx, :] = coarse_res["updated_pos"][:, i, :]
                            full_updated_occlusion[:, idx, :] = coarse_res["updated_occlusion"][
                                :, i, :
                            ]
                            full_updated_certainty[:, idx, :] = coarse_res["updated_certainty"][
                                :, i, :
                            ]
                            if "updated_velocity" in coarse_res:
                                full_updated_velocity[:, idx, :] = coarse_res["updated_velocity"][
                                    :, i, :
                                ]
                            if "mconf_logits_coarse" in coarse_res:
                                full_mconf_logits_coarse[:, idx, :] = coarse_res[
                                    "mconf_logits_coarse"
                                ][:, i, :]

                    # For non-first-frame queries, use initial positions
                    non_first_mask = ~first_frame_mask
                    if non_first_mask.any():
                        full_updated_pos[:, non_first_mask, :] = original_queries[
                            :, non_first_mask, :
                        ]
                        full_updated_certainty[:, non_first_mask, :] = (
                            0.5  # Medium certainty for new queries
                        )
                        full_mconf_logits_coarse[:, non_first_mask, :] = (
                            0.0  # Low confidence for new queries
                        )

                    # Update coarse_res with full-size results
                    coarse_res["queries"] = original_queries
                    coarse_res["updated_pos"] = full_updated_pos
                    coarse_res["updated_occlusion"] = full_updated_occlusion
                    coarse_res["updated_certainty"] = full_updated_certainty
                    coarse_res["updated_velocity"] = full_updated_velocity
                    coarse_res["mconf_logits_coarse"] = full_mconf_logits_coarse

                    # Also update pred_cls_map, pred_certainty_map, pred_occlusion_map if needed
                    # These are used by coarse_matching_by_cls, but may not be needed for non-first-frame queries
                    if "pred_cls_map" in coarse_res:
                        # Keep original pred_cls_map (it's per-pixel, not per-query)
                        pass
                else:
                    # Normal path: do coarse matching on all queries
                    coarse_res = self._global_stage(data)

                if self.train_fine:
                    coarse_res["queries"] = coarse_res["queries"][:, : self.fixed_fine_queries_num]
                    coarse_res["updated_pos"] = coarse_res["updated_pos"][
                        :, : self.fixed_fine_queries_num
                    ]
                    coarse_res["updated_occlusion"] = coarse_res["updated_occlusion"][
                        :, : self.fixed_fine_queries_num
                    ]
                    coarse_res["updated_certainty"] = coarse_res["updated_certainty"][
                        :, : self.fixed_fine_queries_num
                    ]
                    coarse_res["mconf_logits_coarse"] = coarse_res["mconf_logits_coarse"][
                        :, : self.fixed_fine_queries_num
                    ]
                    coarse_res = self._filter_confident_matches(data, coarse_res, is_aug=is_aug)
                return coarse_res
            else:
                return self._follow_previous_preds(data)
        else:
            raise ValueError(f"Unknown model_task_type: {self.model_task_type}")

    def _get_fine_results(
        self, data: dict[str, Any], coarse_res: dict[str, Any], mode: str
    ) -> dict[str, Any]:
        """Get fine refinement results"""
        # updated_pos with velocity
        # if self.mem_manager.exists('last_frame_pred_dict'):
        #     velocity = self.mem_manager.get_memory('last_frame_pred_dict')['updated_velocity']
        #     coarse_res['updated_pos'] += velocity
        updated_pos = coarse_res["updated_pos"].reshape(-1, 1, 2)
        updated_occ = torch.zeros_like(coarse_res["updated_occlusion"].reshape(-1, 1, 1))
        updated_exp = torch.ones_like(coarse_res["updated_certainty"].reshape(-1, 1, 1))

        fine_pips_data = {
            "updated_pos": updated_pos,
            "updated_occlusion": updated_occ,
            "updated_certainty": updated_exp,
        }

        if self.train_fine:
            return self.causal_refinement(data, fine_pips_data, mode=mode, always_use_warped=False)
        else:
            return {
                "updated_pos": updated_pos,
                "occlusion_logits": torch.zeros_like(updated_pos[..., :1]),
                "certainty_logits": torch.ones_like(updated_pos[..., :1]),
            }

    def _process_final_output(
        self,
        data: dict[str, Any],
        fine_res: dict[str, Any],
        coarse_res: dict[str, Any],
        dump_last_pred: bool,
    ) -> None:
        """Process and update final output data"""
        updated_pos = fine_res["updated_pos"]
        updated_pos_nlvl = fine_res.get("updated_pos_nlvl", None)
        updated_occ_nlvl = fine_res.get("updated_occ_nlvl", None)
        updated_exp_nlvl = fine_res.get("updated_exp_nlvl", None)

        updated_pos_nlvl_flow = fine_res.get("updated_pos_nlvl_flow", None)
        updated_occ_nlvl_flow = fine_res.get("updated_occ_nlvl_flow", None)
        updated_exp_nlvl_flow = fine_res.get("updated_exp_nlvl_flow", None)

        fine_occlusion_logits = fine_res["occlusion_logits"]
        fine_certainty_logits = fine_res["certainty_logits"]

        is_masked_list = fine_res.get("is_masked_list", None)
        # Calculate visibility with numerical stability
        pred_visibles, visibility_scores = self._calculate_visibility(
            fine_certainty_logits, fine_occlusion_logits
        )

        # Update data dictionary
        data.update(
            {
                "mkpts1_f": updated_pos,
                "pred_visibles": pred_visibles,
                "visibility_scores": visibility_scores,  # Raw float visibility scores
                "fine_occlusion_logits": fine_occlusion_logits,
                "fine_certainty_logits": fine_certainty_logits,
                "updated_pos_nlvl": updated_pos_nlvl,
                "updated_occ_nlvl": updated_occ_nlvl,
                "updated_exp_nlvl": updated_exp_nlvl,
                "updated_pos_nlvl_flow": updated_pos_nlvl_flow,
                "updated_occ_nlvl_flow": updated_occ_nlvl_flow,
                "updated_exp_nlvl_flow": updated_exp_nlvl_flow,
                "is_masked_list": is_masked_list,
            }
        )

        if dump_last_pred:
            # Reshape updated_pos to (B, N, 2) for velocity calculation
            updated_pos_reshaped = updated_pos.reshape(data["bs"], -1, 2)
            # Compute instantaneous velocity
            velocity = self._compute_velocity(updated_pos_reshaped)

            self.mem_manager.memory["last_frame_pred_dict"] = {
                "updated_pos": updated_pos_reshaped,
                "updated_occlusion": coarse_res["updated_occlusion"],
                "updated_certainty": coarse_res["updated_certainty"],
                "mconf_logits_coarse": coarse_res["mconf_logits_coarse"],
                "updated_velocity": velocity,
            }

        # Handle NaN values with better fallback strategy
        self._handle_nan_values(updated_pos, updated_pos_nlvl)

    def _calculate_visibility(
        self, certainty_logits: torch.Tensor, occlusion_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate visibility with numerical stability.

        Returns:
            pred_visibles: Boolean tensor, visibility > threshold
            visibility_scores: Float tensor, raw visibility scores (certainty * (1 - occlusion))
        """
        certainty = torch.sigmoid(certainty_logits)
        occlusion = torch.sigmoid(occlusion_logits)
        visibility_scores = certainty * (1 - occlusion)
        pred_visibles = visibility_scores > self._get_visibility_threshold()
        return pred_visibles, visibility_scores

    def _handle_nan_values(self, updated_pos: torch.Tensor, updated_pos_nlvl: torch.Tensor) -> None:
        """Handle NaN values in position predictions"""
        if torch.isnan(updated_pos).any():
            nan_mask = torch.isnan(updated_pos.max(dim=-1)[0])
            if updated_pos_nlvl is not None:
                # Use middle level as fallback
                fallback_pos = updated_pos_nlvl[:, :, 1]
                updated_pos[nan_mask] = fallback_pos[nan_mask]
            else:
                # If no multi-level predictions, use zeros
                updated_pos[nan_mask] = 0.0

    def _global_stage(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pairwise matching: provide coarse tracks"""
        # Extract DINO features
        with global_profiler.record("dino_backbone"):
            feats_d = self._extract_dino_features(data)

        # Process features
        with global_profiler.record("dino_process"):
            feat_d0_en_with_queries, feat_d1_en, dino_pe = self._process_dino_features(
                feats_d, data
            )

        # Compute affinity matrices and GPS
        with global_profiler.record("affinity_gps"):
            gps = self._compute_affinity_and_gps(feat_d0_en_with_queries, feat_d1_en, dino_pe, data)

        # Decode predictions
        with global_profiler.record("decode"):
            predictions = self._decode_predictions(gps, feat_d0_en_with_queries, data)

        # Update data and return coarse results
        data.update(predictions)
        # obtain coarse flow
        data.update(
            {
                "pred_cls_map": predictions["pred_cls_map"],
                "pred_certainty_map": predictions["pred_certainty_map"],
                "pred_occlusion_map": predictions["pred_occlusion_map"],
            }
        )
        data = self.coarse_matching_by_cls(data)
        return data

    def _extract_dino_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Extract DINO backbone features"""
        if self.use_dinov3:
            # When using DINOv3, get 16x features from the unified backbone
            # DINOv3 backbone returns (16x, 8x, 2x) when return_type='multi'
            # We need to temporarily get only 16x for global matching
            # So we call it with return_type='dino' or extract the first output
            input_tensor = torch.cat([data["image0"], data["image1"]], dim=0)
            # Temporarily set return_type to 'dino' to get only 16x features
            # Or we can extract from multi output
            all_feats = self.backbone(input_tensor)
            if isinstance(all_feats, tuple):
                # return_type='multi': (16x, 8x, 2x)
                feats_d = all_feats[0]  # Extract 16x features
            else:
                # return_type='dino': only 16x
                feats_d = all_feats
            return feats_d
        else:
            # Traditional DINOv2 setup
            return self.dino_backbone(torch.cat([data["image0"], data["image1"]], dim=0))

    def _process_dino_features(
        self, feats_d: torch.Tensor, data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process DINO features for matching"""
        data.update({"feats_d": feats_d})
        feat_d0, feat_d1 = feats_d.split(data["bs"])
        data.update({"hw_d": feat_d0.shape[2:]})

        # Get positional encoding
        dino_pe = get_pos_enc(feat_d1, self.pos_conv)
        dino_pe = rearrange(dino_pe, "n c h w -> n (h w) c")

        # Reshape features
        feat_d0 = rearrange(feat_d0, "n c h w -> n (h w) c")
        feat_d1 = rearrange(feat_d1, "n c h w -> n (h w) c")
        data.update({"feat_d1": feat_d1})

        # Get query coordinates
        scale_d = data["hw_i"][1] / data["hw_d"][1]
        b_ids, i_ids = queries_to_coarse_ids(data["queries"], data["hw_d"], scale_d)
        data.update({"m_bids": b_ids, "b_ids": b_ids, "i_ids": i_ids})

        # Encode features
        feat_d0_en, feat_d1_en = self.dino_encoder(feat_d0), self.dino_encoder(feat_d1)

        # Extract query features
        feat_d0_queries = extract_interpolated_features(
            data["queries_norm_pos"], feat_d0_en, size_hw=data["hw_d"]
        )
        feat_d0_en_with_queries = torch.cat([feat_d0_en, feat_d0_queries], dim=1)

        # Apply temporal attention if needed
        if self.use_temporal_dino and "video_matching" in self.model_task_type:
            feat_d0_en_with_queries, feat_d1_en = self.temporal_attn(
                b_ids, i_ids, feat_d0_en_with_queries, feat_d1_en, mem_manager=self.mem_manager
            )

        return feat_d0_en_with_queries, feat_d1_en, dino_pe

    def _compute_affinity_and_gps(
        self,
        feat_d0_en_with_queries: torch.Tensor,
        feat_d1_en: torch.Tensor,
        dino_pe: torch.Tensor,
        data: dict[str, Any],
    ) -> torch.Tensor:
        """Compute affinity matrices and GPS"""
        feat_mix0_en = torch.cat([feat_d0_en_with_queries], dim=2)
        feat_mix1_en = torch.cat([feat_d1_en], dim=2)

        affinity_matrix_01 = self.K(feat_mix0_en, feat_mix1_en)
        affinity_matrix_11 = self.K(feat_mix1_en, feat_mix1_en)

        # Add regularization for numerical stability
        sigma_noise = ReTrackerConfig.SIGMA_NOISE * torch.eye(
            affinity_matrix_11.shape[-1], device=affinity_matrix_11.device
        )
        affinity_matrix_11_inv = torch.linalg.inv(affinity_matrix_11 + sigma_noise)

        # Compute GPS
        gps = torch.einsum("bnm,bmk,bkd->bnd", affinity_matrix_01, affinity_matrix_11_inv, dino_pe)

        return gps

    def _decode_predictions(
        self,
        gps: torch.Tensor,
        feat_d0_en_with_queries: torch.Tensor,
        data: dict[str, Any],
        decode_anchor: bool = True,
    ) -> dict[str, Any]:
        """Decode final predictions"""
        res_dict = {}
        q_num = data["queries_real_num"]
        # Decode features
        z = self.dino_decoder(encoder_out=gps, tokens=feat_d0_en_with_queries)
        queries_tokens = z[:, -q_num:]
        anchor_tokens = z[:, :-q_num]

        if decode_anchor:
            tokens = torch.cat([anchor_tokens, queries_tokens], dim=1)
            pred_cls_logit = self.cls_decoder(tokens)
            pred_certainty = self.certainty_decoder(tokens)
            pred_occlusion = self.occlusion_decoder(tokens)

            pred_cls_queries = pred_cls_logit[:, -q_num:]
            pred_certainty_queries = pred_certainty[:, -q_num:]
            pred_occlusion_queries = pred_occlusion[:, -q_num:]
            pred_cls_map_logit = pred_cls_logit[:, :-q_num]
            pred_certainty_map = pred_certainty[:, :-q_num]
            pred_occlusion_map = pred_occlusion[:, :-q_num]

            pred_cls_map = rearrange(
                pred_cls_map_logit[..., :-1],
                "b (h w) c -> b c h w",
                h=data["hw_d"][0],
                w=data["hw_d"][1],
            )

            res_dict.update(
                {
                    "pred_cls_queries": pred_cls_queries,
                    "pred_certainty_queries": pred_certainty_queries,
                    "pred_occlusion_queries": pred_occlusion_queries,
                    "pred_cls_map": pred_cls_map,
                    "pred_certainty_map": pred_certainty_map,
                    "pred_occlusion_map": pred_occlusion_map,
                }
            )

        else:
            # Get predictions
            pred_cls_queries = self.cls_decoder(queries_tokens)
            pred_certainty_queries = self.certainty_decoder(queries_tokens)
            pred_occlusion_queries = self.occlusion_decoder(queries_tokens)

            res_dict.update(
                {
                    "pred_cls_queries": pred_cls_queries,
                    "pred_certainty_queries": pred_certainty_queries,
                    "pred_occlusion_queries": pred_occlusion_queries,
                }
            )

        return res_dict

    def _follow_previous_preds(self, data: dict[str, Any]) -> dict[str, Any]:
        """get coarse location from previous prediction"""
        coarse_res: dict = {}
        prev_res = self.mem_manager.get_memory("last_frame_pred_dict", default=None)
        if prev_res is None:
            queries = data["queries"][:, : data["queries_real_num"]]
            coarse_res = {
                "updated_pos": queries,
                "updated_certainty": torch.ones_like(queries[..., 0:1]),
                "updated_occlusion": torch.zeros_like(queries[..., 0:1]),
                "updated_velocity": torch.zeros_like(
                    queries
                ),  # Initialize velocity as zeros for first frame
            }
        else:
            coarse_res = prev_res
        return coarse_res

    def _causal_video_matching_aug(self, data: dict[str, Any], coarse_res: dict[str, Any]) -> None:
        """Add Gaussian jitter noise to coarse predictions for training augmentation."""
        if (
            self.training
            and self.model_task_type == "causal_video_matching"
            and self.use_jitter_aug
        ):
            scale_factor = ReTrackerConfig.AUGMENTATION_SCALE_FACTOR * data["hw_i"][0]
            noise = torch.randn_like(coarse_res["updated_pos"]) * scale_factor
            coarse_res["updated_pos"] = coarse_res["updated_pos"] + noise

    def _filter_confident_matches(
        self, data: dict[str, Any], coarse_res: dict[str, Any], is_aug: bool
    ) -> dict[str, Any]:
        """Filter confident matches for next stage"""
        # replace mask:
        _coarse_res: dict = {}
        prev_res = self.mem_manager.get_memory("last_frame_pred_dict", default=None)
        if is_aug and prev_res is not None:
            prev_res["updated_pos"] = _rotate_coords(prev_res["updated_pos"], data["hw_i"][1])
        if prev_res is None:
            # queries = data['queries'][:,:data['queries_real_num']]
            queries = coarse_res["queries"]
            prev_res = {
                "queries": queries,
                "updated_pos": queries,
                "updated_certainty": torch.ones_like(queries[..., 0:1]),
                "updated_occlusion": torch.zeros_like(queries[..., 0:1]),
                "mconf_logits_coarse": torch.ones_like(queries[..., 0:1]) * 10,
            }
            # prev_res = coarse_res

        task_mode = getattr(self, "task_mode", "tracking")
        if task_mode == "matching":
            # Always trust coarse predictions in matching mode.
            replace_mask = torch.ones_like(coarse_res["updated_certainty"], dtype=torch.bool)
        else:
            if self.use_compicated_replace_strategy:
                is_uncertainy_pred = (
                    1 - torch.sigmoid(prev_res["updated_certainty"])
                ) < ReTrackerConfig.UNCERTAINTY_THRESHOLD
                _unc_count = self.mem_manager.get_memory(
                    "unc_count", default=torch.zeros_like(is_uncertainy_pred).long()
                )
                _unc_count = torch.where(
                    is_uncertainy_pred, _unc_count + 1, torch.zeros_like(_unc_count).long()
                )
                self.mem_manager.memory["unc_count"] = _unc_count
                lose_track_mask = _unc_count > ReTrackerConfig.LOSE_TRACK_THRESHOLD
                USE_MATCHING_PREDICTION_THRESHOLD = torch.where(
                    lose_track_mask, 0.2, ReTrackerConfig.MATCHING_PREDICTION_THRESHOLD
                )
            else:
                USE_MATCHING_PREDICTION_THRESHOLD = ReTrackerConfig.MATCHING_PREDICTION_THRESHOLD

            # and mconf_logits_coarse
            replace_mask = (
                torch.sigmoid(coarse_res["updated_certainty"]) > USE_MATCHING_PREDICTION_THRESHOLD
            ) & (
                torch.sigmoid(coarse_res["mconf_logits_coarse"]) > USE_MATCHING_PREDICTION_THRESHOLD
            )
        _coarse_res["updated_pos"] = torch.where(
            replace_mask, coarse_res["updated_pos"], prev_res["updated_pos"]
        )
        _coarse_res["updated_certainty"] = torch.where(
            replace_mask, coarse_res["updated_certainty"], prev_res["updated_certainty"]
        )
        _coarse_res["updated_occlusion"] = torch.where(
            replace_mask, coarse_res["updated_occlusion"], prev_res["updated_occlusion"]
        )
        _coarse_res["mconf_logits_coarse"] = torch.where(
            replace_mask, coarse_res["mconf_logits_coarse"], prev_res["mconf_logits_coarse"]
        )
        _coarse_res["queries"] = coarse_res["queries"]
        return _coarse_res

    def _extract_features(self, data: dict[str, Any]) -> None:
        """extract features from backbone(s)"""
        input_tensor = torch.cat([data["image0"], data["image1"]], dim=0)

        if self.use_dinov3:
            # DINOv3 can output all levels (16x, 8x, 2x) in one forward pass
            if self.training:
                all_feats = checkpoint(self.backbone, input_tensor, use_reentrant=False)
            else:
                all_feats = self.backbone(input_tensor)

            # DINOv3 returns (feats_16x, feats_8x, feats_2x) when return_type='multi'
            feats_d, feats_c, feats_f = all_feats

            # Normalize features
            feats_c = feats_c / feats_c.shape[1] ** 0.5
            feats_f = feats_f / feats_f.shape[1] ** 0.5
        else:
            # Traditional setup: separate DINO and ResNet backbones
            feats_d = self.dino_backbone(input_tensor)
            if self.training:
                feats_c, feats_f = checkpoint(self.backbone, input_tensor, use_reentrant=False)
            else:
                feats_c, feats_f = self.backbone(input_tensor)

            # Normalize features
            feats_c = feats_c / feats_c.shape[1] ** 0.5
            feats_f = feats_f / feats_f.shape[1] ** 0.5

        data.update(
            {
                "hw0_d": feats_d.shape[2:],
                "hw1_d": feats_d.shape[2:],
                "hw0_c": feats_c.shape[2:],
                "hw1_c": feats_c.shape[2:],
                "hw0_f": feats_f.shape[2:],
                "hw1_f": feats_f.shape[2:],
                "feats_d": feats_d,
                "feats_c": feats_c,
                "feats_f": feats_f,
            }
        )

        return

    def causal_refinement(
        self,
        data: dict[str, Any],
        coarse_res: dict[str, Any],
        mode: str,
        always_use_warped: bool = False,
    ) -> dict[str, Any]:
        """refactored causal refinement
        1. prepare feature maps
        2. warp features;
        3. prepare 2 entries: unwarped+coarse pos / warped + query pos ;
        """
        # 1. extract features
        with global_profiler.record("extract_features"):
            self._extract_features(data)

        # 2. prepare 2 entries: unwarped+coarse pos / warped + query pos ;
        with global_profiler.record("prepare_features"):
            feats_d = data["feats_d"]
            feats_c = data["feats_c"]
            feats_f = data["feats_f"]

            feats_d = rearrange(feats_d, "(B F) C H W -> B F (H W) C", B=data["bs"])
            feats_c = rearrange(feats_c, "(B F) C H W -> B F (H W) C", B=data["bs"])
            feats_f = rearrange(feats_f, "(B F) C H W -> B F (H W) C", B=data["bs"])
            feats_d0, feats_d1 = feats_d.split(1, dim=1)
            feats_c0, feats_c1 = feats_c.split(1, dim=1)
            feats_f0, feats_f1 = feats_f.split(1, dim=1)

            fine_matches_data = {
                "hw_i": data["hw_i"],
                "hw0_d": data["hw0_d"],
                "hw0_c": data["hw0_c"],
                "hw0_f": data["hw0_f"],
                "hw1_f": data["hw1_f"],
                # sample some points for fine training
                "queries": data["queries"][:, : data["queries_real_num"]][
                    :, : self.fixed_fine_queries_num
                ],  # B N 2
            }

            fine_matches_data.update(
                {
                    "image0": data["image0"],
                    "image1": data["image1"],
                }
            )

            fine_matches_data.update(
                {
                    "updated_occlusion": coarse_res["updated_occlusion"],  # BxN, s, 1
                    "updated_certainty": coarse_res["updated_certainty"],  # BxN, s, 1
                    "updated_pos": coarse_res["updated_pos"],  # BxN, s=1, 2
                }
            )

        with global_profiler.record("matches_refinement"):
            refined_matches = self.matches_refinement(
                fine_matches_data,
                [feats_d0, feats_c0, feats_f0],
                [feats_d1, feats_c1, feats_f1],
                None,
                is_get_causal_context=True,
                always_use_warped=always_use_warped,
            )

        res = {
            "updated_pos": refined_matches["updated_pos"],  # BxN, s, 2
            "updated_pos_nlvl": refined_matches["updated_pos_nlvl"],  # BxN, s, nlvl, 2
            "occlusion_logits": refined_matches["occlusion_logits"],
            "certainty_logits": refined_matches["certainty_logits"],
            "updated_occ_nlvl": refined_matches["updated_occ_nlvl"],  # BxN, s, nlvl, 1
            "updated_exp_nlvl": refined_matches["updated_exp_nlvl"],  # BxN, s, nlvl, 1
            "is_masked_list": refined_matches.get("is_masked_list"),
        }
        if "updated_pos_nlvl_flow" in refined_matches:
            res.update(
                {
                    "updated_pos_nlvl_flow": refined_matches[
                        "updated_pos_nlvl_flow"
                    ],  # BxN, s, nlvl, 2
                    "updated_occ_nlvl_flow": refined_matches[
                        "updated_occ_nlvl_flow"
                    ],  # BxN, s, nlvl, 1
                    "updated_exp_nlvl_flow": refined_matches[
                        "updated_exp_nlvl_flow"
                    ],  # BxN, s, nlvl, 1
                }
            )
        return res

    def matching_only(self, data: dict[str, Any]) -> None:
        """Function for matching task with grid queries"""
        device = data["image0"].device
        b, _, h, w = data["image0"].shape

        # Generate grid queries
        queries = self._generate_grid_queries(h, w, b, device)

        # Randomly select queries
        data["queries"] = self._select_random_queries(queries)

        # Forward pass
        self.forward(data)

    def _generate_grid_queries(self, h: int, w: int, b: int, device: torch.device) -> torch.Tensor:
        """Generate grid-based queries"""
        grid_hw = ReTrackerConfig.GRID_SIZE
        queries = torch.stack(
            torch.meshgrid(
                torch.arange(0, w, grid_hw[1], device=device) + grid_hw[1] // 2,
                torch.arange(0, h, grid_hw[0], device=device) + grid_hw[0] // 2,
            ),
            dim=-1,
        ).float()
        return repeat(queries, "h w c -> B (h w) c", B=b)

    def _select_random_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """Randomly select a subset of queries"""
        random_indices = torch.randperm(queries.shape[1])[: self.config["queries_keypoints_num"]]
        return queries[:, random_indices]

    @torch.no_grad()
    def video_forward(
        self,
        data: dict[str, Any],
        mode: str = "train",
        use_aug: bool = False,
        return_dense_flow: bool = False,
    ) -> dict[str, Any]:
        """Video forward pass for evaluation

        Args:
            data (dict): {
                'images': (torch.Tensor): (B, F, 3, H, W)
                'queries': (torch.Tensor): [B, N, 2], (x,y) for test
            }
            return_dense_flow: If True, return dense flow predictions (updated_pos_nlvl_flow)
                              for dense matching. Default False to save memory.

        Returns:
            dict: Video tracking results. When return_dense_flow=True, also includes
                  'updated_pos_nlvl_flow' with shape (N, nlvl, T, 49, 2)
        """
        self.mem_manager.reset_all_memory()

        # Initialize video results
        if return_dense_flow:
            video_res = self._initialize_video_results(data)
        else:
            video_res = self._initialize_video_results_minimal(data)

        # Process video frames
        # Lazy import to keep model code decoupled from dataset/eval utilities at import time.
        from retracker.data.video_sequences import construct_triplets_for_eval

        triplet_N = construct_triplets_for_eval(data)
        for _idx, triplet in enumerate(triplet_N):
            ret_state = self.forward(triplet, use_aug=use_aug)
            if ret_state:  # exception
                return ret_state

            # Extract results (full or minimal based on return_dense_flow)
            if return_dense_flow:
                video_res.append(self._extract_triplet_results(triplet))
            else:
                video_res.append(self._extract_triplet_results_minimal(triplet))

            # Clean up triplet to free GPU memory
            for key in list(triplet.keys()):
                if isinstance(triplet[key], torch.Tensor):
                    del triplet[key]

            # Periodic memory cleanup
            if _idx % 50 == 0:
                torch.cuda.empty_cache()

        # Combine all results
        if return_dense_flow:
            return self._combine_video_results(video_res)
        else:
            return self._combine_video_results_minimal(video_res)

    def _initialize_video_results_minimal(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Initialize video results with first frame (minimal version)"""
        queries = data["queries"].transpose(0, 1)  # [B=1, N, C] to [BN, 1, C]
        q_conf = queries[..., 0:1]
        pred_visibles = torch.ones_like(q_conf).bool()
        # First frame has full visibility (score = 1.0)
        visibility_scores = torch.ones_like(q_conf).float()

        return [
            {
                "mkpts1_f": queries.cpu(),  # Move to CPU immediately
                "pred_visibles": pred_visibles.cpu(),
                "visibility_scores": visibility_scores.cpu(),
            }
        ]

    def _extract_triplet_results_minimal(self, triplet: dict[str, Any]) -> dict[str, Any]:
        """Extract minimal results from a triplet (memory efficient)"""
        result = {
            "mkpts1_f": triplet["mkpts1_f"].cpu(),  # BN F 2
            "pred_visibles": triplet["pred_visibles"].cpu(),  # BN F 1
        }
        # Also extract visibility_scores if available
        if "visibility_scores" in triplet:
            result["visibility_scores"] = triplet["visibility_scores"].cpu()
        return result

    def _combine_video_results_minimal(self, video_res):
        """Combine all video frame results (minimal version)"""
        result = {
            "mkpts1_f": torch.cat([res["mkpts1_f"] for res in video_res], dim=1),
            "pred_visibles": torch.cat([res["pred_visibles"] for res in video_res], dim=1),
        }
        # Also combine visibility_scores if available
        if "visibility_scores" in video_res[0]:
            result["visibility_scores"] = torch.cat(
                [res["visibility_scores"] for res in video_res], dim=1
            )
        return result

    def _initialize_video_results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Initialize video results with first frame"""
        queries = data["queries"].transpose(0, 1)  # [B=1, N, C] to [BN, 1, C]
        q_conf = queries[..., 0:1]
        nlvl = self.corr_num_levels
        pred_visibles = torch.ones_like(q_conf).bool()
        zeros_like_q_conf = torch.zeros_like(q_conf)
        # First frame has full visibility (score = 1.0)
        visibility_scores = torch.ones_like(q_conf).float()

        return [
            {
                "mkpts1_f": queries,  # [BN, F=1, C]
                "pred_visibles": pred_visibles,
                "visibility_scores": visibility_scores,
                "fine_occlusion_logits": zeros_like_q_conf
                * ReTrackerConfig.DEFAULT_OCCLUSION_VALUE,
                "fine_certainty_logits": pred_visibles * ReTrackerConfig.DEFAULT_CERTAINTY_VALUE,
                "updated_pos_nlvl": repeat(queries, f"BN F C -> BN F {nlvl} C"),
                "updated_occ_nlvl": repeat(zeros_like_q_conf, f"BN F C -> BN F {nlvl} C"),
                "updated_exp_nlvl": repeat(zeros_like_q_conf, f"BN F C -> BN F {nlvl} C"),
                # Shape: (BN, nlvl, F, 49, C) - will be concatenated on F dimension (dim=2)
                "updated_pos_nlvl_flow": repeat(queries, f"BN F C -> BN {nlvl} F 49 C"),
                "updated_occ_nlvl_flow": repeat(zeros_like_q_conf, f"BN F C -> BN {nlvl} F 49 C"),
                "updated_exp_nlvl_flow": repeat(zeros_like_q_conf, f"BN F C -> BN {nlvl} F 49 C"),
            }
        ]

    def _extract_triplet_results(self, triplet: dict[str, Any]) -> dict[str, Any]:
        """Extract results from a triplet"""
        result = {
            "mkpts1_f": triplet["mkpts1_f"],  # BN F 2
            "pred_visibles": triplet["pred_visibles"],  # BN F 1
            "fine_occlusion_logits": triplet["fine_occlusion_logits"],
            "fine_certainty_logits": triplet["fine_certainty_logits"],
            "updated_pos_nlvl": triplet["updated_pos_nlvl"],  # BN F nlvl 2
            "updated_occ_nlvl": triplet["updated_occ_nlvl"],  # BN F nlvl 1
            "updated_exp_nlvl": triplet["updated_exp_nlvl"],  # BN F nlvl 1
            "updated_pos_nlvl_flow": triplet["updated_pos_nlvl_flow"],  # BN F nlvl 2
            "updated_occ_nlvl_flow": triplet["updated_occ_nlvl_flow"],  # BN F nlvl 1
            "updated_exp_nlvl_flow": triplet["updated_exp_nlvl_flow"],  # BN F nlvl 1
        }
        # Also extract visibility_scores if available
        if "visibility_scores" in triplet:
            result["visibility_scores"] = triplet["visibility_scores"]
        return result

    def _combine_video_results(self, video_res):
        """Combine all video frame results"""
        result = {
            "mkpts1_f": torch.cat([res["mkpts1_f"] for res in video_res], dim=1),
            "pred_visibles": torch.cat([res["pred_visibles"] for res in video_res], dim=1),
            "fine_certainty_logits": torch.cat(
                [res["fine_certainty_logits"] for res in video_res], dim=1
            ),
            "fine_occlusion_logits": torch.cat(
                [res["fine_occlusion_logits"] for res in video_res], dim=1
            ),
            "updated_pos_nlvl": torch.cat([res["updated_pos_nlvl"] for res in video_res], dim=1),
            "updated_occ_nlvl": torch.cat([res["updated_occ_nlvl"] for res in video_res], dim=1),
            "updated_exp_nlvl": torch.cat([res["updated_exp_nlvl"] for res in video_res], dim=1),
            "updated_pos_nlvl_flow": torch.cat(
                [res["updated_pos_nlvl_flow"] for res in video_res], dim=2
            ),
            "updated_occ_nlvl_flow": torch.cat(
                [res["updated_occ_nlvl_flow"] for res in video_res], dim=2
            ),
            "updated_exp_nlvl_flow": torch.cat(
                [res["updated_exp_nlvl_flow"] for res in video_res], dim=2
            ),
        }
        # Also combine visibility_scores if available
        if "visibility_scores" in video_res[0]:
            result["visibility_scores"] = torch.cat(
                [res["visibility_scores"] for res in video_res], dim=1
            )
        return result

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

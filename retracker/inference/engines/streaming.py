"""Streaming inference engine implementation (StreamingEngine).

The stable import path is still `retracker.inference.engine`.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from retracker.models import ReTracker
from retracker.inference.engines.offline import ReTrackerEngine
from retracker.utils.rich_utils import get_progress


class StreamingEngine(torch.nn.Module):
    """
    Streaming engine that supports dynamic query point addition during inference.
    Maintains internal state for continuous tracking across frames.
    """

    def __init__(
        self,
        retracker_model: Union[ReTracker, None] = None,
        ckpt_path: Union[str, None] = None,
        locotrack_ckpt_path: Union[str, None] = None,
        interp_shape: Tuple = (256, 256),
        enable_highres_inference: bool = False,
        coarse_resolution: Tuple[int, int] = (512, 512),
        query_batch_size: int = 256,
        fast_start: bool = False,
    ) -> None:
        super(StreamingEngine, self).__init__()
        self.interp_shape = interp_shape
        self.query_batch_size = query_batch_size

        # Initialize base engine for shared functionality
        self.base_engine = ReTrackerEngine(
            retracker_model=retracker_model,
            ckpt_path=ckpt_path,
            locotrack_ckpt_path=locotrack_ckpt_path,
            interp_shape=interp_shape,
            enable_highres_inference=enable_highres_inference,
            coarse_resolution=coarse_resolution,
            query_batch_size=query_batch_size,
            fast_start=fast_start,
        )
        
        # Expose model for compatibility with offline engine interface
        self.model = self.base_engine.model
        
        # Streaming state
        self.current_frame_idx: int = 0
        self.video_shape: Tuple[int, int, int] = None  # (C, H, W)
        self.queries: Tensor = None  # [B=1, N, 3] (t, x, y) - all active queries
        self.query_ids: Tensor = None  # [N] - unique IDs for each query
        self.tracks_history: Tensor = None  # [T_processed, N, 2] - trajectory history
        self.visibility_history: Tensor = None  # [T_processed, N] - visibility history
        self.next_query_id: int = 0
        self.is_initialized: bool = False
        
        # Frame cache for triplet construction
        self.frame_cache: List[Tensor] = []  # Cache last 2 frames [C, H, W]
    
    def _update_memory_for_new_queries(
        self, 
        active_queries: Tensor, 
        active_query_ids: Tensor, 
        query_positions: Tensor
    ):
        """
        Update memory manager state to handle dynamically added queries.
        Ensures last_frame_pred_dict has correct shape for current active queries.
        Maps previous queries to current ones based on query_ids.
        
        Args:
            active_queries: [1, N_active, 3] queries active at current frame
            active_query_ids: [N_active] query IDs
            query_positions: [1, N_active, 2] current positions in interp_shape space
        """
        mem_manager = self.base_engine.model.mem_manager
        device = query_positions.device
        N_active = active_queries.shape[1]
        
        if not mem_manager.exists('last_frame_pred_dict'):
            # First frame: initialize memory
            mem_manager.memory['last_frame_pred_dict'] = {
                'updated_pos': query_positions.reshape(1, N_active, 2),  # [1, N, 2]
                'updated_occlusion': torch.zeros((1, N_active, 1), device=device),
                'updated_certainty': torch.ones((1, N_active, 1), device=device) * 100,
                'updated_velocity': torch.zeros((1, N_active, 2), device=device),
                'mconf_logits_coarse': torch.ones((1, N_active, 1), device=device) * 10,  # Required by _filter_confident_matches
            }
            # Store query_ids mapping for future reference
            mem_manager.memory['_streaming_query_ids'] = active_query_ids.clone()
        else:
            # Get previous query IDs from memory (if stored)
            prev_query_ids = mem_manager.memory.get('_streaming_query_ids', None)
            prev_dict = mem_manager.memory['last_frame_pred_dict']
            prev_N = prev_dict['updated_pos'].shape[1]
            
            # Initialize new state tensors
            updated_pos = torch.zeros((1, N_active, 2), device=device)
            updated_occlusion = torch.zeros((1, N_active, 1), device=device)
            updated_certainty = torch.ones((1, N_active, 1), device=device) * 100
            updated_velocity = torch.zeros((1, N_active, 2), device=device)
            mconf_logits_coarse = torch.ones((1, N_active, 1), device=device) * 10  # Default value
            
            # Map previous queries to current ones based on query_ids
            if prev_query_ids is not None:
                # Create mapping: for each current query_id, find its index in previous queries
                for curr_idx, curr_qid in enumerate(active_query_ids):
                    prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                    if len(prev_idx) > 0:
                        # Query existed before: copy previous state
                        prev_idx = prev_idx[0].item()
                        updated_pos[0, curr_idx] = prev_dict['updated_pos'][0, prev_idx]
                        updated_occlusion[0, curr_idx] = prev_dict['updated_occlusion'][0, prev_idx]
                        updated_certainty[0, curr_idx] = prev_dict['updated_certainty'][0, prev_idx]
                        if 'updated_velocity' in prev_dict:
                            updated_velocity[0, curr_idx] = prev_dict['updated_velocity'][0, prev_idx]
                        if 'mconf_logits_coarse' in prev_dict:
                            mconf_logits_coarse[0, curr_idx] = prev_dict['mconf_logits_coarse'][0, prev_idx]
                    else:
                        # New query: initialize with current position
                        updated_pos[0, curr_idx] = query_positions[0, curr_idx]
            else:
                # Fallback: simple index-based mapping (assumes order preserved)
                min_N = min(prev_N, N_active)
                if min_N > 0:
                    updated_pos[0, :min_N] = prev_dict['updated_pos'][0, :min_N]
                    updated_occlusion[0, :min_N] = prev_dict['updated_occlusion'][0, :min_N]
                    updated_certainty[0, :min_N] = prev_dict['updated_certainty'][0, :min_N]
                    if 'updated_velocity' in prev_dict:
                        updated_velocity[0, :min_N] = prev_dict['updated_velocity'][0, :min_N]
                    if 'mconf_logits_coarse' in prev_dict:
                        mconf_logits_coarse[0, :min_N] = prev_dict['mconf_logits_coarse'][0, :min_N]
                
                # Initialize new queries
                if N_active > prev_N:
                    updated_pos[0, prev_N:] = query_positions[0, prev_N:]
            
            # Update memory
            mem_manager.memory['last_frame_pred_dict'] = {
                'updated_pos': updated_pos,
                'updated_occlusion': updated_occlusion,
                'updated_certainty': updated_certainty,
                'updated_velocity': updated_velocity,
                'mconf_logits_coarse': mconf_logits_coarse,  # Required by _filter_confident_matches
            }
            # Update query_ids mapping
            mem_manager.memory['_streaming_query_ids'] = active_query_ids.clone()
            
            # Update pips_refinement related memory keys
            self._update_pips_memory_for_new_queries(
                mem_manager, prev_query_ids, active_query_ids, N_active, device, query_positions
            )
    
    def _update_pips_memory_for_new_queries(
        self,
        mem_manager,
        prev_query_ids: Tensor,
        curr_query_ids: Tensor,
        N_active: int,
        device: torch.device,
        query_positions: Tensor
    ):
        """
        Extend memory keys used by pips_refinement module when new queries are added.
        Preserves existing query data and initializes new queries with zeros/defaults.
        
        Strategy: When query count increases, extend memory tensors along the BN dimension.
        New queries get zero-initialized values, existing queries keep their history.
        
        Args:
            mem_manager: Memory manager instance
            prev_query_ids: Previous query IDs tensor
            curr_query_ids: Current query IDs tensor
            N_active: Current number of active queries
            device: Device for tensors
        """
        # Check if query count changed
        if prev_query_ids is None or len(prev_query_ids) == 0:
            # First time: no need to extend, pips_refinement will initialize
            return
        
        prev_N = len(prev_query_ids)
        if prev_N == N_active:
            # Query count unchanged: no action needed
            return
        
        if prev_N > N_active:
            # Query count decreased: this shouldn't happen in streaming, but handle gracefully
            # Reset to let pips_refinement reinitialize
            if mem_manager.exists('updated_pos'):
                mem_manager.reset_memory('updated_pos')
            for level_idx in range(3):
                key = f'causal_corr_{level_idx}'
                if mem_manager.exists(key):
                    mem_manager.reset_memory(key)
            for iter_idx in range(3):
                key = f'walking_context_{iter_idx}'
                if mem_manager.exists(key):
                    mem_manager.reset_memory(key)
            return
        
        # Query count increased: extend memory tensors for new queries
        num_new = N_active - prev_N
        
        # Create mapping: which current queries are new
        new_query_mask = torch.ones(N_active, dtype=torch.bool, device=device)
        for curr_idx, curr_qid in enumerate(curr_query_ids):
            if (prev_query_ids == curr_qid).any():
                new_query_mask[curr_idx] = False
        
        # Handle updated_pos: shape [BN, F, 2]
        if mem_manager.exists('updated_pos'):
            prev_mem = mem_manager.memory['updated_pos']
            if prev_mem is not None:
                # Get shape info
                prev_BN = prev_mem.shape[0]
                F = prev_mem.shape[1] if prev_mem.dim() >= 2 else 1
                C = prev_mem.shape[2] if prev_mem.dim() >= 3 else 2
                
                # Create extended memory: [BN_new, F, C]
                # Map existing queries and initialize new ones
                new_mem = torch.zeros((N_active, F, C), device=device, dtype=prev_mem.dtype)
                
                # Map existing queries based on query_ids
                for curr_idx, curr_qid in enumerate(curr_query_ids):
                    prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                    if len(prev_idx) > 0:
                        # Existing query: copy data
                        prev_idx = prev_idx[0].item()
                        if prev_idx < prev_BN:
                            new_mem[curr_idx, :, :] = prev_mem[prev_idx, :, :]
                    else:
                        # New query: initialize with current position (use same across F)
                        pos_curr = query_positions[0, curr_idx].view(1, 2)
                        new_mem[curr_idx, :, :] = pos_curr.expand(F, -1)
                
                # Replace memory (pips_refinement expects this shape)
                mem_manager.memory['updated_pos'] = new_mem
        
        # Handle causal_corr_{0,1,2}: shape [BN, F, WW, C]
        for level_idx in range(3):
            key = f'causal_corr_{level_idx}'
            if mem_manager.exists(key):
                prev_mem = mem_manager.memory[key]
                if prev_mem is not None:
                    prev_BN = prev_mem.shape[0]
                    F = prev_mem.shape[1] if prev_mem.dim() >= 2 else 1
                    WW = prev_mem.shape[2] if prev_mem.dim() >= 3 else 49
                    C = prev_mem.shape[3] if prev_mem.dim() >= 4 else 256
                    
                    # Create extended memory
                    new_mem = torch.zeros((N_active, F, WW, C), device=device, dtype=prev_mem.dtype)
                    
                    # Map existing queries
                    for curr_idx, curr_qid in enumerate(curr_query_ids):
                        prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                        if len(prev_idx) > 0:
                            prev_idx = prev_idx[0].item()
                            if prev_idx < prev_BN:
                                new_mem[curr_idx, :, :, :] = prev_mem[prev_idx, :, :, :]
                    
                    mem_manager.memory[key] = new_mem
        
        # Handle walking_context_{0,1,2}: shape varies, typically [BN, C]
        for iter_idx in range(3):
            key = f'walking_context_{iter_idx}'
            if mem_manager.exists(key):
                prev_mem = mem_manager.memory[key]
                if prev_mem is not None and prev_mem.shape[0] != N_active:
                    prev_BN = prev_mem.shape[0]
                    
                    if prev_mem.dim() == 1:
                        # [BN] -> [N_active]
                        new_mem = torch.zeros((N_active,), device=device, dtype=prev_mem.dtype)
                        for curr_idx, curr_qid in enumerate(curr_query_ids):
                            prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                            if len(prev_idx) > 0:
                                prev_idx = prev_idx[0].item()
                                if prev_idx < prev_BN:
                                    new_mem[curr_idx] = prev_mem[prev_idx]
                            else:
                                new_mem[curr_idx] = 0
                        mem_manager.memory[key] = new_mem
                    elif prev_mem.dim() == 2:
                        # [BN, C] -> [N_active, C]
                        C = prev_mem.shape[1]
                        new_mem = torch.zeros((N_active, C), device=device, dtype=prev_mem.dtype)
                        # base token for new queries
                        base_token = getattr(self.base_engine.model.temporal_attn, 'walking_memory_token', None)
                        base_token = base_token.detach() if base_token is not None else None
                        for curr_idx, curr_qid in enumerate(curr_query_ids):
                            prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                            if len(prev_idx) > 0:
                                prev_idx = prev_idx[0].item()
                                if prev_idx < prev_BN:
                                    new_mem[curr_idx, :] = prev_mem[prev_idx, :]
                            else:
                                if base_token is not None:
                                    new_mem[curr_idx, :] = base_token.view(-1)[:C]
                        mem_manager.memory[key] = new_mem
                    elif prev_mem.dim() == 3:
                        # [BN, F, C] -> [N_active, F, C]
                        F_ctx, C = prev_mem.shape[1], prev_mem.shape[2]
                        new_mem = torch.zeros((N_active, F_ctx, C), device=device, dtype=prev_mem.dtype)
                        base_token = getattr(self.base_engine.model.temporal_attn, 'walking_memory_token', None)
                        base_token = base_token.detach() if base_token is not None else None
                        base_token = base_token.view(1, 1, -1) if base_token is not None else None
                        for curr_idx, curr_qid in enumerate(curr_query_ids):
                            prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                            if len(prev_idx) > 0:
                                prev_idx = prev_idx[0].item()
                                if prev_idx < prev_BN:
                                    new_mem[curr_idx, :, :] = prev_mem[prev_idx, :, :]
                            else:
                                if base_token is not None:
                                    new_mem[curr_idx, :, :] = base_token.expand(F_ctx, -1)
                        mem_manager.memory[key] = new_mem
                    # Higher dims: reset (shouldn't happen normally)
        
        # Handle temporal fusion memory keys: feat_0_{module_id} and feat_last_{module_id}
        # These are used by temporal_attn and need to match query count
        # Shape is [B, H*W + N_queries, C] where H*W is anchor points, N_queries is query points
        # Check for common module_ids (typically '0' for DINO features)
        for module_id in ['0', '1', '2']:  # Common module IDs
            for feat_key in [f'feat_0_{module_id}', f'feat_last_{module_id}']:
                if mem_manager.exists(feat_key):
                    prev_mem = mem_manager.memory[feat_key]
                    if prev_mem is not None:
                        # Check if shape needs updating
                        # feat shape is [B, H*W + N_queries, C]
                        if prev_mem.dim() == 3:
                            B, prev_N_total, C = prev_mem.shape
                            # Estimate H*W from prev_N_total (assuming it was H*W + N_prev_queries)
                            # We need to infer H*W from the difference
                            # For now, we'll try to detect if this is the anchor+queries format
                            # by checking if prev_N_total > N_active (which suggests anchor points are included)
                            
                            # Get current expected total: H*W + N_active
                            # We need to get H*W from somewhere - it should be consistent across frames
                            # For DINO features, H*W is typically 32*32 = 1024
                            # We can infer it from prev_N_total - N_prev_queries
                            if prev_query_ids is not None and len(prev_query_ids) > 0:
                                N_prev_queries = len(prev_query_ids)
                                # Estimate H*W = prev_N_total - N_prev_queries
                                HW_estimate = prev_N_total - N_prev_queries
                                
                                # Current expected total
                                curr_N_total = HW_estimate + N_active
                                
                                if prev_N_total != curr_N_total:
                                    # Need to update: separate anchor and query parts
                                    # Anchor part: [B, HW_estimate, C] (keep unchanged)
                                    # Query part: [B, N_prev_queries, C] (needs remapping)
                                    anchor_part = prev_mem[:, :HW_estimate, :]  # [B, HW_estimate, C]
                                    prev_query_part = prev_mem[:, HW_estimate:, :]  # [B, N_prev_queries, C]
                                    
                                    # Create new query part with correct size
                                    new_query_part = torch.zeros((B, N_active, C), device=device, dtype=prev_mem.dtype)
                                    for curr_idx, curr_qid in enumerate(curr_query_ids):
                                        prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                                        if len(prev_idx) > 0:
                                            prev_idx = prev_idx[0].item()
                                            if prev_idx < N_prev_queries:
                                                new_query_part[:, curr_idx, :] = prev_query_part[:, prev_idx, :]
                                    
                                    # Concatenate anchor and new query parts
                                    new_mem = torch.cat([anchor_part, new_query_part], dim=1)  # [B, HW_estimate + N_active, C]
                                    mem_manager.memory[feat_key] = new_mem
                            else:
                                # Cannot remap: reset (temporal_fusion will reinitialize)
                                mem_manager.reset_memory(feat_key)
                        elif prev_mem.dim() == 2:
                            # Handle [BN, C] format (less common for temporal fusion)
                            prev_BN = prev_mem.shape[0]
                            if prev_BN != N_active:
                                if prev_query_ids is not None and len(prev_query_ids) > 0:
                                    C = prev_mem.shape[1]
                                    new_mem = torch.zeros((N_active, C), device=device, dtype=prev_mem.dtype)
                                    for curr_idx, curr_qid in enumerate(curr_query_ids):
                                        prev_idx = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                                        if len(prev_idx) > 0:
                                            prev_idx = prev_idx[0].item()
                                            if prev_idx < prev_BN:
                                                new_mem[curr_idx, :] = prev_mem[prev_idx, :]
                                    mem_manager.memory[feat_key] = new_mem
                                else:
                                    mem_manager.reset_memory(feat_key)

    def _restore_memory_for_batch(
        self,
        saved_streaming_query_ids: Optional[Tensor],
        saved_last_frame_pred_dict: Optional[Dict],
        batch_query_ids: Tensor,
        batch_query_positions: Tensor,
        device: torch.device
    ):
        """
        Restore memory state for a batch of queries from the saved previous frame state.

        This is called during batch processing to ensure each batch starts with the
        correct memory state from the previous frame (not from a different batch).

        Args:
            saved_streaming_query_ids: Query IDs from previous frame (on CPU)
            saved_last_frame_pred_dict: Memory dict from previous frame (on CPU)
            batch_query_ids: Query IDs for current batch (on CPU)
            batch_query_positions: Query positions for current batch [1, N_batch, 2] (on GPU)
            device: Device for output tensors
        """
        mem_manager = self.base_engine.model.mem_manager
        N_batch = batch_query_positions.shape[1]

        # Ensure batch_query_ids is on CPU for comparison
        batch_query_ids_cpu = batch_query_ids.cpu() if batch_query_ids.is_cuda else batch_query_ids

        if saved_last_frame_pred_dict is None or saved_streaming_query_ids is None:
            # No previous frame state, initialize fresh
            mem_manager.memory['last_frame_pred_dict'] = {
                'updated_pos': batch_query_positions.clone().to(device),
                'updated_occlusion': torch.zeros((1, N_batch, 1), device=device),
                'updated_certainty': torch.ones((1, N_batch, 1), device=device) * 100,
                'updated_velocity': torch.zeros((1, N_batch, 2), device=device),
                'mconf_logits_coarse': torch.ones((1, N_batch, 1), device=device) * 10,
            }
            mem_manager.memory['_streaming_query_ids'] = batch_query_ids_cpu.clone()
        else:
            # Restore state for queries in this batch from the saved previous frame state
            prev_dict = saved_last_frame_pred_dict
            prev_query_ids = saved_streaming_query_ids  # Already on CPU

            # Initialize new state tensors on GPU
            updated_pos = torch.zeros((1, N_batch, 2), device=device)
            updated_occlusion = torch.zeros((1, N_batch, 1), device=device)
            updated_certainty = torch.ones((1, N_batch, 1), device=device) * 100
            updated_velocity = torch.zeros((1, N_batch, 2), device=device)
            mconf_logits_coarse = torch.ones((1, N_batch, 1), device=device) * 10

            # Map previous queries to current batch queries
            for curr_idx in range(len(batch_query_ids_cpu)):
                curr_qid = batch_query_ids_cpu[curr_idx]
                prev_idx_matches = (prev_query_ids == curr_qid).nonzero(as_tuple=True)[0]
                if len(prev_idx_matches) > 0:
                    # Query existed in previous frame: copy state
                    prev_idx = prev_idx_matches[0].item()
                    if prev_idx < prev_dict['updated_pos'].shape[1]:
                        updated_pos[0, curr_idx] = prev_dict['updated_pos'][0, prev_idx].to(device)
                        updated_occlusion[0, curr_idx] = prev_dict['updated_occlusion'][0, prev_idx].to(device)
                        updated_certainty[0, curr_idx] = prev_dict['updated_certainty'][0, prev_idx].to(device)
                        if 'updated_velocity' in prev_dict:
                            updated_velocity[0, curr_idx] = prev_dict['updated_velocity'][0, prev_idx].to(device)
                        if 'mconf_logits_coarse' in prev_dict:
                            mconf_logits_coarse[0, curr_idx] = prev_dict['mconf_logits_coarse'][0, prev_idx].to(device)
                    else:
                        # Fallback to current position
                        updated_pos[0, curr_idx] = batch_query_positions[0, curr_idx]
                else:
                    # New query: initialize with current position
                    updated_pos[0, curr_idx] = batch_query_positions[0, curr_idx]

            mem_manager.memory['last_frame_pred_dict'] = {
                'updated_pos': updated_pos,
                'updated_occlusion': updated_occlusion,
                'updated_certainty': updated_certainty,
                'updated_velocity': updated_velocity,
                'mconf_logits_coarse': mconf_logits_coarse,
            }
            mem_manager.memory['_streaming_query_ids'] = batch_query_ids_cpu.clone()

    def _merge_batch_memory_states(
        self,
        batch_results: List[Dict],
        all_query_ids: Tensor,
        device: torch.device
    ):
        """
        Merge memory states from all batches into a single state for the next frame.

        After batch processing, each batch has its own memory state. This method
        combines them into a unified state that covers all queries.

        Args:
            batch_results: List of dicts, each with 'query_ids' and 'last_frame_pred_dict'
            all_query_ids: All query IDs [N_total] (on CPU)
            device: Device for output tensors
        """
        if not batch_results:
            return

        mem_manager = self.base_engine.model.mem_manager
        N_total = len(all_query_ids)

        # Ensure all_query_ids is on CPU for mapping
        all_query_ids_cpu = all_query_ids.cpu() if all_query_ids.is_cuda else all_query_ids

        # Initialize merged state tensors on GPU
        merged_pos = torch.zeros((1, N_total, 2), device=device)
        merged_occlusion = torch.zeros((1, N_total, 1), device=device)
        merged_certainty = torch.ones((1, N_total, 1), device=device) * 100
        merged_velocity = torch.zeros((1, N_total, 2), device=device)
        merged_mconf_logits = torch.ones((1, N_total, 1), device=device) * 10

        # Build mapping from query_id to index in all_query_ids
        qid_to_idx = {int(qid.item()): i for i, qid in enumerate(all_query_ids_cpu)}

        # Merge results from each batch
        for batch_result in batch_results:
            batch_query_ids = batch_result['query_ids']
            batch_pred = batch_result['last_frame_pred_dict']

            # Ensure batch_query_ids is on CPU
            batch_query_ids_cpu = batch_query_ids.cpu() if batch_query_ids.is_cuda else batch_query_ids

            for batch_idx in range(len(batch_query_ids_cpu)):
                qid = batch_query_ids_cpu[batch_idx]
                global_idx = qid_to_idx.get(int(qid.item()))
                if global_idx is not None and batch_idx < batch_pred['updated_pos'].shape[1]:
                    merged_pos[0, global_idx] = batch_pred['updated_pos'][0, batch_idx].to(device)
                    merged_occlusion[0, global_idx] = batch_pred['updated_occlusion'][0, batch_idx].to(device)
                    merged_certainty[0, global_idx] = batch_pred['updated_certainty'][0, batch_idx].to(device)
                    if 'updated_velocity' in batch_pred:
                        merged_velocity[0, global_idx] = batch_pred['updated_velocity'][0, batch_idx].to(device)
                    if 'mconf_logits_coarse' in batch_pred:
                        merged_mconf_logits[0, global_idx] = batch_pred['mconf_logits_coarse'][0, batch_idx].to(device)

        # Update memory with merged state
        # _streaming_query_ids should be on CPU for consistency
        mem_manager.memory['last_frame_pred_dict'] = {
            'updated_pos': merged_pos,
            'updated_occlusion': merged_occlusion,
            'updated_certainty': merged_certainty,
            'updated_velocity': merged_velocity,
            'mconf_logits_coarse': merged_mconf_logits,
        }
        mem_manager.memory['_streaming_query_ids'] = all_query_ids_cpu.clone()

    def initialize(self, video_shape: Tuple[int, int, int], initial_queries: Tensor = None):
        """
        Initialize streaming engine with video shape and optional initial queries.

        Args:
            video_shape: (C, H, W) shape of video frames
            initial_queries: Optional [N, 3] tensor with (t, x, y) format.
                           If None, starts with empty queries.
        """
        self.video_shape = video_shape
        C, H, W = video_shape
        device = next(self.base_engine.model.parameters()).device
        
        if initial_queries is not None:
            if initial_queries.dim() == 2:
                initial_queries = initial_queries.unsqueeze(0)  # [1, N, 3]
            self.queries = initial_queries.clone().to(device)
            N = self.queries.shape[1]
            self.query_ids = torch.arange(N, device='cpu')  # Always on CPU
            self.next_query_id = N
        else:
            self.queries = torch.empty((1, 0, 3), device=device)
            self.query_ids = torch.empty((0,), dtype=torch.long, device='cpu')  # Always on CPU
            self.next_query_id = 0
        
        self.current_frame_idx = 0
        self.tracks_history = torch.empty((0, self.queries.shape[1], 2), device='cpu')
        self.visibility_history = torch.empty((0, self.queries.shape[1]), dtype=torch.bool, device='cpu')
        self.is_initialized = True

        # Initialize query ID to index mapping cache (for fast lookup in process_frame)
        self._qid_to_idx_map = None

        # Reset memory manager for fresh start
        self.base_engine.model.mem_manager.reset_all_memory()
        
    def add_queries(self, new_queries: Tensor, frame_idx: int = None):
        """
        Add new query points to track. New queries start tracking from current frame.
        
        Args:
            new_queries: [N_new, 2] tensor with (x, y) coordinates in original video space
            frame_idx: Optional frame index. If None, uses current_frame_idx.
        
        Returns:
            query_ids: Tensor of assigned query IDs for the new queries
        """
        if not self.is_initialized:
            raise RuntimeError("StreamingEngine not initialized. Call initialize() first.")
        
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        
        device = self.queries.device
        C, H, W = self.video_shape
        
        # Convert to [N_new, 3] format: (t, x, y)
        if new_queries.dim() == 1:
            new_queries = new_queries.unsqueeze(0)  # [1, 2] or [1, 3]
        
        # Handle different input formats
        if new_queries.dim() == 2:
            if new_queries.shape[1] == 2:
                # Add frame index as first dimension
                t_coords = torch.full(
                    (new_queries.shape[0], 1), 
                    frame_idx, 
                    device=new_queries.device, 
                    dtype=new_queries.dtype
                )
                new_queries = torch.cat([t_coords, new_queries], dim=1)  # [N_new, 3]
            # new_queries is now [N_new, 3]
            new_queries = new_queries.unsqueeze(0).to(device)  # [1, N_new, 3]
        elif new_queries.dim() == 3:
            # Already in [1, N_new, 3] format
            if new_queries.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {new_queries.shape[0]}")
            new_queries = new_queries.to(device)
            # Update frame indices if needed
            if new_queries.shape[2] == 3:
                new_queries[0, :, 0] = frame_idx
        else:
            raise ValueError(f"Invalid new_queries shape: {new_queries.shape}")
        N_new = new_queries.shape[1]
        
        # Assign new query IDs (always on CPU since query_ids is on CPU)
        new_ids = torch.arange(
            self.next_query_id,
            self.next_query_id + N_new,
            device='cpu',
            dtype=torch.long
        )
        self.next_query_id += N_new

        # Concatenate with existing queries
        self.queries = torch.cat([self.queries, new_queries], dim=1)  # [1, N_old + N_new, 3]
        self.query_ids = torch.cat([self.query_ids, new_ids], dim=0)  # [N_old + N_new]

        # Invalidate query ID to index cache (will be rebuilt in process_frame)
        self._qid_to_idx_map = None

        # Extend history tensors with zeros for new queries
        if self.tracks_history.shape[0] > 0:
            T_hist, N_old, _ = self.tracks_history.shape
            new_tracks = torch.zeros((T_hist, N_new, 2), device='cpu')
            new_vis = torch.zeros((T_hist, N_new), dtype=torch.bool, device='cpu')
            self.tracks_history = torch.cat([self.tracks_history, new_tracks], dim=1)
            self.visibility_history = torch.cat([self.visibility_history, new_vis], dim=1)

        return new_ids
    
    @torch.no_grad()
    def process_frame(self, frame: Tensor, use_aug: bool = False):
        """
        Process a single frame and update tracking results.
        
        Args:
            frame: [C, H, W] tensor of current frame
            use_aug: Whether to use augmentation
        
        Returns:
            dict with 'tracks' and 'visibility' for current frame
        """
        if not self.is_initialized:
            raise RuntimeError("StreamingEngine not initialized. Call initialize() first.")
        
        device = next(self.base_engine.model.parameters()).device
        C, H, W = self.video_shape
        
        # Prepare current frame for processing
        frame_gpu = frame.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]
        frame_processed = self.base_engine._preprocess_frame_chunk(frame_gpu, device)
        frame_processed = frame_processed[0, 0]  # [C, H_interp, W_interp]
        
        # Update frame cache (keep last 2 frames for triplet)
        # Move to CPU to save GPU memory
        self.frame_cache.append(frame_processed.cpu())
        if len(self.frame_cache) > 2:
            self.frame_cache.pop(0)

        # Save first frame (anchor) for coarse/temporal modules (on CPU to save memory)
        if self.current_frame_idx == 0:
            self._first_frame_image = frame_processed.cpu()

        # Get all queries that should be tracked at current frame
        # For streaming: track all queries that were added at or before current frame
        active_mask = self.queries[0, :, 0] <= self.current_frame_idx

        if active_mask.sum() == 0:
            # No active queries, return empty results
            N = self.queries.shape[1]
            tracks_full = torch.zeros((N, 2), device='cpu')
            visibility_full = torch.zeros((N,), dtype=torch.bool, device='cpu')

            # Update history
            self.tracks_history = torch.cat([self.tracks_history, tracks_full.unsqueeze(0)], dim=0)
            self.visibility_history = torch.cat([self.visibility_history, visibility_full.unsqueeze(0)], dim=0)
            self.current_frame_idx += 1

            return {
                'tracks': tracks_full,
                'visibility': visibility_full
            }

        # Special case: For query frame (frame 0), directly return query positions with visibility=True
        # No need to run the model since query positions are known at the query frame
        if self.current_frame_idx == 0:
            N = self.queries.shape[1]
            tracks_full = self.queries[0, :, 1:3].cpu().clone()  # (x, y) coordinates
            visibility_full = torch.ones((N,), dtype=torch.bool, device='cpu')

            # Update history
            self.tracks_history = torch.cat([self.tracks_history, tracks_full.unsqueeze(0)], dim=0)
            self.visibility_history = torch.cat([self.visibility_history, visibility_full.unsqueeze(0)], dim=0)
            self.current_frame_idx += 1

            return {
                'tracks': tracks_full,
                'visibility': visibility_full
            }

        active_queries = self.queries[0, active_mask].unsqueeze(0)  # [1, N_active, 3]
        active_query_ids = self.query_ids[active_mask.cpu()]  # query_ids is on CPU
        
        # Separate queries: from frame 0 (can do coarse matching) vs later frames (skip coarse)
        first_frame_mask = active_queries[0, :, 0] == 0
        later_frame_mask = ~first_frame_mask
        
        first_frame_queries = active_queries[0, first_frame_mask]  # [N_first, 3]
        later_frame_queries = active_queries[0, later_frame_mask]  # [N_later, 3]
        first_frame_ids = active_query_ids[first_frame_mask.cpu()]  # active_query_ids is on CPU
        later_frame_ids = active_query_ids[later_frame_mask.cpu()]  # active_query_ids is on CPU

        # Get current positions for active queries (in interp space)
        query_positions = torch.zeros((1, active_queries.shape[1], 2), device=device)

        # Vectorized position setup (instead of O(N^2) loop)
        # Scale factors for coordinate conversion
        scale_x = (self.interp_shape[1] - 1) / (W - 1)
        scale_y = (self.interp_shape[0] - 1) / (H - 1)

        # Get query frame indices
        q_frames = active_queries[0, :, 0]  # [N_active]

        # Mask for different query types:
        # - first_frame_mask_pos: queries that started from frame 0 (always use initial position)
        # - current_frame_mask_pos: queries added at current frame (use initial position)
        # - other_frame_mask_pos: queries from intermediate frames (use tracked history position)
        #
        # IMPORTANT: Queries from frame 0 should ALWAYS use initial position to avoid drift!
        # The model uses temporal memory to track, not the query position itself.
        first_frame_mask_pos = (q_frames == 0)
        current_frame_mask_pos = (q_frames == self.current_frame_idx)
        other_frame_mask_pos = ~(first_frame_mask_pos | current_frame_mask_pos)

        # For first frame queries and current frame queries: use initial position
        init_mask = first_frame_mask_pos | current_frame_mask_pos
        if init_mask.any():
            query_positions[0, init_mask, 0] = active_queries[0, init_mask, 1] * scale_x
            query_positions[0, init_mask, 1] = active_queries[0, init_mask, 2] * scale_y

        # For other queries (from intermediate frames): use last tracked position from history
        if other_frame_mask_pos.any() and self.tracks_history.shape[0] > 0:
            # Build a mapping from query_id to global index (do this once, cached)
            if not hasattr(self, '_qid_to_idx_map') or self._qid_to_idx_map is None:
                self._qid_to_idx_map = {int(qid.item()): i for i, qid in enumerate(self.query_ids)}

            # Get global indices for active queries that need history lookup
            other_indices = other_frame_mask_pos.nonzero(as_tuple=True)[0].cpu()  # Move to CPU for indexing
            for local_idx in other_indices:
                local_idx_int = int(local_idx.item())
                qid = int(active_query_ids[local_idx_int].item())
                global_idx = self._qid_to_idx_map.get(qid)
                if global_idx is not None and global_idx < self.tracks_history.shape[1]:
                    last_pos = self.tracks_history[-1, global_idx]  # [2] in original space
                    query_positions[0, local_idx_int, 0] = last_pos[0] * scale_x
                    query_positions[0, local_idx_int, 1] = last_pos[1] * scale_y
                else:
                    # Fallback to initial position if history not available
                    query_positions[0, local_idx_int, 0] = active_queries[0, local_idx_int, 1] * scale_x
                    query_positions[0, local_idx_int, 1] = active_queries[0, local_idx_int, 2] * scale_y

        # Store which queries are from first frame (for coarse matching)
        # This will be used to filter queries in coarse stage
        self._first_frame_query_mask = first_frame_mask.clone()
        
        # Construct triplet: need 3 frames (image0, image_m, image1)
        # Always use the first frame as anchor (image0) to mimic original video_forward behavior
        # image1: current frame; image_m: use previous processed frame if available, else first frame
        # Note: frame_cache and _first_frame_image are on CPU to save memory, move to GPU here
        image0 = getattr(self, "_first_frame_image", frame_processed.cpu()).to(device)
        if len(self.frame_cache) >= 2:
            image_m = self.frame_cache[-2].to(device)  # previous processed frame
        else:
            image_m = image0
        image1 = frame_processed  # already on GPU

        N_active = active_queries.shape[1]

        # ==================================================
        # [Query Batching] DISABLED for StreamingEngine
        # The batch processing logic has complex memory state management that
        # causes tracking drift issues. For StreamingEngine, process all queries
        # at once. If OOM occurs, reduce max_points or increase GPU memory.
        # ==================================================
        if False and N_active > self.query_batch_size:  # Disabled
            # Process in batches to prevent OOM
            num_batches = (N_active + self.query_batch_size - 1) // self.query_batch_size

            all_traj_e = []
            all_pred_visibles = []

            # Save the memory state from last frame BEFORE batch processing
            # This is critical: we need to preserve the previous frame's state for all queries
            mem_manager = self.base_engine.model.mem_manager
            saved_streaming_query_ids = mem_manager.memory.get('_streaming_query_ids', None)
            saved_last_frame_pred_dict = mem_manager.memory.get('last_frame_pred_dict', None)
            if saved_streaming_query_ids is not None:
                # Move to CPU for consistent comparison with batch_active_query_ids
                saved_streaming_query_ids = saved_streaming_query_ids.clone().cpu()
            if saved_last_frame_pred_dict is not None:
                # Deep copy and move to CPU to avoid GPU memory issues
                saved_last_frame_pred_dict = {k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                                               for k, v in saved_last_frame_pred_dict.items()}

            # Collect results for each batch to merge at the end
            batch_results = []

            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.query_batch_size
                batch_end = min(batch_start + self.query_batch_size, N_active)

                # Slice queries for this batch
                batch_query_positions = query_positions[:, batch_start:batch_end, :]  # [1, batch_size, 2]
                batch_first_frame_mask = self._first_frame_query_mask[batch_start:batch_end]
                batch_active_queries = active_queries[:, batch_start:batch_end, :]
                batch_active_query_ids = active_query_ids[batch_start:batch_end]

                # Reset memory manager for each batch
                mem_manager.reset_all_memory()

                # Restore the saved memory state for this batch's queries
                # This ensures we use the correct previous frame positions
                self._restore_memory_for_batch(
                    saved_streaming_query_ids,
                    saved_last_frame_pred_dict,
                    batch_active_query_ids,
                    batch_query_positions,
                    device
                )

                # Construct triplet data for this batch
                batch_triplet_data = {
                    'image0': image0.unsqueeze(0),
                    'image_m': image_m.unsqueeze(0),
                    'image1': image1.unsqueeze(0),
                    'queries': batch_query_positions,
                    'first_frame_query_mask': batch_first_frame_mask.to(device)
                }

                # Forward pass for this batch
                self.base_engine.model(batch_triplet_data)

                # Extract and store results
                batch_traj = batch_triplet_data['mkpts1_f'].detach()
                batch_vis = batch_triplet_data['pred_visibles'].detach()

                # Also save the batch's memory state for later merging
                batch_last_pred = mem_manager.memory.get('last_frame_pred_dict', None)
                if batch_last_pred is not None:
                    batch_results.append({
                        'query_ids': batch_active_query_ids.clone(),
                        'last_frame_pred_dict': {k: v.clone() if isinstance(v, torch.Tensor) else v
                                                  for k, v in batch_last_pred.items()}
                    })

                # Normalize shapes
                if batch_traj.dim() == 3:
                    batch_traj = batch_traj.squeeze(1) if batch_traj.shape[1] == 1 else batch_traj[:, -1, :]
                if batch_vis.dim() == 3:
                    batch_vis = batch_vis.squeeze(1) if batch_vis.shape[1] == 1 else batch_vis[:, -1, :]
                if batch_vis.dim() == 2:
                    batch_vis = batch_vis.squeeze(-1)

                all_traj_e.append(batch_traj)
                all_pred_visibles.append(batch_vis)

                del batch_triplet_data

            # Concatenate batch results
            traj_e = torch.cat(all_traj_e, dim=0)  # [N_active, 2]
            pred_visibles = torch.cat(all_pred_visibles, dim=0)  # [N_active]

            # Merge all batch memory states into a single memory state for next frame
            # This ensures all queries have their state preserved for the next frame
            self._merge_batch_memory_states(batch_results, active_query_ids, device)

            # Skip the non-batched path below
            N_active_result = traj_e.shape[0]

        else:
            # Original path for small N_active
            # Update memory manager state for new queries if needed
            # This handles the case when queries are dynamically added
            self._update_memory_for_new_queries(active_queries, active_query_ids, query_positions)

            # Construct triplet data
            # Add mask to indicate which queries are from first frame (can do coarse matching)
            triplet_data = {
                'image0': image0.unsqueeze(0),  # [1, C, H, W]
                'image_m': image_m.unsqueeze(0),
                'image1': image1.unsqueeze(0),
                'queries': query_positions,  # [1, N_active, 2] (x, y) in interp_shape space
                'first_frame_query_mask': self._first_frame_query_mask.to(device)  # [N_active] bool mask
            }

            # Forward pass
            self.base_engine.model(triplet_data)

            # Extract results
            traj_e = triplet_data['mkpts1_f']  # Shape: [BN, F, 2] or [BN, 2] or [1, N_active, 2]
            pred_visibles = triplet_data['pred_visibles']  # Shape: [BN, F, 1] or [BN, 1] or [1, N_active, 1]

            # Normalize traj_e to [N_active, 2] format
            if traj_e.dim() == 3:
                # [BN, F, 2] or [1, N_active, 2]
                if traj_e.shape[1] == 1:
                    traj_e = traj_e.squeeze(1)  # [BN, 2] or [1, N_active, 2]
                else:
                    traj_e = traj_e[:, -1, :]  # [BN, F, 2] -> [BN, 2], take last frame

            # Ensure traj_e is 2D: [N_active, 2]
            if traj_e.dim() == 2:
                # [BN, 2] or [1, N_active, 2]
                if traj_e.shape[0] == 1 and traj_e.shape[1] == 2:
                    # Single query case: [1, 2] -> keep as is
                    pass
                # Otherwise it's already [N_active, 2] or [BN, 2]
            elif traj_e.dim() == 1:
                traj_e = traj_e.unsqueeze(0)  # [2] -> [1, 2]

            # Handle pred_visibles shape - normalize to [N_active] format
            if pred_visibles.dim() == 3:
                # [BN, F, 1] or [1, N_active, 1]
                if pred_visibles.shape[1] == 1:
                    pred_visibles = pred_visibles.squeeze(1)  # [BN, 1] or [1, N_active, 1]
                else:
                    pred_visibles = pred_visibles[:, -1, :]  # [BN, F, 1] -> [BN, 1]

            if pred_visibles.dim() == 2:
                pred_visibles = pred_visibles.squeeze(-1)  # [BN, 1] -> [BN] or [1, N_active, 1] -> [1, N_active]
                if pred_visibles.dim() == 1 and pred_visibles.shape[0] == 1:
                    # [1] -> keep as is for single query
                    pass

            # Ensure shapes match: traj_e [N_active, 2], pred_visibles [N_active]
            N_active_result = traj_e.shape[0]
            if pred_visibles.dim() == 0:
                pred_visibles = pred_visibles.unsqueeze(0)
            if pred_visibles.shape[0] != N_active_result:
                # Reshape to match
                if pred_visibles.shape[0] == 1 and N_active_result > 1:
                    pred_visibles = pred_visibles.expand(N_active_result)
                elif pred_visibles.shape[0] > N_active_result:
                    pred_visibles = pred_visibles[:N_active_result]

        # Denormalize coordinates back to original space
        traj_e = traj_e.cpu()
        # traj_e is [N_active, 2]
        traj_e[:, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        
        # Create full result tensor for all queries
        N_total = self.queries.shape[1]
        tracks_full = torch.zeros((N_total, 2), device='cpu')
        visibility_full = torch.zeros((N_total,), dtype=torch.bool, device='cpu')
        
        # Map active query results to full tensor
        for local_idx, qid in enumerate(active_query_ids):
            if local_idx >= N_active_result:
                break
            global_idx = (self.query_ids == qid).nonzero(as_tuple=True)[0][0]
            tracks_full[global_idx] = traj_e[local_idx]  # traj_e is [N_active, 2]
            if local_idx < pred_visibles.shape[0]:
                visibility_full[global_idx] = pred_visibles[local_idx].bool()  # pred_visibles is [N_active]
        
        # Update history
        self.tracks_history = torch.cat([self.tracks_history, tracks_full.unsqueeze(0)], dim=0)
        self.visibility_history = torch.cat([self.visibility_history, visibility_full.unsqueeze(0)], dim=0)

        # Update current frame index
        self.current_frame_idx += 1

        # Clean up GPU memory
        del frame_gpu
        torch.cuda.empty_cache()

        return {
            'tracks': tracks_full,  # [N_total, 2]
            'visibility': visibility_full  # [N_total]
        }
    
    def get_tracks(self, query_ids: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Get tracking history for specified queries or all queries.
        
        Args:
            query_ids: Optional tensor of query IDs. If None, returns all.
        
        Returns:
            tracks: [T, N, 2] trajectory history
            visibility: [T, N] visibility history
        """
        if not self.is_initialized:
            raise RuntimeError("StreamingEngine not initialized.")
        
        if query_ids is None:
            return self.tracks_history, self.visibility_history
        
        # Filter by query IDs
        mask = torch.isin(self.query_ids, query_ids)
        return self.tracks_history[:, mask], self.visibility_history[:, mask]
    
    @torch.no_grad()
    def video_forward(self, video: Tensor, queries: Tensor, use_aug: bool = False):
        """
        Offline video forward pass using streaming approach.
        Leverages streaming capability to process all frames in one pass,
        supporting queries with different starting frame indices.
        
        Args:
            video: [B, T, C, H, W] video tensor
            queries: [B, N, 3] queries tensor with (t, x, y) format, where t is frame index
            use_aug: Whether to use augmentation
        
        Returns:
            traj_e: [B, T, N, 2] trajectory predictions
            vis_e: [B, T, N] visibility predictions
        """
        video = video.cpu()
        B, T, C, H, W = video.shape
        B, N_orig, D = queries.shape
        assert D == 3, f"queries should have 3 channels (t, x, y), got {D}"
        
        device = next(self.base_engine.model.parameters()).device
        
        # Initialize streaming engine with all queries at once
        # This allows queries to start from different frames
        self.initialize(video_shape=(C, H, W), initial_queries=queries[0])
        
        # Initialize output buffers
        tracks = torch.zeros((B, T, N_orig, 2), dtype=torch.float32, device='cpu')
        occlusion_pred = torch.zeros((B, T, N_orig), dtype=torch.bool, device='cpu')
        
        # Process video frame by frame using streaming approach
        desc_base = f"Processing Frames [Streaming Mode | Queries: {N_orig}]"
        progress = get_progress("{task.description}")
        task = progress.add_task(desc_base, total=T)
        progress.start()

        for frame_idx in range(T):
            frame = video[0, frame_idx]  # [C, H, W]
            
            # Process frame using streaming engine
            result = self.process_frame(frame, use_aug=use_aug)
            
            # Get current tracks and visibility
            frame_tracks = result['tracks']  # [N_total, 2]
            frame_visibility = result['visibility']  # [N_total]
            
            # Write to output buffers (only original queries, not padded ones)
            tracks[0, frame_idx, :] = frame_tracks[:N_orig]
            occlusion_pred[0, frame_idx, :] = frame_visibility[:N_orig]
            
            # Update progress
            active_queries = (self.queries[0, :, 0] <= frame_idx).sum().item()
            progress.update(
                task,
                description=f"{desc_base} | Active Queries: {active_queries}/{N_orig}",
            )
            progress.advance(task)

        progress.stop()
        
        self.reset()
        
        return tracks, occlusion_pred
    
    def reset(self):
        """Reset streaming engine state."""
        self.current_frame_idx = 0
        self.queries = None
        self.query_ids = None
        self.tracks_history = None
        self.visibility_history = None
        self.next_query_id = 0
        self.is_initialized = False
        self.frame_cache = []
        # Also reset first frame image and query mask
        self._first_frame_image = None
        self._first_frame_query_mask = None
        self._qid_to_idx_map = None
        self.base_engine.model.mem_manager.reset_all_memory()

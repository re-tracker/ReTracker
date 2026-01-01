"""Multi-GPU streaming inference engine implementation (MultiGPUStreamingEngine).

The stable import path is still `retracker.inference.engine`.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from retracker.inference.engines.streaming import StreamingEngine
from retracker.utils.rich_utils import CONSOLE


class MultiGPUStreamingEngine(torch.nn.Module):
    """
    Multi-GPU streaming engine that distributes queries across multiple GPUs.
    Uses parallel processing to speed up inference and reduce per-GPU memory usage.
    """

    def __init__(
        self,
        devices: List[str],
        ckpt_path: str,
        interp_shape: Tuple = (256, 256),
        enable_highres_inference: bool = False,
        coarse_resolution: Tuple[int, int] = (512, 512),
        query_batch_size: int = 256,
        fast_start: bool = False,
    ) -> None:
        """
        Initialize multi-GPU streaming engine.

        Args:
            devices: List of device strings, e.g., ['cuda:0', 'cuda:1']
            ckpt_path: Path to model checkpoint
            interp_shape: Interpolation shape for inference
            query_batch_size: Batch size per GPU (total batch = query_batch_size * num_gpus)
        """
        super(MultiGPUStreamingEngine, self).__init__()

        if len(devices) < 2:
            raise ValueError(f"MultiGPUStreamingEngine requires at least 2 devices, got {len(devices)}")

        self.devices = devices
        self.num_gpus = len(devices)
        self.interp_shape = interp_shape
        self.query_batch_size = query_batch_size

        CONSOLE.print(f"[MultiGPU] Initializing {self.num_gpus} GPU engines on {devices}")

        # Create a streaming engine for each GPU
        self.engines: List[StreamingEngine] = []
        for i, device in enumerate(devices):
            CONSOLE.print(f"[MultiGPU] Loading model on {device}...")
            engine = StreamingEngine(
                ckpt_path=ckpt_path,
                interp_shape=interp_shape,
                enable_highres_inference=enable_highres_inference,
                coarse_resolution=coarse_resolution,
                query_batch_size=query_batch_size,
                fast_start=fast_start,
            )
            engine.to(device)
            engine.eval()
            self.engines.append(engine)

        CONSOLE.print(f"[MultiGPU] All {self.num_gpus} engines initialized")

        # Warmup each GPU to initialize all lazy CUDA operations
        # This prevents "lazy wrapper should be called at most once" error
        self._warmup_engines()

        # Shared state
        self.is_initialized = False
        self.video_shape: Optional[Tuple[int, int, int]] = None
        self.queries: Optional[Tensor] = None
        self.query_ids: Optional[Tensor] = None
        self.current_frame_idx: int = 0
        self.tracks_history: Optional[Tensor] = None
        self.visibility_history: Optional[Tensor] = None
        self.next_query_id: int = 0

        # GPU assignment for each query
        self._query_to_gpu: dict = {}  # query_id -> gpu_idx

        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.num_gpus)

    def initialize(
        self,
        video_shape: Tuple[int, int, int],
        initial_queries: Tensor = None
    ) -> None:
        """
        Initialize streaming state.

        Args:
            video_shape: (C, H, W) shape of video frames
            initial_queries: Optional initial query points [N, 3] with (t, x, y)
        """
        self.video_shape = video_shape
        self.current_frame_idx = 0
        self._query_to_gpu = {}

        # Initialize all engines
        for engine in self.engines:
            engine.initialize(video_shape, None)  # Don't pass queries yet

        if initial_queries is not None:
            self._distribute_queries(initial_queries)

        # Initialize history
        N = initial_queries.shape[0] if initial_queries is not None else 0
        self.tracks_history = torch.zeros((0, N, 2), device='cpu')
        self.visibility_history = torch.zeros((0, N), dtype=torch.bool, device='cpu')
        self.queries = initial_queries.unsqueeze(0) if initial_queries is not None else None
        self.query_ids = torch.arange(N) if N > 0 else torch.tensor([], dtype=torch.long)
        self.next_query_id = N

        self.is_initialized = True
        CONSOLE.print(f"[MultiGPU] Initialized with {N} queries distributed across {self.num_gpus} GPUs")

    def _distribute_queries(self, queries: Tensor) -> None:
        """Distribute queries across GPUs using round-robin."""
        N = queries.shape[0]
        queries_per_gpu = [[] for _ in range(self.num_gpus)]
        query_ids_per_gpu = [[] for _ in range(self.num_gpus)]

        for i in range(N):
            gpu_idx = i % self.num_gpus
            queries_per_gpu[gpu_idx].append(queries[i])
            query_ids_per_gpu[gpu_idx].append(i)
            self._query_to_gpu[i] = gpu_idx

        # Add queries to each engine
        for gpu_idx, (engine, q_list, id_list) in enumerate(zip(self.engines, queries_per_gpu, query_ids_per_gpu)):
            if len(q_list) > 0:
                q_tensor = torch.stack(q_list, dim=0)  # [N_gpu, 3]
                # Initialize engine with its subset of queries
                engine.queries = q_tensor.unsqueeze(0).to(engine.base_engine.model.parameters().__next__().device)
                engine.query_ids = torch.tensor(id_list, dtype=torch.long)
                engine.next_query_id = max(id_list) + 1 if id_list else 0

                # Initialize history for this subset
                N_gpu = len(q_list)
                device = engine.base_engine.model.parameters().__next__().device
                engine.tracks_history = torch.zeros((0, N_gpu, 2), device='cpu')
                engine.visibility_history = torch.zeros((0, N_gpu), dtype=torch.bool, device='cpu')

    def _warmup_engines(self) -> None:
        """
        Warmup all GPU engines to initialize lazy CUDA operations.

        This prevents "lazy wrapper should be called at most once" error
        when running inference in parallel threads.
        """
        CONSOLE.print(f"[MultiGPU] Warming up {self.num_gpus} GPU engines...")

        for i, engine in enumerate(self.engines):
            device = self.devices[i]
            CONSOLE.print(f"[MultiGPU] Warming up engine on {device}...")

            # Run a dummy forward pass to initialize all lazy operations
            # Use the same shape as interp_shape for consistency
            H, W = self.interp_shape
            dummy_video = torch.randn(1, 3, 3, H, W, device=device)
            dummy_queries = torch.tensor([[[0, W//2, H//2]]], dtype=torch.float32, device=device)

            try:
                with torch.no_grad():
                    # Run video_forward to warmup all model components including linalg.inv
                    engine.base_engine.video_forward(dummy_video, dummy_queries)
            except Exception as e:
                CONSOLE.print(f"[MultiGPU] Warmup on {device} completed with minor issues (expected): {type(e).__name__}")

            # Clear CUDA cache after warmup
            if 'cuda' in device:
                torch.cuda.empty_cache()

        CONSOLE.print(f"[MultiGPU] All engines warmed up")

    def _process_frame_on_gpu(self, gpu_idx: int, frame: Tensor) -> dict:
        """Process frame on a single GPU."""
        engine = self.engines[gpu_idx]
        if engine.queries is None or engine.queries.shape[1] == 0:
            return {'tracks': None, 'visibility': None, 'query_ids': None}

        result = engine.process_frame(frame, use_aug=False)
        return {
            'tracks': result['tracks'],
            'visibility': result['visibility'],
            'query_ids': engine.query_ids.clone()
        }

    @torch.no_grad()
    def process_frame(self, frame: Tensor, use_aug: bool = False) -> dict:
        """
        Process a single frame using multiple GPUs in parallel.

        Args:
            frame: Frame tensor [C, H, W]
            use_aug: Whether to use augmentation (unused, for API compatibility)

        Returns:
            dict with 'tracks' and 'visibility' for current frame
        """
        if not self.is_initialized:
            raise RuntimeError("MultiGPUStreamingEngine not initialized.")

        # Handle frame 0 specially - need to initialize each engine's internal state
        if self.current_frame_idx == 0:
            N = self.queries.shape[1] if self.queries is not None else 0
            tracks_full = self.queries[0, :, 1:3].cpu().clone() if N > 0 else torch.zeros((0, 2))
            visibility_full = torch.ones((N,), dtype=torch.bool)

            self.tracks_history = torch.cat([self.tracks_history, tracks_full.unsqueeze(0)], dim=0)
            self.visibility_history = torch.cat([self.visibility_history, visibility_full.unsqueeze(0)], dim=0)

            # IMPORTANT: Process frame 0 through each engine to initialize internal state
            # This sets up _first_frame_image, frame_cache, and memory manager
            for engine in self.engines:
                if engine.queries is not None and engine.queries.shape[1] > 0:
                    device = next(engine.base_engine.model.parameters()).device
                    frame_gpu = frame.to(device)
                    # Call process_frame to properly initialize engine state
                    engine.process_frame(frame_gpu, use_aug=False)

            self.current_frame_idx = 1
            return {'tracks': tracks_full, 'visibility': visibility_full}

        # Submit frame processing to all GPUs in parallel
        futures = []
        for gpu_idx in range(self.num_gpus):
            future = self._executor.submit(self._process_frame_on_gpu, gpu_idx, frame)
            futures.append(future)

        # Gather results
        results = [f.result() for f in futures]

        # Merge results into full tensors
        N_total = self.queries.shape[1] if self.queries is not None else 0
        tracks_full = torch.zeros((N_total, 2), device='cpu')
        visibility_full = torch.zeros((N_total,), dtype=torch.bool, device='cpu')

        for result in results:
            if result['tracks'] is not None and result['query_ids'] is not None:
                for local_idx, qid in enumerate(result['query_ids']):
                    global_idx = int(qid.item())
                    if global_idx < N_total and local_idx < result['tracks'].shape[0]:
                        tracks_full[global_idx] = result['tracks'][local_idx]
                        visibility_full[global_idx] = result['visibility'][local_idx]

        # Update history
        self.tracks_history = torch.cat([self.tracks_history, tracks_full.unsqueeze(0)], dim=0)
        self.visibility_history = torch.cat([self.visibility_history, visibility_full.unsqueeze(0)], dim=0)
        self.current_frame_idx += 1

        return {'tracks': tracks_full, 'visibility': visibility_full}

    def add_queries(self, points: Tensor, frame_idx: int = None) -> Tensor:
        """
        Add new query points dynamically during tracking.

        Args:
            points: New query points [N_new, 2] with (x, y) coordinates
            frame_idx: Frame index for new queries (default: current frame)

        Returns:
            Tensor of assigned query IDs
        """
        if frame_idx is None:
            frame_idx = self.current_frame_idx

        N_new = points.shape[0]
        new_ids = torch.arange(self.next_query_id, self.next_query_id + N_new, dtype=torch.long)

        # Distribute new queries across GPUs
        for i, qid in enumerate(new_ids):
            gpu_idx = int(qid.item()) % self.num_gpus
            self._query_to_gpu[int(qid.item())] = gpu_idx

            # Add to corresponding engine
            engine = self.engines[gpu_idx]
            point = points[i:i+1]  # [1, 2]
            engine.add_queries(point, frame_idx=frame_idx)

        # Update global state
        t_col = torch.full((N_new, 1), frame_idx, dtype=points.dtype)
        new_queries = torch.cat([t_col, points], dim=1)  # [N_new, 3]

        if self.queries is None:
            self.queries = new_queries.unsqueeze(0)
            self.query_ids = new_ids
        else:
            self.queries = torch.cat([self.queries, new_queries.unsqueeze(0)], dim=1)
            self.query_ids = torch.cat([self.query_ids, new_ids])

        # Expand history
        T = self.tracks_history.shape[0]
        new_tracks = torch.zeros((T, N_new, 2), device='cpu')
        new_vis = torch.zeros((T, N_new), dtype=torch.bool, device='cpu')
        self.tracks_history = torch.cat([self.tracks_history, new_tracks], dim=1)
        self.visibility_history = torch.cat([self.visibility_history, new_vis], dim=1)

        self.next_query_id += N_new
        return new_ids

    def get_tracks(self, query_ids: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Get tracking history."""
        if query_ids is None:
            return self.tracks_history, self.visibility_history

        mask = torch.isin(self.query_ids, query_ids)
        return self.tracks_history[:, mask], self.visibility_history[:, mask]

    def reset(self) -> None:
        """Reset all engines and clear state."""
        for engine in self.engines:
            engine.reset()
        self.is_initialized = False
        self.current_frame_idx = 0
        self._query_to_gpu = {}
        # Also reset shared state
        self.queries = None
        self.query_ids = None
        self.tracks_history = None
        self.visibility_history = None
        self.next_query_id = 0

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

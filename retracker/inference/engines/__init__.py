"""Internal inference engine implementations.

The stable, public import path is :mod:`retracker.inference.engine`.

We keep this package import-safe even when only a subset of engines is used by
loading implementations lazily.
"""

__all__ = ["ReTrackerEngine", "StreamingEngine", "MultiGPUStreamingEngine"]


def __getattr__(name: str):  # pragma: no cover
    if name == "ReTrackerEngine":
        from retracker.inference.engines.offline import ReTrackerEngine

        return ReTrackerEngine
    if name == "StreamingEngine":
        from retracker.inference.engines.streaming import StreamingEngine

        return StreamingEngine
    if name == "MultiGPUStreamingEngine":
        from retracker.inference.engines.streaming_multi_gpu import MultiGPUStreamingEngine

        return MultiGPUStreamingEngine
    raise AttributeError(name)

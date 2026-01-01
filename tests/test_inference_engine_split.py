import sys


def test_import_retracker_inference_does_not_import_lightning():
    # Inference imports must stay dependency-light.
    assert "lightning" not in sys.modules
    import retracker.inference  # noqa: F401
    assert "lightning" not in sys.modules


def test_engine_classes_come_from_split_modules():
    # `retracker.inference.engine` is expected to be a thin facade.
    # Implementations should live under `retracker.inference.engines.*`.
    from retracker.inference.engine import MultiGPUStreamingEngine, ReTrackerEngine, StreamingEngine

    assert ReTrackerEngine.__module__ == "retracker.inference.engines.offline"
    assert StreamingEngine.__module__ == "retracker.inference.engines.streaming"
    assert MultiGPUStreamingEngine.__module__ == "retracker.inference.engines.streaming_multi_gpu"


def test_facade_re_exports_same_objects():
    from retracker.inference.engine import ReTrackerEngine as E1

    from retracker.inference.engines.offline import ReTrackerEngine as E2

    assert E1 is E2

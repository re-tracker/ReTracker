import numpy as np


def test_frame_buffer_eviction_and_order():
    from retracker.apps.slam.frame_buffer import FrameBuffer

    fb = FrameBuffer(capacity=3)
    fb.add(0, "f0")
    fb.add(1, "f1")
    fb.add(2, "f2")
    fb.add(3, "f3")
    assert list(fb.frames.keys()) == [1, 2, 3]
    assert fb.get_future_frames(1, 2) == ["f2", "f3"]



def test_tracklet_manager_retire_on_lost():
    from retracker.apps.slam.tracklet import Tracklet, TrackletManager

    mgr = TrackletManager(max_lost=2)
    tr = Tracklet(
        id=1,
        anchor_kf_id=0,
        anchor_position=np.array([10.0, 20.0]),
        inverse_depth=1.0,
        inverse_depth_var=1.0,
        predictions={1: (np.array([11.0, 20.5]), 1.0, 0.0)},
        status="active"
    )
    mgr.tracklets[1] = tr
    mgr.update_for_frame(5)
    mgr.update_for_frame(6)
    mgr.update_for_frame(7)
    assert 1 not in mgr.tracklets

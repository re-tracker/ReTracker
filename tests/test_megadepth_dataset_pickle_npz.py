from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def test_megadepth_dataset_accepts_pickled_scene_info(tmp_path: Path):
    """MegaDepth indices in this repo may be pickled dicts with a `.npz` suffix.

    numpy.load() returns a Python dict in that case (not an NpzFile), so the
    dataset loader must handle both formats.
    """
    from retracker.data.datasets.Megadepth import MegaDepthDataset

    scene_info = {
        # Minimal pair_infos format used by the dataset: ((idx0, idx1), overlap_score)
        "pair_infos": [((0, 1), 0.9)],
        "dataset_name": "megadepth",
        "image_paths": ["a.png", "b.png"],
        "depth_paths": ["depths/a.h5", "depths/b.h5"],
        "intrinsics": np.stack([np.eye(3), np.eye(3)], axis=0),
        "poses": np.stack([np.eye(4), np.eye(4)], axis=0),
    }

    fake_npz = tmp_path / "scene_info_fake.npz"
    with fake_npz.open("wb") as f:
        pickle.dump(scene_info, f)

    # Should not crash during __init__.
    ds = MegaDepthDataset(
        root_dir=str(tmp_path),
        npz_path=str(fake_npz),
        mode="test",
        min_overlap_score=0.0,
    )
    assert len(ds) == 1


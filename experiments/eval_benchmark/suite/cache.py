from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.eval_benchmark.datasets.tapvid_davis import (
    load_tapvid_davis_pickle,
    make_tapvid_davis_sequence,
)


def _write_queries_txt(path: Path, queries_txy: np.ndarray) -> None:
    q = np.asarray(queries_txy, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"queries_txy must be (N,3), got {q.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# format: t x y\n")
        f.write(f"# n={int(q.shape[0])}\n")
        for t, x, y in q.tolist():
            f.write(f"{t:.6f} {x:.6f} {y:.6f}\n")


def build_tapvid_davis_first_cache(
    *,
    pkl_path: str | Path,
    out_dir: str | Path,
    resize_hw: tuple[int, int] | None = (256, 256),
    max_videos: int | None = None,
    video_names: list[str] | None = None,
) -> list[str]:
    """
    Build a per-sequence cache for TAP-Vid DAVIS (first-query protocol).

    Output layout:
      <out_dir>/sequences/<seq>.npz
      <out_dir>/queries/<seq>.txt   (t x y, in the same pixel coords as frames)
    """

    pkl_path = Path(pkl_path)
    out_dir = Path(out_dir)

    raw = load_tapvid_davis_pickle(pkl_path)
    if video_names is None:
        names = sorted(raw.keys())
    else:
        names = list(video_names)
    if max_videos is not None:
        names = names[: int(max_videos)]

    seq_dir = out_dir / "sequences"
    q_dir = out_dir / "queries"
    seq_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        seq = make_tapvid_davis_sequence(name=str(name), item=raw[str(name)], resize_hw=resize_hw)

        # TAP-Vid: [t, y, x] -> runner expects [t, x, y].
        queries_txy = np.asarray(seq.query_points_tyx, dtype=np.float32)[:, [0, 2, 1]]

        seq_path = seq_dir / f"{seq.name}.npz"
        np.savez_compressed(
            seq_path,
            frames=np.asarray(seq.frames_uint8, dtype=np.uint8),
            query_points_tyx=np.asarray(seq.query_points_tyx, dtype=np.float32),
            queries_txy=queries_txy.astype(np.float32, copy=False),
            gt_tracks_xy=np.asarray(seq.gt_tracks_xy, dtype=np.float32),
            gt_occluded=np.asarray(seq.gt_occluded).astype(bool, copy=False),
        )

        _write_queries_txt(q_dir / f"{seq.name}.txt", queries_txy)

    return [str(n) for n in names]


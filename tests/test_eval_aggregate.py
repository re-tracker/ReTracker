from __future__ import annotations

import json
from pathlib import Path

from retracker.evaluation.aggregate import aggregate_metrics


def test_aggregate_metrics_means_over_files(tmp_path: Path) -> None:
    exp_dir = tmp_path
    metrics_dir = exp_dir / "metrics_per_seq"
    metrics_dir.mkdir(parents=True)

    dataset_name = "tapvid_davis_first"

    (metrics_dir / f"{dataset_name}_retracker_seqA.json").write_text(
        json.dumps({"average_jaccard": 0.5, "occlusion_accuracy": 0.9})
    )
    (metrics_dir / f"{dataset_name}_retracker_seqB.json").write_text(
        json.dumps({"average_jaccard": 0.7, "occlusion_accuracy": 0.1})
    )

    out = aggregate_metrics(str(exp_dir), dataset_name)

    assert out["dataset_name"] == dataset_name
    assert out["num_sequences"] == 2
    assert abs(out["avg"]["average_jaccard"] - 0.6) < 1e-6
    assert abs(out["avg"]["occlusion_accuracy"] - 0.5) < 1e-6

"""Aggregate per-sequence evaluation metrics.

Multi-GPU evaluation in this repo is typically done by launching multiple
processes, each evaluating a disjoint shard of sequences. Each process writes
per-sequence metrics into:

  <exp_dir>/metrics_per_seq/<dataset_name>_<engine>_<seq>.json

This module merges those files and computes the final average metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def _mean(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(sum(values) / len(values))


def aggregate_metrics(exp_dir: str, dataset_name: str) -> Dict:
    exp_dir = os.path.realpath(exp_dir)
    metrics_dir = Path(exp_dir) / "metrics_per_seq"
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"metrics_per_seq dir not found: {metrics_dir}")

    files = sorted(metrics_dir.glob(f"{dataset_name}_*.json"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No per-seq metrics found for dataset_name='{dataset_name}' under: {metrics_dir}"
        )

    by_sequence: Dict[str, Dict[str, float]] = {}
    metric_keys = set()

    for fp in files:
        with fp.open("r") as f:
            data = json.load(f)
        # File name stem is a stable artifact id: <dataset>_<engine>_<seq>.
        seq_id = fp.stem
        by_sequence[seq_id] = {k: float(v) for k, v in data.items()}
        metric_keys.update(by_sequence[seq_id].keys())

    avg: Dict[str, float] = {}
    for k in sorted(metric_keys):
        avg[k] = _mean([m[k] for m in by_sequence.values() if k in m])

    return {
        "dataset_name": dataset_name,
        "num_sequences": len(by_sequence),
        "avg": avg,
        "by_sequence": by_sequence,
    }


def main() -> None:
    enable_file_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Evaluation output directory (shared across shards).")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (e.g. tapvid_davis_first).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <exp_dir>/result_eval_<dataset_name>.json).",
    )
    args = parser.parse_args()

    out = aggregate_metrics(args.exp_dir, args.dataset_name)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.realpath(args.exp_dir), f"result_eval_{args.dataset_name}.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    CONSOLE.print(f"[cyan][aggregate][/cyan] Wrote: {out_path}")
    CONSOLE.print(f"[cyan][aggregate][/cyan] num_sequences={out['num_sequences']}")
    CONSOLE.print(f"[cyan][aggregate][/cyan] avg keys={list(out['avg'].keys())}")


if __name__ == "__main__":
    main()

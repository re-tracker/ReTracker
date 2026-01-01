#!/bin/bash
# Stage4 training smoke test (no external datasets required).
#
# Runs a single train/val batch via Lightning's `fast_dev_run`, using the
# synthetic "dummy" tracking dataset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="./outputs/training"
mkdir -p "$OUTPUT_DIR"

# Use the underlying training CLI module directly to avoid `retracker` wrapper
# interactions with Lightning's multi-process launchers.
python -m retracker.training.cli fit \
  --defaults_config configs/train/stage4_unified.yaml \
  --exp_config configs/train/smoke/stage4_dummy.yaml \
  --trainer.accelerator=cpu \
  --trainer.devices=1 \
  --trainer.strategy=auto \
  --trainer.precision=32-true \
  --trainer.default_root_dir="$OUTPUT_DIR" \
  --trainer.logger=false

#!/usr/bin/env bash
# Evaluate on TAP-Vid DAVIS dataset.
#
# Wrapper around `python -m retracker.evaluation.cli` via `common_eval.sh`.
#
# Usage:
#   bash scripts/evaluation/retracker_davis_first.sh <run_id|run_name_prefix|ckpt_path> [dataset_root]
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <run_id|run_name_prefix|ckpt_path> [dataset_root]"
  echo ""
  echo "Examples:"
  echo "  $0 my_run_abcd1234 data/tapvid"
  echo "  $0 ./weights/last.ckpt"
  exit 0
fi

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <run_id|run_name_prefix|ckpt_path> [dataset_root]"
  exit 1
fi

# Get script directory (portable, avoids `readlink -f`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common evaluation helpers.
source "$SCRIPT_DIR/common_eval.sh"

# Run evaluation with dataset-specific settings.
run_evaluation \
  "eval_tapvid_davis_first" \
  "tracking_val/tapvid_val" \
  "$1" \
  "${2:-}"


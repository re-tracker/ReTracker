#!/usr/bin/env bash
# Evaluate on TAP-Vid DAVIS dataset
set -euo pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <exp_name_or_ckpt_path> [dataset_root]"
    echo ""
    echo "Examples:"
    echo "  $0 my_experiment                    # Use checkpoint from experiment"
    echo "  $0 /path/to/model.ckpt             # Use specific checkpoint"
    echo "  $0 my_exp /custom/data/path        # Override dataset path"
    exit 1
fi

# Get script directory (portable, avoids `readlink -f`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common evaluation functions
source "$SCRIPT_DIR/common_eval.sh"

# Run evaluation with dataset-specific settings
run_evaluation \
    "eval_tapvid_davis_first" \
    "tracking_val/tapvid_val" \
    "$1" \
    "${2:-}"

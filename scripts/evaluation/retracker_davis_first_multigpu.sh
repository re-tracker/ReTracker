#!/usr/bin/env bash
# Multi-GPU evaluation on TAP-Vid DAVIS ("first" query mode).
#
# This script launches one process per GPU (sequence-level sharding) and writes
# all artifacts into a single shared output directory. It supports resume via:
#   - per-seq dumps (*.npz)
#   - per-seq metrics (*.json)
#   - done markers (*.done)
#
# Usage:
#   bash scripts/evaluation/retracker_davis_first_multigpu.sh <run_id|run_name_prefix|ckpt_path> [dataset_root] [gpus] [out_dir]
#
# Examples:
#   # Use all visible GPUs
#   bash scripts/evaluation/retracker_davis_first_multigpu.sh /path/to/retracker_b1.7.ckpt data/tapvid
#
#   # Use specific GPUs (indices within the current CUDA_VISIBLE_DEVICES mask)
#   bash scripts/evaluation/retracker_davis_first_multigpu.sh /path/to/retracker_b1.7.ckpt data/tapvid 0,1,2,3
#
#   # Resume into an existing output dir (recommended for interrupt/resume):
#   RETRACKER_EVAL_OUT_DIR=/path/to/outputs/eval/<run_id>/<config>/<timestamp> \
#   bash scripts/evaluation/retracker_davis_first_multigpu.sh /path/to/retracker_b1.7.ckpt data/tapvid 0,1

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <run_id|run_name_prefix|ckpt_path> [dataset_root] [gpus] [out_dir]" >&2
  exit 1
fi

INPUT_ARG="$1"
DATASET_ROOT_ARG="${2:-}"
GPUS_ARG="${3:-}"
OUT_DIR_ARG="${4:-}"

# Optional environment overrides.
SAVE_VIDEO="${RETRACKER_EVAL_SAVE_VIDEO:-true}"
VISUALIZE_EVERY="${RETRACKER_EVAL_VISUALIZE_EVERY:-1}"
OUT_DIR_OVERRIDE="${RETRACKER_EVAL_OUT_DIR:-}"

# Get repo root (portable, avoids `readlink -f`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRAIN_OUTPUT_ROOT="${RETRACKER_OUTPUT_ROOT:-./outputs/training}"
EVAL_OUTPUT_ROOT="${RETRACKER_EVAL_OUTPUT_ROOT:-./outputs/eval}"
DATA_ROOT_OVERRIDE="${RETRACKER_DATA_ROOT:-}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-./data}"

_abspath() {
  python -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

_die() {
  echo "ERROR: $*" >&2
  exit 1
}

# --------------------------------------------------------------------
# Resolve checkpoint (same behavior as common_eval.sh)
# --------------------------------------------------------------------
CHECKPOINT_PATH=""
RUN_ID=""

if [[ "$INPUT_ARG" == *.ckpt ]] || [[ -f "$INPUT_ARG" ]]; then
  CHECKPOINT_PATH="$(_abspath "$INPUT_ARG")"
  [[ -f "$CHECKPOINT_PATH" ]] || _die "Checkpoint file not found: $CHECKPOINT_PATH"
  RUN_ID="$(basename "$CHECKPOINT_PATH" .ckpt)"
else
  candidate="${TRAIN_OUTPUT_ROOT}/${INPUT_ARG}/version_shared_ckpt/last.ckpt"
  if [[ -f "$candidate" ]]; then
    RUN_ID="$INPUT_ARG"
    CHECKPOINT_PATH="$(_abspath "$candidate")"
  else
    matches=()
    while IFS= read -r -d '' d; do
      if [[ -f "$d/version_shared_ckpt/last.ckpt" ]]; then
        matches+=("$d")
      fi
    done < <(find "$TRAIN_OUTPUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name "${INPUT_ARG}_*" -print0 2>/dev/null || true)

    if [[ "${#matches[@]}" -eq 1 ]]; then
      RUN_ID="$(basename "${matches[0]}")"
      CHECKPOINT_PATH="$(_abspath "${matches[0]}/version_shared_ckpt/last.ckpt")"
    elif [[ "${#matches[@]}" -eq 0 ]]; then
      _die "Checkpoint not found for run '$INPUT_ARG'. Looked for:\n  - ${candidate}\n  - ${TRAIN_OUTPUT_ROOT}/${INPUT_ARG}_*/version_shared_ckpt/last.ckpt"
    else
      echo "ERROR: Multiple runs match prefix '$INPUT_ARG':" >&2
      for d in "${matches[@]}"; do
        echo "  - $(basename "$d")" >&2
      done
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------
# Resolve dataset root
# --------------------------------------------------------------------
DEFAULT_DATASET_PATH="tracking_val/tapvid_val"
EVAL_DATASET_ROOT=""
PASS_DATASET_ROOT="0"
if [[ -n "$DATASET_ROOT_ARG" ]]; then
  EVAL_DATASET_ROOT="$(_abspath "$DATASET_ROOT_ARG")"
  PASS_DATASET_ROOT="1"
elif [[ -n "$DATA_ROOT_OVERRIDE" ]]; then
  EVAL_DATASET_ROOT="$(_abspath "${DATA_ROOT}/${DEFAULT_DATASET_PATH}")"
  PASS_DATASET_ROOT="1"
fi

# --------------------------------------------------------------------
# Create shared output dir
# --------------------------------------------------------------------
CONFIG_NAME="eval_tapvid_davis_first"
if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
  OUT_DIR="$(_abspath "$OUT_DIR_OVERRIDE")"
elif [[ -n "$OUT_DIR_ARG" ]]; then
  OUT_DIR="$(_abspath "$OUT_DIR_ARG")"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="$(_abspath "${EVAL_OUTPUT_ROOT}/${RUN_ID}/${CONFIG_NAME}/${TS}")"
fi
mkdir -p "$OUT_DIR"

echo "[multigpu-eval] Config: $CONFIG_NAME"
echo "[multigpu-eval] Run id: $RUN_ID"
echo "[multigpu-eval] Checkpoint: $CHECKPOINT_PATH"
if [[ "$PASS_DATASET_ROOT" == "1" ]]; then
  echo "[multigpu-eval] Dataset root: $EVAL_DATASET_ROOT"
else
  echo "[multigpu-eval] Dataset root: (from configs/eval/default_config_eval.yaml + configs/paths_local.yaml)"
fi
echo "[multigpu-eval] Output dir: $OUT_DIR"

# --------------------------------------------------------------------
# GPU list
# --------------------------------------------------------------------
gpu_list=()
if [[ -n "$GPUS_ARG" ]]; then
  IFS=',' read -r -a gpu_list <<< "$GPUS_ARG"
else
  gpu_count="$(python -c "import torch; print(torch.cuda.device_count())")"
  if [[ "$gpu_count" -le 0 ]]; then
    _die "No GPUs visible. Set CUDA_VISIBLE_DEVICES or pass an explicit GPU list."
  fi
  for ((i=0; i<gpu_count; i++)); do
    gpu_list+=("$i")
  done
fi

NUM_SHARDS="${#gpu_list[@]}"
echo "[multigpu-eval] Using ${NUM_SHARDS} GPU(s): ${gpu_list[*]}"

# --------------------------------------------------------------------
# Launch one process per GPU
# --------------------------------------------------------------------
pids=()
for shard_id in "${!gpu_list[@]}"; do
  gpu_idx="${gpu_list[$shard_id]}"
  echo "[multigpu-eval] Launch shard ${shard_id}/${NUM_SHARDS} on gpu_idx=${gpu_idx}"

  cmd=(
    python -m retracker.evaluation.cli
    --config-name "$CONFIG_NAME"
    hydra/hydra_logging=disabled
    hydra/job_logging=disabled
    hydra.output_subdir=null
    hydra.run.dir="$OUT_DIR"
    exp_dir="$OUT_DIR"
    checkpoint="$CHECKPOINT_PATH"
    shard_id="$shard_id"
    num_shards="$NUM_SHARDS"
    gpu_idx="$gpu_idx"
    load_dump=true
    save_dump=true
    skip_if_done=true
    save_video="$SAVE_VIDEO"
    visualize_every="$VISUALIZE_EVERY"
  )
  if [[ "$PASS_DATASET_ROOT" == "1" ]]; then
    cmd+=(dataset_root="$EVAL_DATASET_ROOT")
  fi
  "${cmd[@]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  _die "One or more shards failed. Re-run with the same OUT_DIR (set RETRACKER_EVAL_OUT_DIR=...) to resume."
fi

# --------------------------------------------------------------------
# Aggregate metrics across all per-seq outputs
# --------------------------------------------------------------------
python -m retracker.evaluation.aggregate \
  --exp_dir "$OUT_DIR" \
  --dataset_name "tapvid_davis_first"

echo "[multigpu-eval] Done."

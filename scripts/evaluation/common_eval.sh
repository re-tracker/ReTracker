#!/usr/bin/env bash
# Common evaluation helpers for scripts in `scripts/evaluation/`.
#
# This file is intentionally open-source friendly:
# - no hard-coded personal paths
# - no conda activation or LD_LIBRARY_PATH mutation
# - no symlink hacks
#
# Environment variables:
#   RETRACKER_OUTPUT_ROOT        Training output root (default: ./outputs/training)
#   RETRACKER_EVAL_OUTPUT_ROOT   Evaluation output root (default: ./outputs/eval)
#   RETRACKER_DATA_ROOT          Data root used when dataset_root is not provided (default: ./data)
#   RETRACKER_EVAL_DRY_RUN       If "1", resolves paths and creates output dir, but skips Python eval.
#
# Function: Run evaluation with specified config and dataset.
# Args:
#   $1: Config name (e.g., eval_tapvid_davis_first)
#   $2: Default dataset path (e.g., tracking_val/tapvid_val)
#   $3: Input argument: <run_id|run_name_prefix|checkpoint_path>
#   $4: Optional dataset root override.
#       If omitted:
#         - if RETRACKER_DATA_ROOT is set: use $RETRACKER_DATA_ROOT/$DEFAULT_DATASET_PATH (legacy)
#         - else: rely on Hydra config defaults (configs/paths.yaml + configs/paths_local.yaml)
run_evaluation() {
  set -euo pipefail

  local CONFIG_NAME="$1"
  local DEFAULT_DATASET_PATH="$2"
  local INPUT_ARG="$3"
  local DATASET_ROOT_ARG="${4:-}"

  local REPO_ROOT
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd "$REPO_ROOT"

  local TRAIN_OUTPUT_ROOT="${RETRACKER_OUTPUT_ROOT:-./outputs/training}"
  local EVAL_OUTPUT_ROOT="${RETRACKER_EVAL_OUTPUT_ROOT:-./outputs/eval}"
  # Keep backwards compatibility:
  # - If RETRACKER_DATA_ROOT is explicitly set, we keep the old behavior and
  #   pass dataset_root=$RETRACKER_DATA_ROOT/$DEFAULT_DATASET_PATH.
  # - Otherwise, if the user does not pass dataset_root, we let Hydra configs
  #   (configs/paths.yaml + configs/paths_local.yaml) decide the default dataset_root.
  local DATA_ROOT_OVERRIDE="${RETRACKER_DATA_ROOT:-}"
  local DATA_ROOT="${DATA_ROOT_OVERRIDE:-./data}"

  _abspath() {
    python -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
  }

  _die() {
    echo "ERROR: $*" >&2
    exit 1
  }

  local CHECKPOINT_PATH=""
  local RUN_ID=""

  # Input can be a direct checkpoint path, or a run identifier.
  if [[ "$INPUT_ARG" == *.ckpt ]] || [[ -f "$INPUT_ARG" ]]; then
    CHECKPOINT_PATH="$(_abspath "$INPUT_ARG")"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
      _die "Checkpoint file not found: $CHECKPOINT_PATH"
    fi
    RUN_ID="$(basename "$CHECKPOINT_PATH" .ckpt)"
  else
    local candidate="${TRAIN_OUTPUT_ROOT}/${INPUT_ARG}/version_shared_ckpt/last.ckpt"
    if [[ -f "$candidate" ]]; then
      RUN_ID="$INPUT_ARG"
      CHECKPOINT_PATH="$(_abspath "$candidate")"
    else
      # Prefix resolution: if the user passes "my_run", and there is exactly one
      # outputs folder like "my_run_<gitsha>", resolve it automatically.
      local matches=()
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

  local EVAL_DATASET_ROOT=""
  local PASS_DATASET_ROOT="0"
  if [[ -n "$DATASET_ROOT_ARG" ]]; then
    EVAL_DATASET_ROOT="$DATASET_ROOT_ARG"
    PASS_DATASET_ROOT="1"
  elif [[ -n "$DATA_ROOT_OVERRIDE" ]]; then
    EVAL_DATASET_ROOT="${DATA_ROOT}/${DEFAULT_DATASET_PATH}"
    PASS_DATASET_ROOT="1"
  fi

  local TS
  TS="$(date +%Y%m%d_%H%M%S)"
  local OUT_DIR="${EVAL_OUTPUT_ROOT}/${RUN_ID}/${CONFIG_NAME}/${TS}"

  mkdir -p "$OUT_DIR"

  echo "[eval] Config: $CONFIG_NAME"
  echo "[eval] Run id: $RUN_ID"
  echo "[eval] Checkpoint: $CHECKPOINT_PATH"
  if [[ "$PASS_DATASET_ROOT" == "1" ]]; then
    echo "[eval] Dataset root: $EVAL_DATASET_ROOT"
  else
    echo "[eval] Dataset root: (from configs/eval/default_config_eval.yaml + configs/paths_local.yaml)"
  fi
  echo "[eval] Output dir: $OUT_DIR"

  if [[ "${RETRACKER_EVAL_DRY_RUN:-0}" == "1" ]]; then
    echo "[eval] DRY RUN (skipping python -m retracker.evaluation.cli)"
    return 0
  fi

  local cmd=(
    python -m retracker.evaluation.cli
    --config-name "$CONFIG_NAME"
    hydra/hydra_logging=disabled
    hydra/job_logging=disabled
    hydra.output_subdir=null
    hydra.run.dir="$OUT_DIR"
    exp_dir="$OUT_DIR"
    checkpoint="$CHECKPOINT_PATH"
  )
  if [[ "$PASS_DATASET_ROOT" == "1" ]]; then
    cmd+=(dataset_root="$EVAL_DATASET_ROOT")
  fi
  # Optional Hydra overrides (space-separated), e.g.:
  #   RETRACKER_EVAL_OVERRIDES='interp_shape=[768,768] tapvid_resize_to=[768,768] save_video=false'
  if [[ -n "${RETRACKER_EVAL_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    extra_overrides=(${RETRACKER_EVAL_OVERRIDES})
    cmd+=("${extra_overrides[@]}")
  fi

  "${cmd[@]}"
}

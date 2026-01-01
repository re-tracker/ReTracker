#!/usr/bin/env bash
# Training launcher (open-source friendly).
#
# This script intentionally avoids:
# - hard-coded personal paths
# - auto symlinking repo folders to private storage
# - cluster-specific helpers (daemons, environment activation)
#
# It is a thin wrapper around `python -m retracker.training.cli`.
#
# Requirements:
#   python -m pip install -e ".[train]"
#
# Usage:
#   bash scripts/train/train.sh stage4 <run_name> [--output_root DIR] [extra LightningCLI args...]
#
# Examples:
#   # Basic run
#   bash scripts/train/train.sh stage4 my_run
#
#   # Fast sanity check (1 train + 1 val batch)
#   bash scripts/train/train.sh stage4 smoke --trainer.fast_dev_run=1 --trainer.devices=1
#
#   # Choose a custom output root (default: ./outputs/training)
#   bash scripts/train/train.sh stage4 my_run --output_root ./outputs/training
#
# Notes:
# - Dataset/weights paths are configured via `configs/paths_local.yaml` (gitignored).
# - This script auto-resumes if `version_shared_ckpt/last.ckpt` exists (can be disabled via
#   `--no_resume`).
set -euo pipefail


usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/train.sh stage4 <run_name> [--output_root DIR] [--no_git_sha] [--no_resume] [extra args...]

Stage:
  stage4  Uses configs/train/stage4_unified.yaml (and configs/train/debug/stage4_debug.yaml for debug runs)

Environment variables:
  RETRACKER_OUTPUT_ROOT      Override the output root directory (default: ./outputs/training)
  RETRACKER_APPEND_GIT_SHA   1/0, append _<gitsha> to run_name (default: 1)
  RETRACKER_AUTO_RESUME      1/0, auto-resume when last.ckpt exists (default: 1)

Examples:
  bash scripts/train/train.sh stage4 my_exp
  bash scripts/train/train.sh stage4 my_exp --trainer.devices=1 --trainer.precision=bf16-mixed
  bash scripts/train/train.sh stage4 debug_run --trainer.fast_dev_run=1
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

STAGE="$1"
RUN_NAME="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${RETRACKER_OUTPUT_ROOT:-./outputs/training}"
APPEND_GIT_SHA="${RETRACKER_APPEND_GIT_SHA:-1}"
AUTO_RESUME="${RETRACKER_AUTO_RESUME:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --no_git_sha)
      APPEND_GIT_SHA=0
      shift
      ;;
    --no_resume)
      AUTO_RESUME=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$STAGE" != "stage4" ]]; then
  echo "Error: unsupported stage: $STAGE" >&2
  echo "Only 'stage4' is supported by this script." >&2
  exit 1
fi

DEFAULTS_CONFIG="configs/train/stage4_unified.yaml"
DEBUG_EXP_CONFIG="configs/train/debug/stage4_debug.yaml"
TRAINER_CONFIG="configs/train/trainer_default.yaml"

if [[ ! -f "$DEFAULTS_CONFIG" ]]; then
  echo "Error: missing config: $DEFAULTS_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$TRAINER_CONFIG" ]]; then
  echo "Error: missing config: $TRAINER_CONFIG" >&2
  exit 1
fi

GIT_SHA="nogit"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git rev-parse --short HEAD)"
fi

RUN_ID="$RUN_NAME"
if [[ "$APPEND_GIT_SHA" == "1" ]]; then
  RUN_ID="${RUN_NAME}_${GIT_SHA}"
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SHARED_CKPT_DIR="${RUN_DIR}/version_shared_ckpt"
RESUME_CKPT="${SHARED_CKPT_DIR}/last.ckpt"

mkdir -p "$OUTPUT_ROOT"

# Decide whether we should use the debug exp_config automatically.
EXP_CONFIG_ARG=()
if [[ "$RUN_NAME" == *"debug"* ]]; then
  if [[ -f "$DEBUG_EXP_CONFIG" ]]; then
    EXP_CONFIG_ARG=(--exp_config "$DEBUG_EXP_CONFIG")
  else
    echo "Warning: debug config not found: $DEBUG_EXP_CONFIG (continuing without it)" >&2
  fi
fi

CKPT_ARG=()
if [[ "$AUTO_RESUME" == "1" && -f "$RESUME_CKPT" ]]; then
  echo "[train] Auto-resume from: $RESUME_CKPT"
  CKPT_ARG=(--ckpt_path="$RESUME_CKPT")
else
  echo "[train] Starting fresh (no resume checkpoint found)."
fi

echo "[train] Repo root:   $REPO_ROOT"
echo "[train] Defaults:    $DEFAULTS_CONFIG"
if [[ ${#EXP_CONFIG_ARG[@]} -gt 0 ]]; then
  echo "[train] Exp config:  ${EXP_CONFIG_ARG[1]}"
fi
echo "[train] Output root: $OUTPUT_ROOT"
echo "[train] Run id:      $RUN_ID"
echo "[train] Run dir:     $RUN_DIR"
echo "[train] Extra args:  ${EXTRA_ARGS[*]:-<none>}"

# Run training.
#
# Note: we pass logger.save_dir with '=' (not as a separate token) because
# retracker.training.cli uses it to locate a shared temp-config directory when
# WORLD_SIZE > 1.
python -m retracker.training.cli fit \
  --trainer "$TRAINER_CONFIG" \
  --defaults_config "$DEFAULTS_CONFIG" \
  "${EXP_CONFIG_ARG[@]}" \
  --trainer.logger.init_args.save_dir="$OUTPUT_ROOT" \
  --trainer.logger.init_args.name="$RUN_ID" \
  --trainer.default_root_dir="$RUN_DIR" \
  --model.dump_dir="$SHARED_CKPT_DIR" \
  "${CKPT_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
  

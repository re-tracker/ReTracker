#!/usr/bin/env bash
set -euo pipefail

# Unified evaluation benchmark runner for TAP-Vid DAVIS (first-query protocol).
#
# This wraps `experiments/eval_benchmark/run_tapvid_davis_first_suite.py` and
# keeps a similar "one-liner" UX to the older eval scripts.
#
# Usage:
#   ./scripts/evaluation/eval_benchmark_davis_first_suite.sh /path/to/retracker_b1.7.ckpt [dataset_root] [extra_args...]
#
# Examples:
#   # ReTracker only (default):
#   ./scripts/evaluation/eval_benchmark_davis_first_suite.sh /path/to/retracker_b1.7.ckpt
#
#   # Multiple methods:
#   METHODS="retracker,trackon2,cotracker3_offline,cotracker3_online,tapir,tapnext" \
#     GPUS="0,1" \
#     ./scripts/evaluation/eval_benchmark_davis_first_suite.sh /path/to/retracker_b1.7.ckpt
#
# Notes:
# - Third-party repos/checkpoints are NOT included by default. Install with:
#     bash experiments/benchmark/scripts/install_third_party.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

CKPT="${1:-}"
ARG2="${2:-}"
DATASET_ROOT=""
EXTRA_ARGS=()

# Arg parsing:
#   $1 = checkpoint (required)
#   $2 = dataset_root (optional) unless it looks like a flag (starts with '-')
#   $3... = extra args forwarded to python runner
if [[ -n "${ARG2}" && "${ARG2}" != "-"* ]]; then
  DATASET_ROOT="${ARG2}"
  EXTRA_ARGS=("${@:3}")
else
  EXTRA_ARGS=("${@:2}")
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Missing checkpoint argument." >&2
  echo "Usage: $0 /path/to/retracker_b1.7.ckpt [dataset_root] [extra_args...]" >&2
  exit 1
fi
CKPT="$(readlink -f "${CKPT}")"
if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ -z "${DATASET_ROOT}" ]]; then
  # Prefer the repo-shipped TAP-Vid subset for out-of-the-box runs.
  if [[ -d "${REPO_ROOT}/data/tapvid_local" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tapvid_local"
  elif [[ -d "${REPO_ROOT}/data/tapvid" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tapvid"
  elif [[ -d "${REPO_ROOT}/data/tracking_val/tapvid_val" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tracking_val/tapvid_val"
  else
    echo "[ERROR] Dataset root not found. Pass it explicitly as arg2." >&2
    exit 1
  fi
fi

# Optional overrides via env vars.
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/eval_benchmark}"
GPUS="${GPUS:-0}"
METHODS="${METHODS:-retracker}"

echo "[EvalSuite] repo_root=${REPO_ROOT}"
echo "[EvalSuite] ckpt=${CKPT}"
echo "[EvalSuite] dataset_root=${DATASET_ROOT}"
echo "[EvalSuite] out_dir=${OUT_DIR}"
echo "[EvalSuite] gpus=${GPUS}"
echo "[EvalSuite] methods=${METHODS}"

# Warn early if the user requested third-party methods but hasn't installed them.
THIRD_PARTY_DIR="${REPO_ROOT}/experiments/third_party"
if [[ "${METHODS}" == *"trackon2"* || "${METHODS}" == *"cotracker3"* || "${METHODS}" == *"tapir"* || "${METHODS}" == *"tapnext"* ]]; then
  if [[ ! -d "${THIRD_PARTY_DIR}" ]]; then
    echo "[ERROR] Missing third-party folder: ${THIRD_PARTY_DIR}" >&2
    echo "[Hint] Run: bash ${REPO_ROOT}/experiments/benchmark/scripts/install_third_party.sh" >&2
    exit 1
  fi
fi

"${PY}" "${REPO_ROOT}/experiments/eval_benchmark/run_tapvid_davis_first_suite.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --methods "${METHODS}" \
  --gpus "${GPUS}" \
  --retracker-ckpt "${CKPT}" \
  "${EXTRA_ARGS[@]}"

echo "[EvalSuite] Done."
echo "  ${OUT_DIR}/tapvid_davis_first/compare_methods.json"

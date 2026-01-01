#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end smoke test for TAP-Vid DAVIS (first protocol) evaluation.
#
# Usage:
#   CKPT=/path/to/retracker_b1.7.ckpt bash experiments/eval_benchmark/scripts/smoke_test_davis_first.sh
#
# Optional overrides:
#   DATASET_ROOT=/path/to/tapvid_val
#   OUT_DIR=/path/to/outputs
#   GPUS=0            # or "0,1" or "" (CPU)
#   MAX_VIDEOS=2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-python}"

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Missing checkpoint. Set CKPT=/abs/path/to/retracker_b1.7.ckpt" >&2
  exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "${DATASET_ROOT}" ]]; then
  if [[ -d "${REPO_ROOT}/data/tapvid_local" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tapvid_local"
  elif [[ -d "${REPO_ROOT}/data/tapvid" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tapvid"
  elif [[ -d "${REPO_ROOT}/data/tracking_val/tapvid_val" ]]; then
    DATASET_ROOT="${REPO_ROOT}/data/tracking_val/tapvid_val"
  else
    echo "[ERROR] Dataset root not found. Set DATASET_ROOT=/abs/path/to/tapvid_val" >&2
    exit 1
  fi
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/eval_benchmark_smoke}"
GPUS="${GPUS:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-2}"

echo "[Smoke] repo_root=${REPO_ROOT}"
echo "[Smoke] ckpt=${CKPT}"
echo "[Smoke] dataset_root=${DATASET_ROOT}"
echo "[Smoke] out_dir=${OUT_DIR}"
echo "[Smoke] gpus=${GPUS} max_videos=${MAX_VIDEOS}"

"${PY}" "${REPO_ROOT}/experiments/eval_benchmark/run_tapvid_davis_first.py" \
  --ckpt "${CKPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --gpus "${GPUS}" \
  --max-videos "${MAX_VIDEOS}"

echo "[Smoke] Result:"
echo "  ${OUT_DIR}/tapvid_davis_first/retracker/result_eval_tapvid_davis_first.json"

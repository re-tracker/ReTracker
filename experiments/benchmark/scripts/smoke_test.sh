#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

VIDEO_DEFAULT="${ROOT_DIR}/data/slam_demo/rgbd_dataset_freiburg1_room/freiburgl_room.mp4"
VIDEO="${VIDEO:-${VIDEO_DEFAULT}}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/benchmark_smoke}"
RESIZED_W="${RESIZED_W:-512}"
RESIZED_H="${RESIZED_H:-384}"
GRID_SIZE="${GRID_SIZE:-10}"
MAX_FRAMES="${MAX_FRAMES:-60}"
FPS="${FPS:-10}"
POINT_SIZE="${POINT_SIZE:-6}"

QUERIES_TXT="${OUT_ROOT}/queries_grid${GRID_SIZE}_t0.txt"

if [[ ! -f "${VIDEO}" ]]; then
  echo "[ERROR] Smoke-test video not found: ${VIDEO}" >&2
  echo "[Hint] Set VIDEO=/path/to/video.mp4 and rerun." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi

export CONDA_NO_PLUGINS=true
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BASE="$(dirname "$(dirname "$(readlink -f "$(command -v conda)")")")"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "${OUT_ROOT}"

echo "[Info] Generating queries: ${QUERIES_TXT}"
python "${ROOT_DIR}/experiments/benchmark/generate_grid_queries.py" \
  --grid-size "${GRID_SIZE}" \
  --height "${RESIZED_H}" \
  --width "${RESIZED_W}" \
  --t 0 \
  --out "${QUERIES_TXT}"

echo "[Info] Running benchmark (MAX_FRAMES=${MAX_FRAMES})..."
conda activate retracker_env
python "${ROOT_DIR}/experiments/benchmark/benchmark.py" \
  --pair "${VIDEO}" "${QUERIES_TXT}" \
  --out-dir "${OUT_ROOT}" \
  --resized-w "${RESIZED_W}" \
  --resized-h "${RESIZED_H}" \
  --max-frames "${MAX_FRAMES}" \
  --fps "${FPS}" \
  --point-size "${POINT_SIZE}" \
  --cols 3
conda deactivate

VIDEO_STEM="$(basename -- "${VIDEO}")"
VIDEO_STEM="${VIDEO_STEM%.*}"
MOSAIC="${OUT_ROOT}/${VIDEO_STEM}/${VIDEO_STEM}_benchmark.mp4"

if [[ ! -f "${MOSAIC}" ]]; then
  echo "[ERROR] Expected mosaic not found: ${MOSAIC}" >&2
  exit 1
fi

echo "[Done] Smoke test OK. Mosaic:"
echo "  ${MOSAIC}"


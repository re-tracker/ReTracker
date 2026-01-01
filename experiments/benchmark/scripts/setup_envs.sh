#!/usr/bin/env bash
set -euo pipefail

# Create conda envs used by the benchmark:
# - retracker_env  (ReTracker)   -> created by this repo's standard install
# - trackon2       (Track-On2 + CoTracker3)
# - tapnext        (TAPIR, PyTorch)  -> clone of trackon2 + a few extra deps
# - tapnet         (TapNext, JAX)    -> lightweight CPU JAX/Flax env (can be swapped for GPU JAX)
#
# Notes:
# - This script is best-effort and intentionally explicit/pinned.
# - If you have flaky connectivity, the script uses `/usr/local/bin/proxy` when available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/experiments/third_party"
TRACKON_DIR="${THIRD_PARTY_DIR}/track_on"

TRACKON_ENV="${TRACKON_ENV:-trackon2}"
TAPNEXT_ENV="${TAPNEXT_ENV:-tapnext}"
TAPNET_ENV="${TAPNET_ENV:-tapnet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TAPNET_PYTHON_VERSION="${TAPNET_PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi

CONDA="conda"
if command -v proxy >/dev/null 2>&1; then
  CONDA="proxy conda"
  echo "[Info] Using proxy wrapper for conda/pip downloads."
fi

# Avoid conda plugins (CUDA detector can crash in restricted environments).
export CONDA_NO_PLUGINS=true
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-}"

if [[ ! -d "${TRACKON_DIR}" ]]; then
  echo "[ERROR] Track-On2 repo not found at: ${TRACKON_DIR}" >&2
  echo "[Hint] Run:" >&2
  echo "  bash ${ROOT_DIR}/experiments/benchmark/scripts/install_third_party.sh" >&2
  exit 1
fi

CONDA_BASE="$(dirname "$(dirname "$(readlink -f "$(command -v conda)")")")"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  local name="$1"
  set +e
  conda run -n "${name}" python -c 'import sys; print(sys.executable)' >/dev/null 2>&1
  local rc=$?
  set -e
  return "${rc}"
}

echo "[Info] Creating env: ${TRACKON_ENV} (python=${PYTHON_VERSION})"
if env_exists "${TRACKON_ENV}"; then
  echo "[Info] Env already exists: ${TRACKON_ENV} (skip create)"
else
  ${CONDA} create -n "${TRACKON_ENV}" "python=${PYTHON_VERSION}" -y

  echo "[Info] Installing PyTorch (CUDA 12.1) + torchvision/torchaudio"
  ${CONDA} install -n "${TRACKON_ENV}" \
    pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

  echo "[Info] Upgrading pip"
  ${CONDA} run -n "${TRACKON_ENV}" python -m pip install -U pip

  echo "[Info] Installing mmcv"
  ${CONDA} run -n "${TRACKON_ENV}" python -m pip install \
    "mmcv==2.2.0" \
    -f "https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html"

  echo "[Info] Installing Track-On2 requirements (patched if needed)"
  REQ_SRC="${TRACKON_DIR}/requirements.txt"
  REQ_TMP="${ROOT_DIR}/experiments/benchmark/checkpoints/trackon_requirements_patched.txt"
  export REQ_SRC REQ_TMP
  python - <<'PY'
import os
from pathlib import Path

req_src = Path(os.environ["REQ_SRC"])
req_tmp = Path(os.environ["REQ_TMP"])

src_lines = req_src.read_text(encoding="utf-8").splitlines()
out_lines = []
for line in src_lines:
    s = line.strip()
    if s == "imageio[ffmpeg]==0.6.0":
        out_lines.append("imageio==2.37.2")
        out_lines.append("imageio-ffmpeg==0.6.0")
    else:
        out_lines.append(line)

req_tmp.parent.mkdir(parents=True, exist_ok=True)
req_tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print("[Info] Wrote patched requirements:", str(req_tmp))
PY
  ${CONDA} run -n "${TRACKON_ENV}" python -m pip install -r "${REQ_TMP}"
fi

echo "[Info] Building iJIT shim"
bash "${ROOT_DIR}/experiments/benchmark/scripts/build_jitprofiling_stub.sh"

echo "[Info] Smoke test: import torch in ${TRACKON_ENV}"
conda activate "${TRACKON_ENV}"
export MPLBACKEND=Agg
export LD_PRELOAD="${ROOT_DIR}/experiments/benchmark/scripts/libjitprofiling_stub.so${LD_PRELOAD:+:${LD_PRELOAD}}"
python -c 'import torch; print("torch", torch.__version__)'
conda deactivate

echo "[Info] Creating env: ${TAPNEXT_ENV} (clone of ${TRACKON_ENV})"
if env_exists "${TAPNEXT_ENV}"; then
  echo "[Info] Env already exists: ${TAPNEXT_ENV} (skip create)"
else
  ${CONDA} create -n "${TAPNEXT_ENV}" --clone "${TRACKON_ENV}" -y
fi

echo "[Info] Installing TAPIR extra deps into ${TAPNEXT_ENV}: dm-tree, einshape"
${CONDA} run -n "${TAPNEXT_ENV}" python -m pip install -U pip
${CONDA} run -n "${TAPNEXT_ENV}" python -m pip install dm-tree einshape

echo "[Info] Smoke test: import torch in ${TAPNEXT_ENV} (with LD_PRELOAD shim)"
conda activate "${TAPNEXT_ENV}"
export MPLBACKEND=Agg
export LD_PRELOAD="${ROOT_DIR}/experiments/benchmark/scripts/libjitprofiling_stub.so${LD_PRELOAD:+:${LD_PRELOAD}}"
python -c 'import torch; print("torch", torch.__version__)'
conda deactivate

echo "[Info] Creating env: ${TAPNET_ENV} (python=${TAPNET_PYTHON_VERSION})"
if env_exists "${TAPNET_ENV}"; then
  echo "[Info] Env already exists: ${TAPNET_ENV} (skip create)"
else
  ${CONDA} create -n "${TAPNET_ENV}" "python=${TAPNET_PYTHON_VERSION}" -y
  ${CONDA} run -n "${TAPNET_ENV}" python -m pip install -U pip
  # TapNext (JAX/Flax) - CPU-only install by default.
  ${CONDA} run -n "${TAPNET_ENV}" python -m pip install "jax[cpu]" flax einops opencv-python-headless

  # Optional: install TapNet (for future use / consistency with upstream demos).
  TAPNET_DIR="${THIRD_PARTY_DIR}/tapnet"
  if [[ -d "${TAPNET_DIR}" ]]; then
    ${CONDA} run -n "${TAPNET_ENV}" python -m pip install -e "${TAPNET_DIR}"
  fi
fi

echo "[Info] Smoke test: import jax/flax in ${TAPNET_ENV}"
conda activate "${TAPNET_ENV}"
python -c 'import jax, flax, einops; print("jax", jax.__version__); print("flax", flax.__version__)'
conda deactivate

echo "[Done] Envs ready:"
echo "  - ${TRACKON_ENV}"
echo "  - ${TAPNEXT_ENV}"
echo "  - ${TAPNET_ENV}"

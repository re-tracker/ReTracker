#!/usr/bin/env bash
set -euo pipefail

# Clone third-party repos into experiments/third_party and download checkpoints
# into experiments/benchmark/checkpoints.
#
# This script is intentionally opt-in: a fresh `git clone` of re-tracker does
# NOT fetch these repos automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/experiments/third_party"
CKPT_DIR="${ROOT_DIR}/experiments/benchmark/checkpoints"

TRACKON_DIR="${THIRD_PARTY_DIR}/track_on"
COTRACKER_DIR="${THIRD_PARTY_DIR}/co-tracker"
TAPNET_DIR="${THIRD_PARTY_DIR}/tapnet"

mkdir -p "${THIRD_PARTY_DIR}" "${CKPT_DIR}"

GIT="git"
WGET="wget"
if command -v proxy >/dev/null 2>&1; then
  echo "[Info] Using proxy wrapper for git/wget downloads."
  GIT="proxy git"
  WGET="proxy wget"
fi

# Pinned commits from a known-good setup (can be changed later).
TRACKON_REPO_URL="${TRACKON_REPO_URL:-https://github.com/gorkaydemir/track_on.git}"
TRACKON_COMMIT="${TRACKON_COMMIT:-7672a5e5432a20930d8a9fbba3340fff383138f7}"

COTRACKER_REPO_URL="${COTRACKER_REPO_URL:-https://github.com/facebookresearch/co-tracker.git}"
COTRACKER_COMMIT="${COTRACKER_COMMIT:-82e02e8029753ad4ef13cf06be7f4fc5facdda4d}"

TAPNET_REPO_URL="${TAPNET_REPO_URL:-https://github.com/google-deepmind/tapnet.git}"
TAPNET_COMMIT="${TAPNET_COMMIT:-c7e3ff3000a53ec628ec676acc719722f637bba4}"

clone_or_update() {
  local url="$1"
  local dst="$2"
  local commit="$3"

  if [[ ! -d "${dst}/.git" ]]; then
    echo "[Info] Cloning: ${url} -> ${dst}"
    ${GIT} clone "${url}" "${dst}"
  else
    echo "[Info] Repo already exists: ${dst} (skip clone)"
  fi

  # Checkout pinned commit for reproducibility (best-effort).
  if [[ -n "${commit}" ]]; then
    echo "[Info] Checking out commit: ${commit}"
    (cd "${dst}" && ${GIT} fetch --all --tags && ${GIT} checkout "${commit}")
  fi
}

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "[Info] Exists: ${out}"
    return 0
  fi
  echo "[Info] Downloading: ${url}"
  ${WGET} -O "${out}" "${url}"
}

echo "[Info] Installing third_party repos..."
clone_or_update "${TRACKON_REPO_URL}" "${TRACKON_DIR}" "${TRACKON_COMMIT}"
clone_or_update "${COTRACKER_REPO_URL}" "${COTRACKER_DIR}" "${COTRACKER_COMMIT}"
clone_or_update "${TAPNET_REPO_URL}" "${TAPNET_DIR}" "${TAPNET_COMMIT}"

echo "[Info] Downloading checkpoints..."
download_if_missing \
  "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov2_checkpoint.pt?download=true" \
  "${CKPT_DIR}/trackon2_dinov2_checkpoint.pt"

download_if_missing \
  "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" \
  "${CKPT_DIR}/scaled_offline.pth"

download_if_missing \
  "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth" \
  "${CKPT_DIR}/scaled_online.pth"

download_if_missing \
  "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt" \
  "${CKPT_DIR}/causal_bootstapir_checkpoint.pt"

download_if_missing \
  "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt" \
  "${CKPT_DIR}/tapir_checkpoint_panning.pt"

download_if_missing \
  "https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz" \
  "${CKPT_DIR}/bootstapnext_ckpt.npz"

echo "[Info] Building iJIT shim..."
bash "${ROOT_DIR}/experiments/benchmark/scripts/build_jitprofiling_stub.sh"

echo "[Done] third_party + checkpoints ready."
echo "[Next] Create conda envs (if you don't have them):"
echo "  bash ${ROOT_DIR}/experiments/benchmark/scripts/setup_envs.sh"
echo "[Next] Run smoke test:"
echo "  bash ${ROOT_DIR}/experiments/benchmark/scripts/smoke_test.sh"

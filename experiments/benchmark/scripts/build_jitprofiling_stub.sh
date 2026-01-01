#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/scripts/jitprofiling_stub.c"
OUT="${ROOT_DIR}/scripts/libjitprofiling_stub.so"

if [[ ! -f "${SRC}" ]]; then
  echo "[ERROR] Missing source: ${SRC}" >&2
  exit 1
fi

cc -shared -fPIC -O2 -o "${OUT}" "${SRC}"
echo "[Info] Built: ${OUT}"


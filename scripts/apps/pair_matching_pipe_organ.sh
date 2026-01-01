#!/bin/bash
# Pair matching example (pipe_organ).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

REF_IMAGE="$PROJECT_ROOT/data/sfm_demo/pipe_organ/017.jpg"
TGT_IMAGE="$PROJECT_ROOT/data/sfm_demo/pipe_organ/019.jpg"
OUTPUT_DIR="$PROJECT_ROOT/outputs/apps/pair_matching"
OUTPUT_PATH="$OUTPUT_DIR/pipe_organ_017_019.png"

if [ ! -f "$REF_IMAGE" ]; then
  echo "Error: ref image not found: $REF_IMAGE"
  exit 1
fi
if [ ! -f "$TGT_IMAGE" ]; then
  echo "Error: target image not found: $TGT_IMAGE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -m retracker.apps.pair_matching \
  --ref_image "$REF_IMAGE" \
  --tgt_image "$TGT_IMAGE" \
  --output "$OUTPUT_PATH" \
  --query_strategy sift \
  --sift_n_features 1000

echo "Saved: $OUTPUT_PATH"

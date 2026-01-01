#!/bin/bash
# Run the triangulation pipeline.
#
# Usage:
#   bash scripts/apps/multiview_triangulation.sh              # Full pipeline, first 25 frames
#   bash scripts/apps/multiview_triangulation.sh --step tracking   # Only tracking step
#   bash scripts/apps/multiview_triangulation.sh --force_rerun     # Rerun all steps
#   bash scripts/apps/multiview_triangulation.sh --end_frame 100   # More frames
#
# Output structure:
#   outputs/multiview_triangulation/{dataset_name}/
#       ├── tracking_{views}_f{start}-{end}.pkl
#       ├── triangulation_{views}_f{start}-{end}.pkl
#       ├── pointcloud_{views}_f{start}-{end}.mp4
#       └── debug/
#           └── matching_*.png

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Running triangulation pipeline..."
echo "Working directory: $(pwd)"

python -m retracker.apps.multiview_triangulation "$@"

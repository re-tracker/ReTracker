#!/bin/bash
# Multi-view tracking launcher
#
# Usage:
#   bash scripts/apps/multiview_tracking.sh [options]
#
# Examples:
#   # Default: track views 19, 25, 28 with reference view 25
#   CKPT=/path/to/retracker_b1.7.ckpt DATA_ROOT=/path/to/sequence/images \
#     bash scripts/apps/multiview_tracking.sh
#
#   # Custom views
#   CKPT=/path/to/retracker_b1.7.ckpt DATA_ROOT=/path/to/sequence/images \
#     bash scripts/apps/multiview_tracking.sh --views 10 20 30 --ref_view 20
#
#   # More points
#   CKPT=/path/to/retracker_b1.7.ckpt DATA_ROOT=/path/to/sequence/images \
#     bash scripts/apps/multiview_tracking.sh --num_points 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
DATA_ROOT="${DATA_ROOT:-}"
VIEWS="19 25 28"
REF_VIEW="25"
NUM_POINTS=2000
OUTPUT="outputs/multiview_tracking.mp4"
DEVICE="${DEVICE:-auto}"
CKPT="${CKPT:-}"
LAYOUT="horizontal"
FPS=30
NO_DISPLAY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --views)
            VIEWS=""
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                VIEWS="$VIEWS $1"
                shift
            done
            ;;
        --ref_view)
            REF_VIEW="$2"
            shift 2
            ;;
        --num_points)
            NUM_POINTS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --layout)
            LAYOUT="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --no_display)
            NO_DISPLAY="--no_display"
            shift
            ;;
        -h|--help)
            echo "Multi-view tracking"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root PATH     Data directory (required)"
            echo "  --views IDs          Space-separated view IDs (default: $VIEWS)"
            echo "  --ref_view ID        Reference view for point detection (default: $REF_VIEW)"
            echo "  --num_points N       Number of points to track (default: $NUM_POINTS)"
            echo "  --output PATH        Output video path (default: $OUTPUT)"
            echo "  --device DEVICE      cuda or cpu (default: $DEVICE)"
            echo "  --ckpt PATH          Model checkpoint (required)"
            echo "  --layout LAYOUT      horizontal, vertical, or grid (default: $LAYOUT)"
            echo "  --fps FPS            Output video FPS (default: $FPS)"
            echo "  --no_display         Disable live display"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Error: missing --data_root (or set DATA_ROOT=/path/to/sequence/images)" >&2
  exit 1
fi

if [[ -z "${CKPT}" ]]; then
  echo "Error: missing --ckpt (or set CKPT=/path/to/retracker_b1.7.ckpt)" >&2
  exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

echo "======================================"
echo "Multi-View Tracking"
echo "======================================"
echo "Data root:    $DATA_ROOT"
echo "Views:        $VIEWS"
echo "Reference:    $REF_VIEW"
echo "Num points:   $NUM_POINTS"
echo "Output:       $OUTPUT"
echo "Device:       $DEVICE"
echo "Checkpoint:   $CKPT"
echo "Layout:       $LAYOUT"
echo "======================================"

# Run the tracker
python -m retracker apps multiview \
    --data_root "$DATA_ROOT" \
    --views $VIEWS \
    --ref_view "$REF_VIEW" \
    --num_points "$NUM_POINTS" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --ckpt_path "$CKPT" \
    --layout "$LAYOUT" \
    --fps "$FPS" \
    $NO_DISPLAY

echo ""
echo "Done! Output saved to: $OUTPUT"

#!/bin/bash
# Smoke test: streaming with brush4.mp4 (global tracking mode)

set -euo pipefail

echo "=========================================="
echo "Streaming Smoke - Global Tracking Mode"
echo "=========================================="
echo ""

cd "$(dirname "$0")/../.."

if ! python -c "import retracker" >/dev/null 2>&1; then
    echo "Error: retracker is not importable in the current Python environment." >&2
    echo "Hint: activate your env and install editable deps:" >&2
    echo "  conda activate retracker_env" >&2
    echo "  python -m pip install -e . -e '.[apps]'" >&2
    exit 1
fi

# Configuration
VIDEO_PATH="${VIDEO_PATH:-./data/demo/teaser_demo/brush4.mp4}"
OUTPUT_PATH="./outputs/streaming/brush4_global.mp4"

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "[smoke] Video not found: $VIDEO_PATH" >&2
    echo "[smoke] Falling back to built-in demo video (may download on first run)..." >&2
    VIDEO_PATH="$(python -c "from retracker.apps.utils import prepare_demo_video; print(prepare_demo_video())")"
fi

# Frame segments to track (optional)
# Format: "start1-end1,start2-end2,..."
# Examples:
#   "0-50"              - Track frames 0 to 50
#   "0-50,100-150"      - Track frames 0-50 and 100-150
#   "50-"               - Track from frame 50 to end
#   ""                  - Track all frames (leave empty or comment out)
FRAME_SEGMENTS="0-50,150-200"

# Query point settings
# For uniform distribution, set grid_size so that grid_size^2 ≈ max_points
# E.g., grid_size=10 gives 100 points, grid_size=15 gives 225 points
GRID_SIZE=10
MAX_POINTS=50

# Visualization options
HIDE_LOW_CONF=false  # Set to true to hide low confidence points

# Build frame segments argument
FRAME_ARGS=""
if [ -n "$FRAME_SEGMENTS" ]; then
    FRAME_ARGS="--frame_segments $FRAME_SEGMENTS"
fi

# Build visualization arguments
VIS_ARGS=""
if [ "$HIDE_LOW_CONF" = true ]; then
    VIS_ARGS="$VIS_ARGS --hide_low_confidence"
fi

# Run streaming smoke with global tracking
python -m retracker apps streaming \
    --source video_file \
    --video_path "$VIDEO_PATH" \
    --target_fps 30 \
    --no_display \
    --record \
    --output_path "$OUTPUT_PATH" \
    --grid_size "$GRID_SIZE" \
    --max_points "$MAX_POINTS" \
    $FRAME_ARGS \
    $VIS_ARGS \
    --verbose

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo "Output: $OUTPUT_PATH"
echo ""
echo "To view the output:"
echo "  ffplay $OUTPUT_PATH"

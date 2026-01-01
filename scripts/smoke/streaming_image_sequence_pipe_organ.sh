#!/bin/bash
# Smoke test: streaming with image sequence (pipe_organ)

set -euo pipefail

echo "=========================================="
echo "Streaming Smoke - Image Sequence Mode"
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
IMAGE_DIR="./data/sfm_demo/pipe_organ"
OUTPUT_PATH="./outputs/streaming/pipe_organ_matching_new.mp4"

# Frame segments to track (optional)
# Format: "start1-end1,start2-end2,..."
# Note: indices are based on sorted image order, not original image IDs
# Examples:
#   "0-3"               - Track first 3 images
#   "0-2,3-5"           - Track images 0-2 and 3-5
#   ""                  - Track all images (leave empty or comment out)
FRAME_SEGMENTS=""

# Image sorting method
# - natural: sorts numbers correctly (1, 2, 10 instead of 1, 10, 2)
# - name: alphabetical sort
# - mtime: sort by modification time
SORT_BY="natural"

# Query strategy
# - sift: SIFT keypoint detection (default for image_sequence)
# - grid: uniform grid points
QUERY_STRATEGY="sift"

# Maximum number of query points (SIFT may detect many, this will sample)
MAX_POINTS=500

# Visualization options
# Plot mode:
#   - pairs: show matching lines between first frame and each subsequent frame (default for image_sequence)
#   - tracks: show trajectories over time (default for video)
PLOT_MODE="pairs"

# Show low confidence points (default: false for image_sequence)
SHOW_LOW_CONF=false

# Build frame segments argument
FRAME_ARGS=""
if [ -n "$FRAME_SEGMENTS" ]; then
    FRAME_ARGS="--frame_segments $FRAME_SEGMENTS"
fi

# Build visualization arguments
VIS_ARGS="--plot_mode $PLOT_MODE"
if [ "$SHOW_LOW_CONF" = true ]; then
    VIS_ARGS="$VIS_ARGS --show_low_confidence"
fi

# Run streaming smoke with image sequence
python -m retracker apps streaming \
    --source image_sequence \
    --image_dir $IMAGE_DIR \
    --sort_by $SORT_BY \
    --query_strategy $QUERY_STRATEGY \
    --max_points $MAX_POINTS \
    --no_realtime \
    --no_display \
    --record \
    --output_path $OUTPUT_PATH \
    --highres \
    --resized_wh 1080 960 \
    --no_matching_lines \
    $FRAME_ARGS \
    $VIS_ARGS \
    --verbose

    # --resized_wh 2432 1824 \
echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo "Output: $OUTPUT_PATH"
echo ""
echo "To view the output:"
echo "  ffplay $OUTPUT_PATH"

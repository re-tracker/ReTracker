#!/bin/bash
# Streaming tracking launcher (records output video).
#
# Prefer the unified CLI:
#   python -m retracker apps streaming ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
SOURCE="camera"
CONFIG=""
VIDEO_PATH=""
OUTPUT_DIR="./outputs/streaming"
DENSE_MATCHING=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            SOURCE="camera"
            shift
            ;;
        --video)
            SOURCE="video_file"
            VIDEO_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dense_matching)
            DENSE_MATCHING=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --camera              Use camera input (default)"
            echo "  --video PATH          Use video file with real-time simulation"
            echo "  --config PATH         Use custom config file"
            echo "  --output_dir DIR      Output directory (default: ./outputs/streaming). Enables video recording."
            echo "  --dense_matching      Enable dense matching output (W*W=49 points per query)"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --camera"
            echo "  $0 --video /path/to/video.mp4"
            echo "  $0 --config retracker/apps/configs/streaming_fast.yaml"
            echo "  $0 --video /path/to/video.mp4 --dense_matching"
            echo "  $0 --video /path/to/video.mp4 --output_dir ./outputs/my_streaming"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
OUTPUT_FILENAME="streaming_output.mp4"
if [ "$SOURCE" = "video_file" ] && [ -n "$VIDEO_PATH" ]; then
    VIDEO_NAME=$(basename "$VIDEO_PATH")
    VIDEO_NAME="${VIDEO_NAME%.*}"
    if [ "$DENSE_MATCHING" -eq 1 ]; then
        OUTPUT_FILENAME="${VIDEO_NAME}_dense.mp4"
    else
        OUTPUT_FILENAME="${VIDEO_NAME}_tracking.mp4"
    fi
fi

cmd=(
    python -m retracker apps streaming
    --output_dir "$OUTPUT_DIR"
    --record
    --output_path "$OUTPUT_DIR/$OUTPUT_FILENAME"
)
if [ "$DENSE_MATCHING" -eq 1 ]; then
    cmd+=(--dense_matching)
fi
if [ -n "$CONFIG" ]; then
    cmd+=(--config "$CONFIG")
else
    cmd+=(--source "$SOURCE")
    if [ "$SOURCE" = "video_file" ] && [ -n "$VIDEO_PATH" ]; then
        cmd+=(--video_path "$VIDEO_PATH")
    fi
fi

echo "Running: ${cmd[*]}"
echo ""

"${cmd[@]}"

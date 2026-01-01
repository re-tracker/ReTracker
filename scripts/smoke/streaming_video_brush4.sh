#!/bin/bash
# Smoke test: streaming with brush4.mp4

set -euo pipefail

cd "$(dirname "$0")/../.."

if ! python -c "import retracker" >/dev/null 2>&1; then
    echo "Error: retracker is not importable in the current Python environment." >&2
    echo "Hint: activate your env and install editable deps:" >&2
    echo "  conda activate retracker_env" >&2
    echo "  python -m pip install -e . -e '.[apps]'" >&2
    exit 1
fi

# Run streaming smoke (no display for server environment)
VIDEO_PATH="${VIDEO_PATH:-./data/demo/teaser_demo/brush4.mp4}"
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "[smoke] Video not found: $VIDEO_PATH" >&2
    echo "[smoke] Falling back to built-in demo video (may download on first run)..." >&2
    VIDEO_PATH="$(python -c "from retracker.apps.utils import prepare_demo_video; print(prepare_demo_video())")"
fi

python -m retracker apps streaming \
    --source video_file \
    --video_path "$VIDEO_PATH" \
    --target_fps 30 \
    --no_display \
    --record \
    --output_path ./outputs/streaming/brush4_output_256x.mp4 \
    --highres \
    --resized_wh 832 624 \
    --grid_size 20 \
    --max_points 200 \
    --verbose
    # --frame_segments 0-50,150-200 \

    # --resized_wh 256 256 \
    # --dense_matching \
echo ""
echo "Streaming smoke completed!"
echo "Output saved to: ./outputs/streaming/brush4_output_256x.mp4"

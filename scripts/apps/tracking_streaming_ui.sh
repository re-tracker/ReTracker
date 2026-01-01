#!/bin/bash
# Interactive streaming launcher script
#
# This script launches the interactive streaming tracking app with UI controls.
#
# UI Controls:
#   - Space: Pause/Resume
#   - K: Add keyframe at current position (adds new query points)
#   - Q/ESC: Quit
#
# When a keyframe is added:
#   1. The current frame is marked as a keyframe
#   2. New query points are generated on that frame
#   3. These points will be tracked in subsequent frames

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
SOURCE="camera"
CONFIG=""
VIDEO_PATH=""
RTSP_URL=""
RTSP_OPTIONS=""
HTTP_URL=""
OUTPUT_DIR="./outputs/interactive_streaming"
DENSE_MATCHING=""
ENABLE_UI=""
KEYFRAME_MODE="auto"
KEYFRAME_INTERVAL=""
KEYFRAME_PARALLAX=""
SHOW_TRACE=""
HEADLESS=""
MAX_DURATION=""
BATCH_MODE=""
CENTER_CROP=""
CROP_RATIO=""
REALTIME_SKIP=""
SPEED_PRESET=""
RESIZED_WH=""
INTERP_SHAPE=""
COMPILE=""

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
        --rtsp)
            SOURCE="rtsp"
            RTSP_URL="$2"
            shift 2
            ;;
        --http)
            SOURCE="http"
            HTTP_URL="$2"
            shift 2
            ;;
        --rtsp_no_threading)
            RTSP_OPTIONS="$RTSP_OPTIONS --rtsp_no_threading"
            shift
            ;;
        --rtsp_gstreamer)
            RTSP_OPTIONS="$RTSP_OPTIONS --rtsp_gstreamer"
            shift
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
            DENSE_MATCHING="--dense_matching"
            shift
            ;;
        --enable_ui)
            ENABLE_UI="--enable_ui"
            shift
            ;;
        --no_ui)
            ENABLE_UI="--no_ui"
            shift
            ;;
        --auto)
            KEYFRAME_MODE="auto"
            shift
            ;;
        --manual)
            KEYFRAME_MODE="manual"
            shift
            ;;
        --keyframe_interval)
            KEYFRAME_INTERVAL="$2"
            shift 2
            ;;
        --keyframe_parallax)
            KEYFRAME_PARALLAX="$2"
            shift 2
            ;;
        --show_trace)
            SHOW_TRACE="--show_trace"
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --max_duration)
            MAX_DURATION="$2"
            shift 2
            ;;
        --batch)
            BATCH_MODE="--batch"
            shift
            ;;
        --center_crop)
            CENTER_CROP="--center_crop"
            shift
            ;;
        --crop_ratio)
            CROP_RATIO="$2"
            shift 2
            ;;
        --realtime_skip)
            REALTIME_SKIP="--realtime_skip"
            shift
            ;;
        --fast)
            SPEED_PRESET="fast"
            shift
            ;;
        --ultra_fast)
            SPEED_PRESET="ultra_fast"
            shift
            ;;
        --quality)
            SPEED_PRESET="quality"
            shift
            ;;
        --resized_wh)
            RESIZED_WH="$2"
            shift 2
            ;;
        --interp_shape)
            INTERP_SHAPE="$2"
            shift 2
            ;;
        --compile)
            COMPILE="--compile"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Interactive Streaming Tracking (UI Controls)"
            echo ""
            echo "Options:"
            echo "  --camera              Use camera input (default)"
            echo "  --video PATH          Use video file with real-time simulation"
            echo "  --rtsp URL            Use RTSP stream (e.g., rtsp://192.168.1.100:554/stream)"
            echo "  --http URL            Use HTTP stream (e.g., http://192.168.1.100:8080/video)"
            echo "                        For IP Webcam apps, try: http://ip:port/video"
            echo "  --rtsp_no_threading   Disable threaded RTSP reading"
            echo "  --rtsp_gstreamer      Use GStreamer backend for RTSP"
            echo "  --config PATH         Use custom config file"
            echo "  --output_dir DIR      Output directory (default: ./outputs/interactive_streaming)"
            echo "  --dense_matching      Enable dense matching output (W*W=49 points per query)"
            echo "  --enable_ui           Enable tkinter UI control panel (popup window)"
            echo "  --no_ui               Disable UI (only use keyboard shortcuts in OpenCV window)"
            echo ""
            echo "Keyframe Mode Options:"
            echo "  --auto                Start in AUTO keyframe mode (default)"
            echo "                        Keyframes created based on parallax, visible ratio, and frame interval"
            echo "  --manual              Start in MANUAL keyframe mode"
            echo "                        Keyframes only created when you press K or click 'Add Keyframe'"
            echo "  --keyframe_interval N Minimum frame interval between auto keyframes (default: 10)"
            echo "  --keyframe_parallax N Minimum parallax (pixels) to trigger auto keyframe (default: 15)"
            echo ""
            echo "Processing Mode:"
            echo "  (default)             Sequential mode for video files, real-time for camera/RTSP"
            echo "  --realtime_skip       Enable real-time frame skipping (skip frames when processing is slow)"
            echo "  --batch               Batch mode - collect all frames then process (slower but more accurate)"
            echo ""
            echo "Visualization Options:"
            echo "  --show_trace          Enable trajectory trace display by default (can toggle with T key)"
            echo "  --headless            Run without display (recording only, for SSH/remote servers)"
            echo "  --max_duration N      Maximum duration in seconds (useful for testing with long videos)"
            echo ""
            echo "Center Crop Options:"
            echo "  --center_crop         Enable center crop (can also toggle in UI)"
            echo "  --crop_ratio N        Crop ratio 0.2-1.0 (e.g., 0.5 = crop to 50% center, default: 1.0)"
            echo ""
            echo "Speed Presets (for faster streaming):"
            echo "  --fast                Use fast preset: 384x288 input, 32 points (~10-15 FPS)"
            echo "  --ultra_fast          Use ultra-fast preset: 320x240 input, 16 points (~15-25 FPS)"
            echo "  --quality             Use quality preset: 512x384 input, 64 points (~3-6 FPS, default)"
            echo ""
            echo "Advanced Speed Options:"
            echo "  --resized_wh W,H      Set input resolution (e.g., 384,288)"
            echo "  --interp_shape H,W    Set model interp shape (e.g., 384,384)"
            echo "  --compile             Enable torch.compile for TROMA blocks (20-30% speedup)"
            echo ""
            echo "  --help                Show this help message"
            echo ""
            echo "UI Controls:"
            echo "  - Space: Pause/Resume streaming"
            echo "  - K: Add keyframe at current position (adds new query points)"
            echo "  - M: Toggle between AUTO and MANUAL keyframe mode"
            echo "  - L: Toggle real-time/sequential frame processing"
            echo "  - P: Start point selection mode (click to add points)"
            echo "  - Enter: Apply selected points and create keyframe"
            echo "  - T: Toggle trajectory trace display"
            echo "  - +/-: Adjust point count"
            echo "  - R: Regenerate query points"
            echo "  - Q: Quit"
            echo "  - ESC: Cancel selection or Quit"
            echo ""
            echo "Examples:"
            echo "  $0 --camera --enable_ui"
            echo "  $0 --video /path/to/video.mp4 --enable_ui --auto"
            echo "  $0 --video /path/to/video.mp4 --manual"
            echo "  $0 --video /path/to/video.mp4 --auto --keyframe_interval 20"
            echo "  $0 --rtsp rtsp://192.168.1.100:554/stream --enable_ui"
            echo "  $0 --http http://192.168.1.100:8080/video --enable_ui  # IP Webcam app"
            echo "  $0 --http http://10.196.149.129:8080/video --enable_ui"
            echo ""
            echo "Fast streaming examples:"
            echo "  $0 --camera --fast --enable_ui                    # Fast camera streaming"
            echo "  $0 --video video.mp4 --fast --enable_ui           # Fast video processing"
            echo "  $0 --camera --ultra_fast --enable_ui              # Ultra-fast for slow hardware"
            echo "  $0 --video video.mp4 --resized_wh 448,336 --compile  # Custom resolution + compile"
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
OUTPUT_FILENAME="interactive_streaming_output.mp4"
if [ "$SOURCE" = "video_file" ] && [ -n "$VIDEO_PATH" ]; then
    VIDEO_NAME=$(basename "$VIDEO_PATH")
    VIDEO_NAME="${VIDEO_NAME%.*}"
    if [ -n "$DENSE_MATCHING" ]; then
        OUTPUT_FILENAME="${VIDEO_NAME}_interactive_dense.mp4"
    else
        OUTPUT_FILENAME="${VIDEO_NAME}_interactive.mp4"
    fi
fi

CMD="python -m retracker.apps.interactive_streaming --output_dir $OUTPUT_DIR --record --output_path $OUTPUT_DIR/$OUTPUT_FILENAME $DENSE_MATCHING $ENABLE_UI --keyframe_mode $KEYFRAME_MODE $SHOW_TRACE $HEADLESS"

# Add optional keyframe parameters
if [ -n "$KEYFRAME_INTERVAL" ]; then
    CMD="$CMD --keyframe_interval $KEYFRAME_INTERVAL"
fi

if [ -n "$KEYFRAME_PARALLAX" ]; then
    CMD="$CMD --keyframe_parallax $KEYFRAME_PARALLAX"
fi

if [ -n "$MAX_DURATION" ]; then
    CMD="$CMD --max_duration $MAX_DURATION"
fi

if [ -n "$BATCH_MODE" ]; then
    CMD="$CMD $BATCH_MODE"
fi

if [ -n "$CENTER_CROP" ]; then
    CMD="$CMD $CENTER_CROP"
fi

if [ -n "$CROP_RATIO" ]; then
    CMD="$CMD --crop_ratio $CROP_RATIO"
fi

if [ -n "$REALTIME_SKIP" ]; then
    CMD="$CMD $REALTIME_SKIP"
fi

# Add speed preset and optimization options
if [ -n "$SPEED_PRESET" ]; then
    CMD="$CMD --speed_preset $SPEED_PRESET"
fi

if [ -n "$RESIZED_WH" ]; then
    # Parse W,H format
    W=$(echo "$RESIZED_WH" | cut -d',' -f1)
    H=$(echo "$RESIZED_WH" | cut -d',' -f2)
    CMD="$CMD --resized_wh $W $H"
fi

if [ -n "$INTERP_SHAPE" ]; then
    # Parse H,W format
    H=$(echo "$INTERP_SHAPE" | cut -d',' -f1)
    W=$(echo "$INTERP_SHAPE" | cut -d',' -f2)
    CMD="$CMD --coarse_resolution $H $W"
fi

if [ -n "$COMPILE" ]; then
    CMD="$CMD $COMPILE"
fi

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
else
    CMD="$CMD --source $SOURCE"
    if [ "$SOURCE" = "video_file" ] && [ -n "$VIDEO_PATH" ]; then
        CMD="$CMD --video_path $VIDEO_PATH"
    elif [ "$SOURCE" = "rtsp" ] && [ -n "$RTSP_URL" ]; then
        CMD="$CMD --rtsp_url $RTSP_URL $RTSP_OPTIONS"
    elif [ "$SOURCE" = "http" ] && [ -n "$HTTP_URL" ]; then
        CMD="$CMD --http_url $HTTP_URL"
    fi
fi

echo "=========================================="
echo "Interactive Streaming Tracking"
echo "=========================================="
echo ""
echo "Keyframe Mode: $KEYFRAME_MODE"
echo ""
echo "UI Controls:"
echo "  - Space: Pause/Resume"
echo "  - K: Add keyframe (adds new query points)"
echo "  - M: Toggle AUTO/MANUAL keyframe mode"
echo "  - P: Start point selection mode (pauses video, click to add points)"
echo "  - Enter: Apply selected points and resume tracking"
echo "  - ESC: Cancel selection or Quit"
echo "  - T: Toggle trajectory trace display"
echo "  - L: Toggle real-time/sequential mode"
echo "  - Q: Quit"
echo ""
echo "Running: $CMD"
echo ""

# Run the command
eval $CMD

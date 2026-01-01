#!/bin/bash
# Offline video tracking launcher (inference).

set -euo pipefail

# Ensure we run from the repo root so `python -m retracker ...` works without an install.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Default values
VIDEO_PATH=""
SEG_PATH=""
PRESET="balanced"
OUTPUT_DIR="./outputs/tracking"
CONFIG=""
CKPT_PATH=""
SAVE_NPZ=""
PROFILE=""
PROFILE_CSV=""
COMPILE=""
COMPILE_MODE="reduce-overhead"
RESIZED_WH=""
VIS_RESIZED_WH=""
ENABLE_HIGHRES=""
COARSE_RESOLUTION=""
MAX_FRAMES=""
GPU_IDX=""
FAST_START=""
# If the user does not pass --video_path, we default to --demo to avoid
# depending on repo-local sample assets.
DEMO=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_path) VIDEO_PATH="$2"; DEMO=""; shift 2 ;;
        --demo) DEMO="--demo"; VIDEO_PATH=""; shift ;;
        --seg_path) SEG_PATH="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
        --resized_wh) RESIZED_WH="--resized_wh $2 $3"; shift 3 ;;
        --vis_resized_wh) VIS_RESIZED_WH="--vis_resized_wh $2 $3"; shift 3 ;;
        --max_frames) MAX_FRAMES="--max_frames $2"; shift 2 ;;
        --gpu_idx) GPU_IDX="--gpu_idx $2"; shift 2 ;;
        --fast_start) FAST_START="--fast_start"; shift ;;
        --save_npz) SAVE_NPZ="--save_npz"; shift ;;
        --profile) PROFILE="--profile"; shift ;;
        --profile_csv) PROFILE_CSV="--profile_csv $2"; shift 2 ;;
        --compile) COMPILE="--compile"; shift ;;
        --compile_mode) COMPILE_MODE="$2"; shift 2 ;;
        --enable_highres_inference) ENABLE_HIGHRES="--enable_highres_inference"; shift ;;
        --coarse_resolution) COARSE_RESOLUTION="--coarse_resolution $2 $3"; shift 3 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --video_path PATH     Input video file"
            echo "  --demo                Use a built-in demo video (downloads/caches if needed)"
            echo "  --seg_path PATH       Segmentation mask file"
            echo "  --preset NAME         Preset config (fast/balanced/high_quality/debug)"
            echo "  --output_dir DIR      Output directory (default: ./outputs/tracking)"
            echo "  --config PATH         Custom YAML config file"
            echo "  --ckpt_path PATH      Model checkpoint path"
            echo "  --resized_wh W H      Resize video for tracking"
            echo "  --vis_resized_wh W H  Resize video for visualization only"
            echo "  --max_frames N        Limit number of frames (for testing)"
            echo "  --gpu_idx IDX         CUDA device index (0-based)"
            echo "  --save_npz            Save results as NPZ file"
            echo "  --profile             Enable profiling for bottleneck analysis"
            echo "  --profile_csv PATH    Export profiling results to CSV file"
            echo "  --compile             Use torch.compile for faster inference"
            echo "  --compile_mode MODE   torch.compile mode (default/reduce-overhead/max-autotune)"
            echo "  --enable_highres_inference  Enable high-resolution inference"
            echo "  --coarse_resolution H W     Coarse/global stage resolution (H, W) for high-res inference"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic usage"
            echo "  $0 --video_path video.mp4 --ckpt_path /path/to/model.ckpt"
            echo ""
            echo "  # Demo video (no local input required; checkpoint still required)"
            echo "  $0 --demo --ckpt_path /path/to/model.ckpt"
            echo ""
            echo "  # With segmentation and preset"
            echo "  $0 --video_path video.mp4 --ckpt_path /path/to/model.ckpt --seg_path mask.png --preset high_quality"
            echo ""
            echo "  # With profiling"
            echo "  $0 --video_path video.mp4 --ckpt_path /path/to/model.ckpt --profile --profile_csv profile.csv"
            echo ""
            echo "  # With torch.compile (faster inference)"
            echo "  $0 --video_path video.mp4 --ckpt_path /path/to/model.ckpt --compile"
            echo ""
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

if [ -z "$VIDEO_PATH" ] && [ -z "$DEMO" ] && [ -z "${CONFIG:-}" ]; then
    DEMO="--demo"
fi

echo "Running video tracking..."
if [ -n "$DEMO" ]; then
    echo "  Video: (demo)"
else
    echo "  Video: $VIDEO_PATH"
fi
echo "  Segmentation: $SEG_PATH"
echo "  Preset: $PRESET"
echo "  Output: $OUTPUT_DIR"
if [ -n "$CKPT_PATH" ]; then
    echo "  Checkpoint: $CKPT_PATH"
fi
if [ -n "$RESIZED_WH" ]; then
    echo "  Tracking resize: ${RESIZED_WH#--resized_wh }"
fi
if [ -n "$VIS_RESIZED_WH" ]; then
    echo "  Visualization resize: ${VIS_RESIZED_WH#--vis_resized_wh }"
fi
if [ -n "$MAX_FRAMES" ]; then
    echo "  Max frames: ${MAX_FRAMES#--max_frames }"
fi
if [ -n "$GPU_IDX" ]; then
    echo "  GPU idx: ${GPU_IDX#--gpu_idx }"
fi
if [ -n "$DEMO" ]; then
    echo "  Demo: Enabled"
fi
if [ -n "$FAST_START" ]; then
    echo "  Fast start: Enabled"
fi
if [ -n "$PROFILE" ]; then
    echo "  Profiling: Enabled"
fi
if [ -n "$ENABLE_HIGHRES" ]; then
    echo "  High-res inference: Enabled"
fi
if [ -n "$COARSE_RESOLUTION" ]; then
    echo "  Coarse resolution: ${COARSE_RESOLUTION#--coarse_resolution }"
fi
echo ""

# Build checkpoint argument if provided
CKPT_ARG=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_ARG="--ckpt_path $CKPT_PATH"
fi

SEG_ARG=""
if [ -n "$SEG_PATH" ]; then
    SEG_ARG="--seg_path $SEG_PATH"
fi

if [ -n "${CONFIG:-}" ]; then
    python -m retracker apps tracking \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        $CKPT_ARG \
        $RESIZED_WH \
        $VIS_RESIZED_WH \
        $MAX_FRAMES \
        $GPU_IDX \
        ${DEMO:+--demo} \
        $FAST_START \
        $SAVE_NPZ \
        $PROFILE \
        $PROFILE_CSV \
        $COMPILE \
        $ENABLE_HIGHRES \
        $COARSE_RESOLUTION \
        --compile_mode "$COMPILE_MODE"
else
    if [ -n "$DEMO" ]; then
        python -m retracker apps tracking \
            --demo \
            $SEG_ARG \
            --preset "$PRESET" \
            --output_dir "$OUTPUT_DIR" \
            $CKPT_ARG \
            $RESIZED_WH \
            $VIS_RESIZED_WH \
            $MAX_FRAMES \
            $GPU_IDX \
            $FAST_START \
            $SAVE_NPZ \
            $PROFILE \
            $PROFILE_CSV \
            $COMPILE \
            $ENABLE_HIGHRES \
            $COARSE_RESOLUTION \
            --compile_mode "$COMPILE_MODE"
    else
        if [ -z "$VIDEO_PATH" ]; then
            echo "Error: --video_path is required (or use --demo)" >&2
            exit 1
        fi
        python -m retracker apps tracking \
            --video_path "$VIDEO_PATH" \
            $SEG_ARG \
            --preset "$PRESET" \
            --output_dir "$OUTPUT_DIR" \
            $CKPT_ARG \
            $RESIZED_WH \
            $VIS_RESIZED_WH \
            $MAX_FRAMES \
            $GPU_IDX \
            $FAST_START \
            $SAVE_NPZ \
            $PROFILE \
            $PROFILE_CSV \
            $COMPILE \
            $ENABLE_HIGHRES \
            $COARSE_RESOLUTION \
            --compile_mode "$COMPILE_MODE"
    fi
fi

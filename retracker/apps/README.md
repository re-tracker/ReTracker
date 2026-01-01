# ReTracker Apps

This folder contains runnable applications/pipelines for tracking, streaming, and multi-view.

## Quick Start

```bash
# 1) Offline tracking (video)
python -m retracker apps tracking --video_path ./data/video.mp4 --ckpt_path weights/retracker_b1.7.ckpt --fast_start --preset balanced

# 2) Streaming tracking (video)
python -m retracker apps streaming --source video_file --video_path ./data/video.mp4 --ckpt_path weights/retracker_b1.7.ckpt --fast_start --record

# 3) Multi-view
python -m retracker apps multiview --data_root ./data/multiview --views 19 25 28 --ref_view 25
```

## App Catalog

### Offline Tracking
- CLI: `python -m retracker apps tracking ...`
- Input: `--video_path` (or `--demo` to download the bundled video)
- Model: `--ckpt_path` is required for real inference; `--fast_start` skips DINOv3 hub weight loading
- Output: `--output_dir` (default: `./outputs/tracking`)
- Notes: presets (`--preset fast|balanced|high_quality|debug`) and YAML config (`--config`) are supported

### Streaming Tracking
- CLI: `python -m retracker apps streaming ...`
- Input: `--source camera|video_file|rtsp|image_sequence` with `--video_path` or `--image_dir`
- Model: `--ckpt_path` is required; `--fast_start` skips DINOv3 hub weight loading
- Output: `--record --output_path <file>` (default output dir: `./outputs/streaming`)
- Dense output: `--dense_matching --dense_level 0|1|2`
- Note: YAML config is supported via `--config`; CLI flags only override when explicitly provided

### Pair Matching (Two Images)
- Module: `python -m retracker.apps.pair_matching`
- Input: `--ref_image` and `--tgt_image`
- Output: `--output` (side-by-side visualization with matching lines)

### Interactive Streaming (UI)
- Module: `python -m retracker.apps.interactive_streaming`
- Input: same as streaming, plus `--enable_ui`
- Output: `--output_dir` (default: `./outputs/interactive_streaming`)
- Speed presets: `--speed_preset fast|ultra_fast|balanced|quality`

### Multi-view Tracking
- CLI: `python -m retracker apps multiview ...`
- Input: `--data_root` with per-view subfolders, `--views 19 25 28`, `--ref_view 25`
- Output: `--output <video.mp4>` (optional)

### Multi-view Triangulation
- Module: `python -m retracker.apps.multiview_triangulation`
- Output: `--output_base` (default: `outputs/multiview_triangulation`)
- Steps: `--step tracking|triangulation|render|all`

## Script Wrappers

- `scripts/apps/tracking_video.sh`: offline video tracking
- `scripts/apps/tracking_streaming.sh`: streaming/online tracking (records output)
- `scripts/apps/tracking_streaming_ui.sh`: interactive streaming
- `scripts/apps/multiview_tracking.sh`: multi-view tracking
- `scripts/apps/multiview_triangulation.sh`: multi-view triangulation pipeline

## Smoke Scripts

These scripts are intentionally opinionated quick checks (they assume you have already activated
your environment):

- `scripts/smoke/streaming_video_brush4.sh`
- `scripts/smoke/streaming_video_brush4_global.sh`
- `scripts/smoke/streaming_image_sequence_pipe_organ.sh`

## Default Output Locations

- Tracking: `./outputs/tracking/`
- Streaming: `./outputs/streaming/`
- Interactive streaming: `./outputs/interactive_streaming/`
- Multi-view triangulation: `./outputs/multiview_triangulation/`

## Requirements

- Tracking/Streaming: PyTorch, OpenCV, NumPy
- Streaming: `mediapy` (fallback video IO)

## Help

```bash
python -m retracker apps tracking --help
python -m retracker apps streaming --help
python -m retracker apps multiview --help
```

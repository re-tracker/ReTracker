#!/usr/bin/env python3
"""
Interactive Streaming Video Tracking

Real-time tracking app with interactive UI controls for:
- Pause/Resume streaming
- Add keyframes manually (to add new tracking points at specific frames)

Usage examples:
    # Camera input with UI controls (default)
    python -m retracker.apps.interactive_streaming --source camera --enable_ui

    # Video file with UI controls
    python -m retracker.apps.interactive_streaming --source video_file --video_path video.mp4 --enable_ui

    # Without UI (keyboard shortcuts still work in OpenCV window)
    python -m retracker.apps.interactive_streaming --source video_file --video_path video.mp4

UI Controls:
    - Space: Pause/Resume
    - K: Add keyframe at current position (adds new query points)
    - M: Toggle Auto/Manual keyframe mode
    - T: Toggle trajectory trace display (default: OFF)
    - Q/ESC: Quit

When a keyframe is added:
    1. The current frame is marked as a keyframe
    2. New query points are generated on that frame
    3. These points will be tracked in subsequent frames
"""

import argparse
import sys

from retracker.apps.config.streaming_config import StreamingConfig
from retracker.apps.runtime.interactive_streaming_pipeline import (
    InteractiveStreamingPipeline,
    KeyframeConfig,
)
from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def parse_frame_segments(segments_str: str) -> list:
    """
    Parse frame segments string into list of (start, end) tuples.
    """
    if not segments_str or segments_str.strip() == '':
        return None

    segments = []
    for part in segments_str.split(','):
        part = part.strip()
        if not part:
            continue

        if '-' not in part:
            raise ValueError(f"Invalid segment format: '{part}'. Expected 'start-end' format.")

        parts = part.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid segment format: '{part}'. Expected 'start-end' format.")

        start_str, end_str = parts
        start_str = start_str.strip()
        if not start_str:
            raise ValueError(f"Invalid segment format: '{part}'. Start frame is required.")
        start = int(start_str)

        end_str = end_str.strip()
        end = int(end_str) if end_str else None

        if end is not None and end <= start:
            raise ValueError(f"Invalid segment: '{part}'. End frame must be greater than start frame.")

        segments.append((start, end))

    segments.sort(key=lambda x: x[0])
    return segments if segments else None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Interactive Streaming Video Tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='''
UI Controls (when enabled):
  - Space: Pause/Resume
  - K: Add keyframe at current position
  - M: Toggle Auto/Manual keyframe mode
  - T: Toggle trajectory trace display (default: OFF)
  - Q/ESC: Quit

When a keyframe is added, new query points are generated on that frame
and tracked in subsequent frames.
'''
    )

    # UI settings
    ui_group = parser.add_argument_group('UI Settings')
    ui_group.add_argument(
        '--enable_ui',
        action='store_true',
        help='Enable interactive UI control panel (tkinter window)'
    )
    ui_group.add_argument(
        '--no_ui',
        action='store_true',
        help='Disable UI (only use keyboard shortcuts in OpenCV window)'
    )
    ui_group.add_argument(
        '--keyframe_mode',
        type=str,
        choices=['auto', 'manual'],
        default='auto',
        help='Initial keyframe mode: auto (creates keyframes based on parallax/interval) or manual (only on user request)'
    )
    ui_group.add_argument(
        '--keyframe_interval',
        type=int,
        default=10,
        help='Minimum frame interval between auto keyframes'
    )
    ui_group.add_argument(
        '--keyframe_parallax',
        type=float,
        default=15.0,
        help='Minimum parallax (pixels) to trigger auto keyframe'
    )
    ui_group.add_argument(
        '--show_trace',
        action='store_true',
        help='Enable trajectory trace display by default (can toggle with T key)'
    )
    ui_group.add_argument(
        '--realtime_skip',
        action='store_true',
        help='Enable real-time frame skipping (skip frames when processing is slow, good for camera/RTSP)'
    )
    ui_group.add_argument(
        '--sequential',
        action='store_true',
        default=True,
        help='Process frames sequentially without skipping (default for video files)'
    )
    ui_group.add_argument(
        '--batch',
        action='store_true',
        help='Use batch mode (collect all frames then process)'
    )

    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    # Source settings
    source_group = parser.add_argument_group('Source Settings')
    source_group.add_argument(
        '--source',
        type=str,
        choices=['camera', 'video_file', 'rtsp', 'http', 'image_sequence'],
        default='camera',
        help='Video source type'
    )
    source_group.add_argument(
        '--camera_id',
        type=int,
        default=0,
        help='Camera device ID'
    )
    source_group.add_argument(
        '--video_path',
        type=str,
        help='Path to video file (for video_file source)'
    )
    source_group.add_argument(
        '--rtsp_url',
        type=str,
        help='RTSP stream URL (for rtsp source). Example: rtsp://user:pass@192.168.1.100:554/stream'
    )
    source_group.add_argument(
        '--http_url',
        type=str,
        help='HTTP stream URL (for http source, e.g., IP Webcam). Example: http://192.168.1.100:8080/video'
    )
    source_group.add_argument(
        '--rtsp_no_threading',
        action='store_true',
        help='Disable threaded RTSP reading (may increase latency but more stable)'
    )
    source_group.add_argument(
        '--rtsp_gstreamer',
        action='store_true',
        help='Use GStreamer backend for RTSP (requires GStreamer installed)'
    )
    source_group.add_argument(
        '--rtsp_timeout',
        type=float,
        default=10.0,
        help='RTSP read timeout in seconds'
    )
    source_group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing images (for image_sequence source)'
    )
    source_group.add_argument(
        '--sort_by',
        type=str,
        choices=['name', 'natural', 'mtime'],
        default='natural',
        help='Sort order for images'
    )
    source_group.add_argument(
        '--no_realtime',
        action='store_true',
        help='Disable real-time simulation for video files'
    )
    source_group.add_argument(
        '--target_fps',
        type=float,
        default=30.0,
        help='Target FPS for real-time simulation'
    )
    source_group.add_argument(
        '--skip_frames',
        type=int,
        default=0,
        help='Skip N frames between processing'
    )
    source_group.add_argument(
        '--frame_segments',
        type=str,
        default=None,
        help='Frame segments to track. Format: "start1-end1,start2-end2,..."'
    )

    # Processing settings
    process_group = parser.add_argument_group('Processing Settings')
    process_group.add_argument(
        '--max_points',
        type=int,
        default=64,
        help='Number of points to track (default: 64 for real-time, can adjust with +/- keys)'
    )

    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument(
        '--ckpt_path',
        type=str,
        help='Path to model checkpoint'
    )
    model_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run on (auto chooses cuda if available, else cpu)'
    )
    model_group.add_argument(
        '--dtype',
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
        default='bf16',
        help='Data type for inference'
    )
    model_group.add_argument(
        '--highres',
        action='store_true',
        help='Enable high-resolution inference'
    )
    model_group.add_argument(
        '--coarse_resolution',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=[512, 512],
        help='Resolution (H, W) for coarse/global stage when using --highres'
    )
    model_group.add_argument(
        '--dense_matching',
        action='store_true',
        help='Enable dense matching output'
    )
    model_group.add_argument(
        '--dense_level',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Refinement level for dense matching'
    )
    model_group.add_argument(
        '--compile',
        action='store_true',
        help='Enable torch.compile for TROMA blocks (20-30%% speedup after warmup)'
    )

    # Speed presets
    speed_group = parser.add_argument_group('Speed Presets')
    speed_group.add_argument(
        '--speed_preset',
        type=str,
        choices=['fast', 'ultra_fast', 'balanced', 'quality'],
        default=None,
        help='Speed preset: fast (10-15 FPS), ultra_fast (15-25 FPS), balanced (6-10 FPS), quality (3-6 FPS)'
    )

    # Query settings
    query_group = parser.add_argument_group('Query Settings')
    query_group.add_argument(
        '--grid_size',
        type=int,
        default=20,
        help='Grid size for query generation'
    )
    query_group.add_argument(
        '--query_strategy',
        type=str,
        choices=['grid', 'segmentation', 'detection', 'custom', 'sift'],
        default=None,
        help='Query generation strategy'
    )

    # Visualization settings
    vis_group = parser.add_argument_group('Visualization Settings')
    vis_group.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no display, recording only). Auto-detected if no DISPLAY.'
    )
    vis_group.add_argument(
        '--no_display',
        action='store_true',
        help='Disable live display window'
    )
    vis_group.add_argument(
        '--record',
        action='store_true',
        help='Record output to video file'
    )
    vis_group.add_argument(
        '--output_path',
        type=str,
        help='Output video path (for recording)'
    )
    vis_group.add_argument(
        '--linewidth',
        type=int,
        default=2,
        help='Line width for trajectories'
    )
    vis_group.add_argument(
        '--trace_length',
        type=int,
        default=5,
        help='Number of frames to leave trace'
    )
    vis_group.add_argument(
        '--max_motion',
        type=float,
        default=50.0,
        help='Maximum motion threshold in pixels'
    )

    # Frame preprocessing
    preprocess_group = parser.add_argument_group('Preprocessing')
    preprocess_group.add_argument(
        '--resized_wh',
        type=int,
        nargs=2,
        metavar=('W', 'H'),
        default=[512, 384],
        help='Resize frames to (Width, Height)'
    )
    preprocess_group.add_argument(
        '--center_crop',
        action='store_true',
        help='Enable center crop (can also toggle in UI)'
    )
    preprocess_group.add_argument(
        '--crop_ratio',
        type=float,
        default=1.0,
        help='Center crop ratio (0.2-1.0, e.g., 0.5 = crop to 50%% center)'
    )

    # General settings
    parser.add_argument(
        '--max_duration',
        type=float,
        help='Maximum duration in seconds'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/interactive_streaming',
        help='Output directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    enable_file_logging()

    # Determine if UI should be enabled
    enable_ui = args.enable_ui and not args.no_ui

    # Load configuration
    if args.config:
        CONSOLE.print(f"[dim]Loading configuration from: {args.config}[/dim]")
        config = StreamingConfig()
    else:
        config = StreamingConfig()

    # Apply speed preset if specified
    if args.speed_preset:
        from retracker.apps.config.streaming_fast_config import STREAMING_PRESETS
        if args.speed_preset in STREAMING_PRESETS:
            preset = STREAMING_PRESETS[args.speed_preset]
            CONSOLE.print(f"[dim]Applying speed preset: {args.speed_preset}[/dim]")
            CONSOLE.print(f"  - {preset['description']}")
            CONSOLE.print(f"  - Expected FPS: {preset['expected_fps']}")

            # Apply preset settings (can be overridden by explicit args)
            config.source.resized_wh = preset['resized_wh']
            config.model.interp_shape = preset['interp_shape']
            config.processing.max_points = preset['max_points']

            # Enable compile for fast presets
            if args.speed_preset in ['fast', 'ultra_fast']:
                config.model.compile = True
                config.model.compile_warmup = True
        else:
            CONSOLE.print(f"[yellow]Warning: Unknown speed preset '{args.speed_preset}'[/yellow]")

    # Apply command-line overrides
    # Source settings
    config.source.source_type = args.source
    config.source.camera_id = args.camera_id
    if args.video_path:
        config.source.video_path = args.video_path
    if args.rtsp_url:
        config.source.rtsp_url = args.rtsp_url
    config.source.rtsp_use_threading = not args.rtsp_no_threading
    config.source.rtsp_use_gstreamer = args.rtsp_gstreamer
    config.source.rtsp_timeout = args.rtsp_timeout
    if args.http_url:
        config.source.http_url = args.http_url
    if args.image_dir:
        config.source.image_dir = args.image_dir
    config.source.sort_by = args.sort_by
    config.source.simulate_realtime = not args.no_realtime
    config.source.target_fps = args.target_fps
    config.source.skip_frames = args.skip_frames
    config.source.resized_wh = tuple(args.resized_wh)

    # Parse frame segments
    if args.frame_segments:
        try:
            config.source.frame_segments = parse_frame_segments(args.frame_segments)
        except ValueError as e:
            CONSOLE.print(f"[red]Error parsing --frame_segments: {e}[/red]")
            sys.exit(1)

    # Processing settings
    config.processing.max_points = args.max_points

    # Model settings
    if args.ckpt_path:
        config.model.ckpt_path = args.ckpt_path
    config.model.device = args.device
    config.model.dtype = args.dtype
    config.model.enable_highres_inference = args.highres
    config.model.coarse_resolution = tuple(args.coarse_resolution)
    config.model.enable_dense_matching = args.dense_matching
    config.model.dense_level = args.dense_level

    # Enable torch.compile if requested (explicit flag or from speed preset)
    if args.compile:
        config.model.compile = True
        config.model.compile_warmup = True

    if args.highres:
        config.model.interp_shape = (args.resized_wh[1], args.resized_wh[0])

    # Query settings
    config.query.grid_size = args.grid_size

    if args.query_strategy is None:
        if args.source == 'image_sequence':
            config.query.strategy = 'sift'
        else:
            config.query.strategy = 'grid'
    else:
        config.query.strategy = args.query_strategy

    # Visualization settings
    if args.headless:
        config.visualization.show_live = False
        enable_ui = False
    else:
        config.visualization.show_live = not args.no_display
    config.visualization.record_output = args.record
    if args.output_path:
        config.visualization.output_path = args.output_path
    config.visualization.linewidth = args.linewidth
    config.visualization.tracks_leave_trace = args.trace_length
    config.visualization.max_motion_threshold = args.max_motion
    config.visualization.plot_mode = 'tracks'  # Always use tracks mode for interactive
    config.visualization.show_low_confidence = True

    # General settings
    config.max_duration = args.max_duration
    config.output.output_dir = args.output_dir
    config.verbose = args.verbose

    # Validate configuration
    if config.source.source_type == 'video_file' and config.source.video_path is None:
        CONSOLE.print("[red]Error: --video_path is required when using video_file source[/red]")
        sys.exit(1)

    if config.source.source_type == 'rtsp' and config.source.rtsp_url is None:
        CONSOLE.print("[red]Error: --rtsp_url is required when using rtsp source[/red]")
        sys.exit(1)

    if config.source.source_type == 'http' and config.source.http_url is None:
        CONSOLE.print("[red]Error: --http_url is required when using http source[/red]")
        sys.exit(1)

    if config.source.source_type == 'image_sequence' and config.source.image_dir is None:
        CONSOLE.print("[red]Error: --image_dir is required when using image_sequence source[/red]")
        sys.exit(1)

    # Create keyframe config
    keyframe_config = KeyframeConfig(
        min_interval=args.keyframe_interval,
        min_parallax=args.keyframe_parallax,
    )

    # Create and run pipeline
    try:
        pipeline = InteractiveStreamingPipeline(
            config,
            enable_ui=enable_ui,
            keyframe_config=keyframe_config
        )

        # Set initial keyframe mode
        pipeline.ui_state.set_keyframe_mode(args.keyframe_mode)

        # Set initial trace display state
        pipeline.ui_state.set_show_trace(args.show_trace)

        # Set initial point count
        pipeline.ui_state.set_num_points(args.max_points)

        # Set initial crop settings
        pipeline.ui_state.set_enable_crop(args.center_crop)
        pipeline.ui_state.set_crop_ratio(args.crop_ratio)

        # Set real-time frame skipping mode
        # Default: OFF for video_file/image_sequence, ON for camera/rtsp/http
        if args.realtime_skip:
            realtime_mode = True
        elif args.source in ['camera', 'rtsp', 'http']:
            realtime_mode = True  # Default ON for live sources
        else:
            realtime_mode = False  # Default OFF for video files
        pipeline.ui_state.set_realtime_mode(realtime_mode)

        # Choose mode: realtime (default) or batch
        if args.batch:
            CONSOLE.print("[dim]Running in BATCH mode (collect all frames, then process)[/dim]")
            pipeline.run()
        else:
            rt_status = "ON (skip frames)" if realtime_mode else "OFF (sequential)"
            CONSOLE.print(f"[dim]Running in STREAMING mode (Real-time: {rt_status})[/dim]")
            pipeline.run_realtime()

    except Exception as e:
        CONSOLE.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

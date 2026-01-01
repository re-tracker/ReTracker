#!/usr/bin/env python3
"""
Streaming Video Tracking (online)

Real-time tracking app with support for:
- Camera input (webcam, USB camera)
- Video file with real-time simulation
- RTSP network streams
- Image sequence (ordered images in a directory)

Usage examples:
    # Camera input (default camera)
    python -m retracker apps streaming --source camera

    # Camera with specific device ID
    python -m retracker apps streaming --source camera --camera_id 0

    # Video file with real-time simulation
    python -m retracker apps streaming --source video_file --video_path video.mp4

    # Video file without real-time simulation (process as fast as possible)
    python -m retracker apps streaming --source video_file --video_path video.mp4 --no_realtime

    # RTSP stream
    python -m retracker apps streaming --source rtsp --rtsp_url rtsp://camera_ip:554/stream

    # Image sequence (ordered images in a directory)
    python -m retracker apps streaming --source image_sequence --image_dir ./data/images

    # Image sequence with frame segments
    python -m retracker apps streaming --source image_sequence --image_dir ./data/images --frame_segments "0-10,20-30"

    # With config file
    python -m retracker apps streaming --config retracker/apps/configs/streaming_camera.yaml

    # Record output
    python -m retracker apps streaming --source camera --record --output_path output.mp4
"""

import argparse
import sys

from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def parse_frame_segments(segments_str: str) -> list:
    """
    Parse frame segments string into list of (start, end) tuples.

    Format: "start1-end1,start2-end2,..."
    Examples:
        "0-50" -> [(0, 50)]
        "0-50,100-150" -> [(0, 50), (100, 150)]
        "0-50,100-150,200-300" -> [(0, 50), (100, 150), (200, 300)]
        "50-" -> [(50, None)]  # From frame 50 to end
        "0-100,200-" -> [(0, 100), (200, None)]

    Args:
        segments_str: Frame segments string

    Returns:
        List of (start_frame, end_frame) tuples. end_frame can be None for "until end".
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

        # Parse start frame
        start_str = start_str.strip()
        if not start_str:
            raise ValueError(f"Invalid segment format: '{part}'. Start frame is required.")
        start = int(start_str)

        # Parse end frame (can be empty for "until end")
        end_str = end_str.strip()
        end = int(end_str) if end_str else None

        # Validate
        if end is not None and end <= start:
            raise ValueError(f"Invalid segment: '{part}'. End frame must be greater than start frame.")

        segments.append((start, end))

    # Sort segments by start frame
    segments.sort(key=lambda x: x[0])

    return segments if segments else None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Streaming video tracking (online)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Press Q or ESC to quit during streaming.'
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
        choices=['camera', 'video_file', 'rtsp', 'image_sequence'],
        default=None,
        help='Video source type'
    )
    source_group.add_argument(
        '--camera_id',
        type=int,
        default=None,
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
        help='RTSP stream URL (for rtsp source)'
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
        default=None,
        help='Sort order for images: name (alphabetical), natural (handles numbers correctly), mtime (modification time)'
    )

    # Real-time simulation toggle. Use tri-state so config files are not overridden
    # by argparse defaults.
    realtime_group = source_group.add_mutually_exclusive_group()
    realtime_group.add_argument(
        '--realtime',
        dest='simulate_realtime',
        action='store_const',
        const=True,
        default=None,
        help='Enable real-time simulation for video files'
    )
    realtime_group.add_argument(
        '--no_realtime',
        dest='simulate_realtime',
        action='store_const',
        const=False,
        default=None,
        help='Disable real-time simulation for video files'
    )
    source_group.add_argument(
        '--target_fps',
        type=float,
        default=None,
        help='Target FPS for real-time simulation'
    )
    source_group.add_argument(
        '--skip_frames',
        type=int,
        default=None,
        help='Skip N frames between processing'
    )
    source_group.add_argument(
        '--frame_segments',
        type=str,
        default=None,
        help='Frame segments to track. Format: "start1-end1,start2-end2,..." '
             'Example: "0-50,100-150,200-300" tracks frames 0-50, 100-150, 200-300. '
             'Use "start-" to track from start to end of video. '
             'If not specified, processes all frames.'
    )

    # Processing settings
    process_group = parser.add_argument_group('Processing Settings')
    process_group.add_argument(
        '--max_points',
        type=int,
        default=None,
        help='Maximum number of points to track'
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
        default=None,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run on (auto chooses cuda if available, else cpu)'
    )
    model_group.add_argument(
        '--dtype',
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
        default=None,
        help='Data type for inference'
    )
    model_group.add_argument(
        '--fast_start',
        action='store_true',
        help=(
            'Faster cold-start by skipping DINOv3 hub weight loading (recommended when using a full '
            'ReTracker checkpoint).'
        ),
    )

    highres_group = model_group.add_mutually_exclusive_group()
    highres_group.add_argument(
        '--highres',
        dest='highres',
        action='store_const',
        const=True,
        default=None,
        help='Enable high-resolution inference. Coarse stage uses 512x512, refinement uses original resolution.'
    )
    highres_group.add_argument(
        '--no_highres',
        dest='highres',
        action='store_const',
        const=False,
        default=None,
        help='Disable high-resolution inference (overrides config file)'
    )
    model_group.add_argument(
        '--coarse_resolution',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='Resolution (H, W) for coarse/global stage when using --highres. Default: 512 512'
    )

    dense_group = model_group.add_mutually_exclusive_group()
    dense_group.add_argument(
        '--dense_matching',
        dest='dense_matching',
        action='store_const',
        const=True,
        default=None,
        help='Enable dense matching output. When enabled, outputs W*W (7*7=49) points per query '
             'representing the dense flow field around each query point.'
    )
    dense_group.add_argument(
        '--no_dense_matching',
        dest='dense_matching',
        action='store_const',
        const=False,
        default=None,
        help='Disable dense matching output (overrides config file)'
    )
    model_group.add_argument(
        '--dense_level',
        type=int,
        default=None,
        choices=[0, 1, 2],
        help='Refinement level for dense matching (0=coarsest, 2=finest). Default: 2'
    )

    # Query settings
    query_group = parser.add_argument_group('Query Settings')
    query_group.add_argument(
        '--grid_size',
        type=int,
        default=None,
        help='Grid size for query generation (used when query_strategy=grid)'
    )
    query_group.add_argument(
        '--query_strategy',
        type=str,
        choices=['grid', 'segmentation', 'detection', 'custom', 'sift'],
        default=None,
        help='Query generation strategy. Default: sift for image_sequence, grid for others'
    )

    # Visualization settings
    vis_group = parser.add_argument_group('Visualization Settings')

    display_group = vis_group.add_mutually_exclusive_group()
    display_group.add_argument(
        '--display',
        dest='show_live',
        action='store_const',
        const=True,
        default=None,
        help='Enable live display window'
    )
    display_group.add_argument(
        '--no_display',
        dest='show_live',
        action='store_const',
        const=False,
        default=None,
        help='Disable live display window'
    )

    record_group = vis_group.add_mutually_exclusive_group()
    record_group.add_argument(
        '--record',
        dest='record_output',
        action='store_const',
        const=True,
        default=None,
        help='Record output to video file'
    )
    record_group.add_argument(
        '--no_record',
        dest='record_output',
        action='store_const',
        const=False,
        default=None,
        help='Disable recording (overrides config file)'
    )
    vis_group.add_argument(
        '--output_path',
        type=str,
        help='Output video path (for recording)'
    )
    vis_group.add_argument(
        '--linewidth',
        type=int,
        default=None,
        help='Line width for trajectories'
    )
    vis_group.add_argument(
        '--trace_length',
        type=int,
        default=None,
        help='Number of frames to leave trace'
    )
    vis_group.add_argument(
        '--hide_low_confidence',
        action='store_true',
        help='Hide low confidence points (vis <= 0.5). Default: True for image_sequence, False for others'
    )
    vis_group.add_argument(
        '--show_low_confidence',
        action='store_true',
        help='Show low confidence points (overrides default hiding for image_sequence)'
    )
    vis_group.add_argument(
        '--plot_mode',
        type=str,
        choices=['tracks', 'pairs'],
        default=None,
        help='Visualization mode: tracks (trajectories over time), pairs (matching lines from first frame). '
             'Default: tracks for video, pairs for image_sequence'
    )
    vis_group.add_argument(
        '--no_matching_lines',
        action='store_true',
        help='Disable drawing matching lines between corresponding points in pairs mode. Only show points.'
    )
    vis_group.add_argument(
        '--max_motion',
        type=float,
        default=None,
        help='Maximum motion threshold in pixels. Skip drawing trajectory when motion exceeds this. '
             'Set to 0 to disable filtering. Default: 50.0'
    )

    # Frame preprocessing
    preprocess_group = parser.add_argument_group('Preprocessing')
    preprocess_group.add_argument(
        '--resized_wh',
        type=int,
        nargs=2,
        metavar=('W', 'H'),
        default=None,
        help='Resize frames to (Width, Height)'
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
        default=None,
        help='Output directory'
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose',
        dest='verbose',
        action='store_const',
        const=True,
        default=None,
        help='Enable verbose output'
    )
    verbosity_group.add_argument(
        '--quiet',
        dest='verbose',
        action='store_const',
        const=False,
        default=None,
        help='Disable verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    enable_file_logging()

    # Import heavy modules lazily so `--help` stays fast and dependency-light.
    from retracker.apps.config.streaming_config import StreamingConfig
    from retracker.apps.runtime.streaming_pipeline import StreamingTrackingPipeline

    # Load configuration
    if args.config:
        CONSOLE.print(f"[dim]Loading configuration from: {args.config}[/dim]")
        config = StreamingConfig.from_yaml(args.config)
    else:
        config = StreamingConfig()

    # Apply command-line overrides
    # Source settings
    if args.source is not None:
        config.source.source_type = args.source
    if args.camera_id is not None:
        config.source.camera_id = args.camera_id
    if args.video_path:
        config.source.video_path = args.video_path
    if args.rtsp_url:
        config.source.rtsp_url = args.rtsp_url
    if args.image_dir:
        config.source.image_dir = args.image_dir
    if args.sort_by is not None:
        config.source.sort_by = args.sort_by
    if args.simulate_realtime is not None:
        config.source.simulate_realtime = args.simulate_realtime
    if args.target_fps is not None:
        config.source.target_fps = args.target_fps
    if args.skip_frames is not None:
        config.source.skip_frames = args.skip_frames
    if args.resized_wh is not None:
        config.source.resized_wh = tuple(args.resized_wh)

    # Parse frame segments
    if args.frame_segments:
        try:
            config.source.frame_segments = parse_frame_segments(args.frame_segments)
        except ValueError as e:
            CONSOLE.print(f"[red]Error parsing --frame_segments: {e}[/red]")
            sys.exit(1)

    # Processing settings
    if args.max_points is not None:
        config.processing.max_points = args.max_points

    # Model settings
    if args.ckpt_path:
        config.model.ckpt_path = args.ckpt_path
    if args.device is not None:
        config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if args.fast_start:
        config.model.fast_start = True
    if args.highres is not None:
        config.model.enable_highres_inference = args.highres
    if args.coarse_resolution is not None:
        config.model.coarse_resolution = tuple(args.coarse_resolution)
    if args.dense_matching is not None:
        config.model.enable_dense_matching = args.dense_matching
    if args.dense_level is not None:
        config.model.dense_level = args.dense_level

    # When highres is enabled, set interp_shape to match input resolution
    # so the engine doesn't resize images before passing to model
    if config.model.enable_highres_inference:
        # resized_wh is (W, H), interp_shape needs (H, W)
        config.model.interp_shape = (config.source.resized_wh[1], config.source.resized_wh[0])

    # Query settings
    if args.grid_size is not None:
        config.query.grid_size = args.grid_size

    # Set default query strategy based on source type
    if args.query_strategy is not None:
        config.query.strategy = args.query_strategy
    elif not args.config:
        if config.source.source_type == 'image_sequence':
            config.query.strategy = 'sift'  # Default to SIFT for image sequence
        else:
            config.query.strategy = 'grid'  # Default to grid for video

    # Visualization settings
    if args.show_live is not None:
        config.visualization.show_live = args.show_live
    if args.record_output is not None:
        config.visualization.record_output = args.record_output
    if args.output_path:
        config.visualization.output_path = args.output_path
    if args.linewidth is not None:
        config.visualization.linewidth = args.linewidth
    if args.trace_length is not None:
        config.visualization.tracks_leave_trace = args.trace_length
    if args.max_motion is not None:
        config.visualization.max_motion_threshold = args.max_motion
    if args.no_matching_lines:
        config.visualization.show_matching_lines = False

    # Set defaults based on source type for image_sequence
    if args.plot_mode is not None:
        config.visualization.plot_mode = args.plot_mode
    elif not args.config:
        # Reasonable defaults when running without a config file.
        if config.source.source_type == 'image_sequence':
            config.visualization.plot_mode = 'pairs'
        else:
            config.visualization.plot_mode = 'tracks'

    # Confidence filtering: only override when flags are explicitly provided.
    if args.hide_low_confidence:
        config.visualization.show_low_confidence = False
    elif args.show_low_confidence:
        config.visualization.show_low_confidence = True
    elif not args.config:
        if config.source.source_type == 'image_sequence':
            config.visualization.show_low_confidence = False
        else:
            config.visualization.show_low_confidence = True

    # General settings
    if args.max_duration is not None:
        config.max_duration = args.max_duration
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.verbose is not None:
        config.verbose = args.verbose

    # Validate configuration
    if config.source.source_type == 'video_file' and config.source.video_path is None:
        CONSOLE.print("[red]Error: --video_path is required when using video_file source[/red]")
        sys.exit(1)

    if config.source.source_type == 'rtsp' and config.source.rtsp_url is None:
        CONSOLE.print("[red]Error: --rtsp_url is required when using rtsp source[/red]")
        sys.exit(1)

    if config.source.source_type == 'image_sequence' and config.source.image_dir is None:
        CONSOLE.print("[red]Error: --image_dir is required when using image_sequence source[/red]")
        sys.exit(1)

    # Create and run pipeline
    try:
        pipeline = StreamingTrackingPipeline(config)
        pipeline.run()

    except Exception as e:
        CONSOLE.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

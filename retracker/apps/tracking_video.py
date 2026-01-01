#!/usr/bin/env python3
"""
Video Tracking App (Offline)

This is the production tracking application (v2 pipeline) with a clean interface.

Usage examples:
    # Basic usage
    python -m retracker apps tracking --video_path video.mp4
    
    # With config file
    python -m retracker apps tracking --config retracker/apps/configs/example.yaml
    
    # With preset
    python -m retracker apps tracking --video_path video.mp4 --preset high_quality
    
    # With segmentation mask
    python -m retracker apps tracking --video_path video.mp4 --seg_path mask.png
    
    # Quick debug mode
    python -m retracker apps tracking --video_path video.mp4 --preset debug
"""

import argparse
import sys

from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Tracking (offline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For more information, see the documentation.",
    )
    
    # Input/Output
    input_group = parser.add_argument_group('Input/Output')
    input_group.add_argument(
        '--video_path', 
        type=str, 
        help='Path to input video file'
    )
    input_group.add_argument(
        '--seg_path', 
        type=str, 
        help='Path to segmentation mask (PNG/JPG)'
    )
    input_group.add_argument(
        '--query_path', 
        type=str, 
        help='Path to custom query points (NPZ/TXT)'
    )
    input_group.add_argument(
        '--output_dir', 
        type=str, 
        help='Output directory for results'
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', 
        type=str, 
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--preset', 
        type=str, 
        choices=['fast', 'balanced', 'high_quality', 'debug'],
        help='Use preset configuration'
    )
    
    # Video processing
    video_group = parser.add_argument_group('Video Processing')
    video_group.add_argument(
        '--resized_wh', 
        type=int, 
        nargs=2, 
        metavar=('W', 'H'),
        help='Resize video to (Width, Height)'
    )
    video_group.add_argument(
        '--max_frames', 
        type=int,
        help='Limit number of frames (for testing)'
    )

    # Visualization
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument(
        '--vis_resized_wh',
        type=int,
        nargs=2,
        metavar=('W', 'H'),
        help='Resize video for visualization output only (tracking still runs at --resized_wh)'
    )
    
    # Query generation
    query_group = parser.add_argument_group('Query Generation')
    query_group.add_argument(
        '--grid_size', 
        type=int,
        help='Grid size for query generation'
    )
    query_group.add_argument(
        '--query_strategy',
        type=str,
        choices=['grid', 'segmentation', 'custom'],
        help='Query generation strategy'
    )
    
    # Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--ckpt_path', 
        type=str,
        help='Path to model checkpoint'
    )
    model_group.add_argument(
        '--device', 
        type=str, 
        help='Device to run on (auto chooses cuda if available, else cpu)'
    )
    model_group.add_argument(
        '--gpu_idx',
        type=int,
        default=None,
        help='CUDA device index to use (0-based). Only applies when running on CUDA.',
    )
    model_group.add_argument(
        '--dtype', 
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
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
    model_group.add_argument(
        '--enable_highres_inference',
        action='store_true',
        help=(
            'Enable high-resolution inference (coarse/global stage uses --coarse_resolution; '
            'refinement runs at the input resolution).'
        ),
    )
    model_group.add_argument(
        '--coarse_resolution',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='Coarse/global stage resolution (H, W) for high-resolution inference (default: 512 512).',
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--save_video', 
        action=argparse.BooleanOptionalAction,
        help='Save visualization video'
    )
    output_group.add_argument(
        '--save_npz', 
        action='store_true',
        help='Save results as NPZ file'
    )
    output_group.add_argument(
        '--save_images', 
        action='store_true',
        help='Save individual frame images'
    )
    
    # Demo mode
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with default video'
    )

    # Profiling
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling for bottleneck analysis'
    )
    parser.add_argument(
        '--profile_csv',
        type=str,
        default=None,
        help='Export profiling results to CSV file'
    )

    # torch.compile
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster inference (20-50%% speedup)'
    )
    parser.add_argument(
        '--compile_mode',
        type=str,
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        help='torch.compile mode (default: reduce-overhead)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    enable_file_logging()

    # Import heavy modules lazily so `--help` stays fast and dependency-light.
    from retracker.apps.config import TrackingConfig, get_preset_config
    from retracker.apps.runtime import TrackingPipeline
    from retracker.apps.utils.video_utils import prepare_demo_video
    
    # Load configuration
    if args.config:
        CONSOLE.print(f"[dim]Loading configuration from: {args.config}[/dim]")
        config = TrackingConfig.from_yaml(args.config)
    elif args.preset:
        CONSOLE.print(f"[dim]Using preset configuration: {args.preset}[/dim]")
        config = get_preset_config(args.preset)
    else:
        config = TrackingConfig()
    
    # Apply command-line overrides
    if args.demo:
        args.video_path = prepare_demo_video()
    
    if args.video_path:
        config.video_path = args.video_path
    if args.seg_path:
        config.seg_path = args.seg_path
    if args.query_path:
        config.query_path = args.query_path
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    
    # Video processing overrides
    if args.resized_wh:
        config.video.resized_wh = tuple(args.resized_wh)
    if args.max_frames:
        config.video.max_frames = args.max_frames
    
    # Query overrides
    if args.grid_size:
        config.query.grid_size = args.grid_size
    if args.query_strategy:
        config.query.strategy = args.query_strategy

    # Model overrides
    if args.ckpt_path:
        config.model.ckpt_path = args.ckpt_path
    if args.device is not None:
        config.model.device = args.device
    if args.gpu_idx is not None:
        import torch

        if config.model.device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_idx)
            else:
                CONSOLE.print(
                    f"[WARN] --gpu_idx {args.gpu_idx} requested, but torch.cuda.is_available() is False; "
                    "falling back to CPU."
                )
    if args.dtype:
        config.model.dtype = args.dtype
    if args.fast_start:
        config.model.fast_start = True
    if args.enable_highres_inference:
        config.model.enable_highres_inference = True
    if args.coarse_resolution is not None:
        config.model.coarse_resolution = tuple(args.coarse_resolution)

    # torch.compile settings (set before creating pipeline so compile happens at init)
    if args.compile:
        config.model.compile = True
        config.model.compile_mode = args.compile_mode
        config.model.compile_warmup = True  # Always warmup when compiling

    # Output overrides
    if args.save_video is not None:
        config.output.save_video = args.save_video
    if args.save_npz:
        config.output.save_npz = True
    if args.save_images:
        config.output.save_images = True
    if args.vis_resized_wh:
        config.visualization.vis_resized_wh = tuple(args.vis_resized_wh)

    # Validate video path
    if config.video_path is None:
        CONSOLE.print("[red]Error: --video_path is required (or use --demo)[/red]")
        sys.exit(1)

    # Create and run pipeline (compile and warmup happen during init if enabled)
    try:
        pipeline = TrackingPipeline(config)

        # Enable profiling if requested
        if args.profile:
            CONSOLE.print("[dim][Profiler] Enabling profiling...[/dim]")
            pipeline.enable_profiling()

        results = pipeline.run()

        # Print summary
        pipeline.print_summary(results)

        # Print profiling results if enabled
        if args.profile:
            CONSOLE.print()
            pipeline.print_profile_summary()

            # Export to CSV if requested
            if args.profile_csv:
                pipeline.export_profile_csv(args.profile_csv)

    except Exception as e:
        CONSOLE.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

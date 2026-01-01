#!/usr/bin/env python3
"""
Multi-view tracking.

Track corresponding points across multiple camera views.

Usage:
    python -m retracker apps multiview \
        --data_root /path/to/multiview_sequence/images \
        --views 19 25 28 \
        --ref_view 25 \
        --num_points 100 \
        --ckpt_path /path/to/retracker_b1.7.ckpt \
        --output multiview_output.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        required=True,
        help='Root directory containing view subdirectories (required)'
    )
    parser.add_argument(
        '--views',
        type=str,
        nargs='+',
        default=['19', '25', '28'],
        help='List of view IDs to track'
    )
    parser.add_argument(
        '--ref_view',
        type=str,
        default='25',
        help='Reference view for point detection'
    )

    # Tracking arguments
    parser.add_argument(
        '--num_points',
        type=int,
        default=400,
        help='Number of points to track (max limit to prevent OOM)'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=0,
        help='Start frame index'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=None,
        help='End frame index (None = all frames)'
    )

    # Model arguments
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        required=True,
        help='Path to model checkpoint (required)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (auto chooses cuda if available, else cpu)'
    )
    parser.add_argument(
        '--tracking_visibility_threshold',
        type=float,
        default=0.1,
        help='Visibility threshold for tracking'
    )
    parser.add_argument(
        '--matching_visibility_threshold',
        type=float,
        default=0.5,
        help='Visibility threshold for matching'
    )

    # Visualization arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Disable live display'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['horizontal', 'vertical', 'grid'],
        default='horizontal',
        help='Output layout'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS'
    )

    # SIFT arguments
    parser.add_argument(
        '--sift_features',
        type=int,
        default=500,
        help='Max SIFT features to detect (0 = all)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Heavy import (kept here so `--help` is fast).
    from retracker.apps.multiview import MultiViewConfig, MultiViewTracker

    # Build configuration
    config = MultiViewConfig(
        data_root=Path(args.data_root),
        view_ids=args.views,
        reference_view=args.ref_view,
        num_points=args.num_points,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    # Model config
    config.model.ckpt_path = args.ckpt_path
    config.model.device = args.device
    config.model.tracking_visibility_threshold = args.tracking_visibility_threshold
    config.model.matching_visibility_threshold = args.matching_visibility_threshold

    # SIFT config
    config.sift.n_features = args.sift_features

    # Visualization config
    config.visualization.show_live = not args.no_display
    config.visualization.save_video = True
    config.visualization.output_path = Path(args.output) if args.output else None
    config.visualization.layout = args.layout
    config.visualization.fps = args.fps

    # Create and run tracker
    tracker = MultiViewTracker(config)
    tracker.run()


if __name__ == '__main__':
    main()

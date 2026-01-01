#!/usr/bin/env python3
"""
Multi-view triangulation pipeline.

Usage:
    # Run full pipeline with 3 views (default)
    python -m retracker.apps.multiview_triangulation

    # 9-view example (uses chain matching by default)
    python -m retracker.apps.multiview_triangulation \\
        --views 01 07 16 19 25 28 37 40 44 \\
        --min_visible_views 4

    # Use star matching (original method, good for few views)
    python -m retracker.apps.multiview_triangulation --matching_strategy star --ref_view 25

    # Disable local feature matching (use propagation mode)
    python -m retracker.apps.multiview_triangulation --no_local_feature_matching

    # Multi-GPU for faster tracking
    python -m retracker.apps.multiview_triangulation --devices cuda:0 cuda:1

    # Run specific step only
    python -m retracker.apps.multiview_triangulation --step tracking
    python -m retracker.apps.multiview_triangulation --step triangulation
    python -m retracker.apps.multiview_triangulation --step render

    # Force rerun all steps
    python -m retracker.apps.multiview_triangulation --force_rerun

Matching strategies:
    - ring: Chain + first-last matching to close the loop (default, best for surrounding cameras)
    - chain: Sequential matching between adjacent views
    - star: All views match to a single reference view (original, good for 3 views)

Local feature matching (--local_feature_matching, default: True):
    - Each view detects its own SIFT/grid features
    - Each view only matches with its adjacent neighbors (no propagation)
    - A track only spans views where real direct matches exist
    - Better for many views as it avoids accumulated matching errors

Output structure:
    outputs/multiview_triangulation/{dataset_name}/
        ├── tracking_{views}_f{start}-{end}.pkl
        ├── triangulation_{views}_f{start}-{end}.pkl
        ├── pointcloud_{views}_f{start}-{end}.mp4
        └── debug/
"""

from pathlib import Path
from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def main():
    import argparse

    enable_file_logging()
    parser = argparse.ArgumentParser(
        description="Multi-view triangulation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/multiview_tracker/0172_05",
        help="Path to data root (dataset name extracted from path)",
    )
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        default=["19", "25", "28"],
        help="View IDs to use (order matters for chain matching)",
    )
    parser.add_argument(
        "--ref_view",
        type=str,
        default="25",
        help="Reference view (only used for star matching strategy)",
    )

    # Tracking arguments
    parser.add_argument("--num_points", type=int, default=400, help="Number of points to track")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame")
    parser.add_argument(
        "--end_frame", type=int, default=None, help="End frame (default: None = all frames)"
    )

    # Multi-GPU arguments
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use for parallel tracking (e.g., cuda:0 cuda:1)",
    )

    # Matching arguments (for cross-view point matching)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence for valid cross-view matches",
    )
    parser.add_argument(
        "--min_visible_views",
        type=int,
        default=2,
        help="Minimum views a point must be visible in (for many views, set lower than total)",
    )
    parser.add_argument(
        "--matching_strategy",
        type=str,
        choices=["star", "chain", "ring"],
        default="ring",
        help="Matching strategy: ring (default, closes loop), chain (sequential), star (all to ref)",
    )
    parser.add_argument(
        "--local_feature_matching",
        action="store_true",
        default=True,
        help="Each view detects own features and matches only with adjacent neighbors (default: True)",
    )
    parser.add_argument(
        "--no_local_feature_matching",
        action="store_false",
        dest="local_feature_matching",
        help="Disable local feature matching, use propagation mode instead",
    )

    # Triangulation arguments
    parser.add_argument(
        "--reproj_threshold", type=float, default=5.0, help="Reprojection error threshold in pixels"
    )
    parser.add_argument(
        "--min_views_triangulation",
        type=int,
        default=2,
        help="Minimum views for triangulation (usually 2)",
    )

    # Outlier filtering arguments
    parser.add_argument(
        "--filter_outliers",
        action="store_true",
        default=True,
        help="Enable outlier filtering (default: True)",
    )
    parser.add_argument(
        "--no_filter_outliers",
        action="store_false",
        dest="filter_outliers",
        help="Disable outlier filtering",
    )
    parser.add_argument(
        "--outlier_std",
        type=float,
        default=2.0,
        help="Outlier threshold: points beyond N std from median are filtered",
    )

    # Rendering arguments
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=1.5,
        help="Camera distance multiplier (>1.0 zooms out for better overview)",
    )
    parser.add_argument(
        "--views_per_cycle",
        type=float,
        default=3.0,
        help="Number of views to traverse per video cycle (smaller = slower camera movement)",
    )

    # Output arguments
    parser.add_argument(
        "--output_base",
        type=str,
        default="outputs/multiview_triangulation",
        help="Base output directory",
    )

    # Execution arguments
    parser.add_argument(
        "--step",
        type=str,
        choices=["tracking", "triangulation", "render", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun all steps even if cached"
    )

    args = parser.parse_args()

    # Import heavy deps only after argparse handles `--help`.
    from retracker.apps.multiview.triangulation_pipeline import (
        TriangulationConfig,
        TriangulationPipeline,
    )

    # Setup paths
    data_root = Path(args.data_root)

    # Parse devices
    devices = tuple(args.devices) if args.devices else None

    # Create config
    config = TriangulationConfig(
        data_root=data_root,
        cameras_path=data_root / "cameras",
        output_base=Path(args.output_base),
        view_ids=args.views,
        reference_view=args.ref_view,
        num_points=args.num_points,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        confidence_threshold=args.confidence_threshold,
        min_visible_views=args.min_visible_views,
        matching_strategy=args.matching_strategy,
        local_feature_matching=args.local_feature_matching,
        reprojection_error_threshold=args.reproj_threshold,
        min_views_for_triangulation=args.min_views_triangulation,
        filter_outliers=args.filter_outliers,
        outlier_std_threshold=args.outlier_std,
        camera_distance_scale=args.camera_distance_scale,
        views_per_cycle=args.views_per_cycle,
        devices=devices,
    )

    # Print config
    CONSOLE.print("\n" + "=" * 60)
    CONSOLE.print("[bold]Configuration:[/bold]")
    CONSOLE.print("=" * 60)
    CONSOLE.print(f"  Dataset: {config.dataset_name}")
    CONSOLE.print(f"  Data root: {config.data_root}")
    CONSOLE.print(f"  Cameras: {config.cameras_path}")
    CONSOLE.print(f"  Output dir: {config.output_dir}")
    CONSOLE.print(f"  Views ({len(config.view_ids)}): {config.view_ids}")
    CONSOLE.print(f"  Num points: {config.num_points}")
    CONSOLE.print(f"  Frame range: {config.start_frame} - {config.end_frame}")
    CONSOLE.print(f"  Devices: {config.devices if config.devices else 'single GPU'}")
    CONSOLE.print(
        f"  Matching: strategy={config.matching_strategy}, local_feature={config.local_feature_matching}, "
        f"conf={config.confidence_threshold}, min_views={config.min_visible_views}"
    )
    CONSOLE.print(
        f"  Triangulation: reproj_threshold={config.reprojection_error_threshold}px, "
        f"min_views={config.min_views_for_triangulation}"
    )
    CONSOLE.print(
        f"  Filter outliers: {config.filter_outliers} (std={config.outlier_std_threshold})"
    )
    CONSOLE.print(
        f"  Render: camera_dist={config.camera_distance_scale}x, views_per_cycle={config.views_per_cycle}"
    )
    CONSOLE.print("=" * 60)

    # Create and run pipeline
    pipeline = TriangulationPipeline(config)

    if args.step == "all":
        pipeline.run(force_rerun_all=args.force_rerun)
    elif args.step == "tracking":
        pipeline.step1_tracking(force_rerun=args.force_rerun)
    elif args.step == "triangulation":
        tracking_results = pipeline.step1_tracking(force_rerun=False)
        pipeline.step2_triangulation(tracking_results, force_rerun=args.force_rerun)
    elif args.step == "render":
        tracking_results = pipeline.step1_tracking(force_rerun=False)
        triangulation_results = pipeline.step2_triangulation(tracking_results, force_rerun=False)
        pipeline.step3_render_video(triangulation_results, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()

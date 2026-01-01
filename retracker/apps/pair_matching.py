#!/usr/bin/env python3
"""
Pair Matching

Match points between two images and save a side-by-side visualization.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from retracker.utils.rich_utils import CONSOLE, enable_file_logging


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - environment-specific (numpy/opencv mismatch)
        CONSOLE.print(
            "Failed to import OpenCV (cv2). This app requires a working OpenCV install.\n"
            "Tip: if you are on NumPy 2.x with `opencv-python<4.10`, install `numpy<2` or upgrade opencv-python.",
            markup=False,
        )
        raise SystemExit(1) from None
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair matching (two images)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ref_image", type=str, required=True, help="Reference image path")
    parser.add_argument("--tgt_image", type=str, required=True, help="Target image path")
    parser.add_argument("--output", type=str, required=True, help="Output visualization path")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to model checkpoint (required)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (auto chooses cuda if available, else cpu)",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--resized_wh", type=int, nargs=2, metavar=("W", "H"), default=[512, 384])
    parser.add_argument("--query_strategy", type=str, choices=["grid", "sift"], default="grid")
    parser.add_argument(
        "--grid_size", type=int, default=20, help="Grid size (grid_size x grid_size)"
    )
    parser.add_argument(
        "--sift_n_features", type=int, default=0, help="SIFT max features (0 = all)"
    )
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--query_batch_size", type=int, default=256)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--tracking_visibility_threshold", type=float, default=0.1)
    parser.add_argument("--matching_visibility_threshold", type=float, default=0.5)
    parser.add_argument("--no_matching_lines", action="store_true")
    return parser.parse_args()


def load_image(path: Path, resized_wh: tuple[int, int]) -> np.ndarray:
    cv2 = _import_cv2()
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if resized_wh is not None:
        img_bgr = cv2.resize(img_bgr, resized_wh, interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def build_video_tensor(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, device: str) -> torch.Tensor:
    video = np.stack([ref_rgb, tgt_rgb], axis=0)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2)  # (T, C, H, W)
    return video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)


def get_point_color(idx: int, total: int) -> tuple[int, int, int]:
    cv2 = _import_cv2()
    hue = int(180 * idx / max(total, 1))
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(map(int, color_bgr))


def draw_pair_matches(
    ref_rgb: np.ndarray,
    tgt_rgb: np.ndarray,
    trajectories: torch.Tensor,
    visibility: torch.Tensor,
    conf_threshold: float,
    draw_lines: bool = True,
) -> tuple[np.ndarray, int]:
    cv2 = _import_cv2()
    h, w = ref_rgb.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = ref_rgb
    canvas[:, w:] = tgt_rgb

    if trajectories is None or visibility is None:
        return canvas, 0

    first_traj = trajectories[0, 0].cpu().numpy()  # (N, 2)
    second_traj = trajectories[0, 1].cpu().numpy()  # (N, 2)
    first_vis = visibility[0, 0].cpu().numpy()
    second_vis = visibility[0, 1].cpu().numpy()

    if first_traj.shape[0] == 0:
        return canvas, 0

    valid_mask = (first_vis > conf_threshold) & (second_vis > conf_threshold)
    num_valid = int(valid_mask.sum())

    for i in range(first_traj.shape[0]):
        if not valid_mask[i]:
            continue
        pt1 = first_traj[i]
        pt2 = second_traj[i]
        if not (
            np.isfinite(pt1[0])
            and np.isfinite(pt1[1])
            and np.isfinite(pt2[0])
            and np.isfinite(pt2[1])
        ):
            continue
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue
        color = get_point_color(i, first_traj.shape[0])
        if draw_lines:
            cv2.line(canvas, (x1, y1), (x2 + w, y2), color, 1)
        cv2.circle(canvas, (x1, y1), 2, color, -1)
        cv2.circle(canvas, (x2 + w, y2), 2, color, -1)

    cv2.putText(
        canvas,
        f"Matches: {num_valid}/{first_traj.shape[0]} (thr={conf_threshold:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return canvas, num_valid


def main() -> None:
    args = parse_args()
    enable_file_logging()
    cv2 = _import_cv2()

    try:
        from retracker.apps.config import ModelConfig, QueryConfig
        from retracker.apps.runtime.query_generator import QueryGeneratorFactory
        from retracker.apps.runtime.tracker import Tracker
    except ModuleNotFoundError:
        CONSOLE.print(
            "Missing runtime dependencies for `retracker.apps.pair_matching`.\n"
            "Tip: install the project dependencies (from repo root): `python -m pip install -e .`",
            markup=False,
        )
        raise SystemExit(1) from None

    ref_path = Path(args.ref_image)
    tgt_path = Path(args.tgt_image)
    if not ref_path.exists():
        CONSOLE.print(f"[red]Error: ref_image not found: {ref_path}[/red]")
        sys.exit(1)
    if not tgt_path.exists():
        CONSOLE.print(f"[red]Error: tgt_image not found: {tgt_path}[/red]")
        sys.exit(1)

    resized_wh = (args.resized_wh[0], args.resized_wh[1])
    ref_rgb = load_image(ref_path, resized_wh)
    tgt_rgb = load_image(tgt_path, resized_wh)

    model_config = ModelConfig(
        ckpt_path=args.ckpt_path,
        device=args.device,
        dtype=args.dtype,
        interp_shape=(512, 512),  # keep fixed to training resolution
        query_batch_size=args.query_batch_size,
    )
    tracker = Tracker(model_config)
    tracker.set_task_mode("matching")
    tracker.set_visibility_thresholds(
        tracking=args.tracking_visibility_threshold,
        matching=args.matching_visibility_threshold,
    )
    if args.max_queries is not None:
        tracker.set_max_queries(args.max_queries)

    query_config = QueryConfig(
        strategy=args.query_strategy,
        grid_size=args.grid_size,
        sift_n_features=args.sift_n_features,
    )
    query_generator = QueryGeneratorFactory.create(query_config)

    video_tensor = build_video_tensor(ref_rgb, tgt_rgb, tracker.config.device)
    queries = query_generator.generate(video_tensor)

    if queries.shape[1] == 0:
        CONSOLE.print(
            "[yellow]No query points generated. Try --query_strategy sift or reduce filters.[/yellow]"
        )
        sys.exit(1)

    trajectories, visibility = tracker.track(video_tensor, queries)
    vis, num_valid = draw_pair_matches(
        ref_rgb,
        tgt_rgb,
        trajectories,
        visibility,
        args.conf_threshold,
        draw_lines=not args.no_matching_lines,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    CONSOLE.print(f"[green]Saved:[/green] {output_path} | matches: {num_valid}/{queries.shape[1]}")


if __name__ == "__main__":
    main()

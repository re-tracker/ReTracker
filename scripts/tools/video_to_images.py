#!/usr/bin/env python3
"""Extract frames from video file with configurable sampling rate."""

import argparse
import cv2
from pathlib import Path
from typing import Optional
import sys


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
    prefix: str = "frame",
    extension: str = "jpg",
    quality: int = 95
) -> int:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every N-th frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)
        prefix: Filename prefix for output images
        extension: Image format extension (jpg, png)
        quality: JPEG quality (0-100, only for jpg)
        
    Returns:
        Number of frames extracted
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Sample rate: every {sample_rate} frame(s)")
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    
    # Set encoding parameters
    if extension.lower() in ['jpg', 'jpeg']:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif extension.lower() == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        encode_params = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should extract this frame
            if frame_count % sample_rate == 0:
                # Generate output filename with zero-padded frame number
                output_filename = f"{prefix}_{extracted_count:06d}.{extension}"
                output_path = output_dir / output_filename
                
                # Save frame
                cv2.imwrite(str(output_path), frame, encode_params)
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    print(f"Extracted {extracted_count} frames...")
                
                # Check max frames limit
                if max_frames is not None and extracted_count >= max_frames:
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Processed: {frame_count} frames")
    print(f"  Extracted: {extracted_count} frames")
    print(f"  Output directory: {output_dir}")
    
    return extracted_count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: video_name without extension)"
    )
    
    parser.add_argument(
        "-s", "--sample_rate",
        type=int,
        default=1,
        help="Extract every N-th frame (1 = all frames, 2 = every other frame, etc.)"
    )
    
    parser.add_argument(
        "-m", "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)"
    )
    
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for output images"
    )
    
    parser.add_argument(
        "-e", "--extension",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Image format"
    )
    
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="JPEG quality (0-100)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        video_path = Path(args.video_path)
        args.output_dir = video_path.parent / video_path.stem
    
    try:
        extract_frames(
            video_path=args.video_path,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames,
            prefix=args.prefix,
            extension=args.extension,
            quality=args.quality
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

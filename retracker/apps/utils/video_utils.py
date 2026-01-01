"""Video processing utilities."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve
import subprocess
from typing import Tuple

from retracker.utils.rich_utils import CONSOLE

_DEMO_VIDEO_URL = "https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4"


def get_assets_dir() -> Path:
    """Return a writable directory for cached app assets.

    ReTracker should not write into its installed package directory (site-packages).
    By default we cache under the user's XDG cache directory (or ~/.cache).

    Override with:
      - RETRACKER_ASSETS_DIR=/custom/path
      - XDG_CACHE_HOME=/custom/cache/root
    """
    override = os.environ.get("RETRACKER_ASSETS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else (Path.home() / ".cache")
    return (cache_root / "retracker" / "assets").resolve()


def prepare_demo_video() -> str:
    """Download (if needed) and return the path to the demo video."""
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_path = assets_dir / Path(_DEMO_VIDEO_URL).name

    if not target_path.exists():
        CONSOLE.print(
            f"[bold yellow]Downloading demo video from {_DEMO_VIDEO_URL} to {assets_dir}"
        )
        try:
            urlretrieve(_DEMO_VIDEO_URL, target_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download demo video: {exc}") from exc
    else:
        CONSOLE.print(f"[bold yellow]Use cached demo video from {assets_dir}")

    return str(target_path)


def detect_video_rotation(video_path: str) -> Tuple[int, bool]:
    """
    Detect video rotation metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (rotation_degrees, needs_180_rotation)
    """
    rotation = 0
    needs_180_rotation = False
    
    # Try to get rotation tag
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.stdout.strip():
            rotation = int(result.stdout.strip())
    except Exception:
        pass
    
    # Check displaymatrix for 180-degree rotation
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'side_data=displaymatrix',
            '-of', 'default=nw=1:nk=1',
            video_path
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        # -65536 in displaymatrix indicates 180-degree rotation
        if '-65536' in result.stdout:
            needs_180_rotation = True
    except Exception:
        pass
    
    return rotation, needs_180_rotation

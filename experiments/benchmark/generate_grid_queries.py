#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Allow running as a script.
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from experiments.benchmark.core.queries import GridQueriesConfig, make_grid_queries  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate grid queries file (t x y per line).")
    p.add_argument("--grid-size", type=int, default=10, help="Grid size S (produces S*S points).")
    p.add_argument("--height", type=int, default=384, help="Frame height H in pixels.")
    p.add_argument("--width", type=int, default=512, help="Frame width W in pixels.")
    p.add_argument("--t", type=int, default=0, help="Query start frame index t (same for all points).")
    p.add_argument("--out", type=str, required=True, help="Output path (.txt).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q = make_grid_queries(GridQueriesConfig(grid_size=args.grid_size, height=args.height, width=args.width, t=args.t))
    out = Path(args.out)
    q.save_txt(out)
    print(f"[Done] Wrote {q.n} queries to: {out}")


if __name__ == "__main__":
    main()


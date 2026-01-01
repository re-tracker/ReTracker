"""Profiler utility for ReTracker inference analysis.

This module provides a lightweight profiler for analyzing inference bottlenecks
in the ReTracker model. It supports:
- CUDA event-based accurate GPU timing
- Hierarchical timing (nested sections)
- Summary statistics with percentages
- CSV export for further analysis

Usage:
    from retracker.utils.profiler import global_profiler

    # Enable profiling
    global_profiler.enable()

    # In your code:
    with global_profiler.record("forward"):
        # ... your code ...
        with global_profiler.record("backbone"):
            # ... nested section ...

    # Print summary
    global_profiler.print_stats()

    # Export to CSV
    global_profiler.export_csv("profile_results.csv")

Environment variable:
    Set RETRACKER_PROFILE=1 to auto-enable profiling
"""
import torch
import time
import os
import numpy as np
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Optional
from contextlib import contextmanager

from retracker.utils.rich_utils import CONSOLE

class CudaProfiler:
    """Enhanced CUDA profiler with hierarchical timing support."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self.records: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[str] = []  # Stack for hierarchy tracking
        self._parent_map: Dict[str, str] = {}  # child -> parent mapping

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        """Enable profiling."""
        self._enabled = True
        self.reset()
        CONSOLE.print("[Profiler] Enabled")

    def disable(self):
        """Disable profiling."""
        self._enabled = False
        CONSOLE.print("[Profiler] Disabled")

    def reset(self):
        """Reset all timing records."""
        self.records.clear()
        self._stack.clear()
        self._parent_map.clear()

    def record(self, name: str):
        """Context manager for timing a code section.

        Args:
            name: Name of the section (e.g., "backbone", "refinement")

        Example:
            with profiler.record("forward"):
                # code to profile
        """
        if not self._enabled:
            return self._DummyContext()
        return self._RecordContext(self, name)

    @contextmanager
    def section(self, name: str):
        """Alternative context manager syntax (alias for record)."""
        ctx = self.record(name)
        ctx.__enter__()
        try:
            yield
        finally:
            ctx.__exit__(None, None, None)

    def decorator(self, name: str = None):
        """Decorator for timing a function.

        Args:
            name: Optional name. Defaults to function name.

        Example:
            @profiler.decorator("my_func")
            def my_func():
                pass
        """
        def real_decorator(func):
            record_name = name if name else func.__name__

            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                with self.record(record_name):
                    return func(*args, **kwargs)

            return wrapper
        return real_decorator

    def print_stats(self, top_n: int = 30):
        """Print profiling statistics summary.

        Args:
            top_n: Number of top sections to display
        """
        if not self._enabled:
            CONSOLE.print("[Profiler] Profiler is disabled.")
            return

        if not self.records:
            CONSOLE.print("[Profiler] No timing data collected.")
            return

        CONSOLE.print("\n" + "=" * 90)
        CONSOLE.print("RETRACKER PROFILER SUMMARY")
        CONSOLE.print("=" * 90)

        # Calculate total time from root sections
        root_total = sum(
            sum(times) for name, times in self.records.items()
            if '/' not in name
        )

        CONSOLE.print(f"Total time (root sections): {root_total:.2f} ms")
        CONSOLE.print("-" * 90)
        CONSOLE.print(
            f"{'Section':<45} {'Calls':>6} {'Total(ms)':>12} {'Mean(ms)':>10} {'Min':>8} {'Max':>8} {'%':>6}"
        )
        CONSOLE.print("-" * 90)

        # Sort by total time descending
        sorted_items = sorted(
            self.records.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )[:top_n]

        for name, times in sorted_items:
            count = len(times)
            total_time = sum(times)
            mean_time = np.mean(times)
            min_time = min(times)
            max_time = max(times)
            pct = (total_time / root_total * 100) if root_total > 0 else 0

            # Indent based on hierarchy depth
            depth = name.count('/')
            indent = "  " * depth
            display_name = indent + name.split('/')[-1]
            if len(display_name) > 43:
                display_name = display_name[:40] + "..."

            CONSOLE.print(
                f"{display_name:<45} {count:>6} {total_time:>12.2f} {mean_time:>10.2f} {min_time:>8.2f} {max_time:>8.2f} {pct:>5.1f}%"
            )

        CONSOLE.print("=" * 90)

        # Print hierarchy breakdown for top-level sections
        CONSOLE.print("\n[Hierarchy Breakdown]")
        self._print_hierarchy()

    def _print_hierarchy(self):
        """Print hierarchical breakdown of timing."""
        # Group by top-level section
        top_level = defaultdict(float)
        children = defaultdict(lambda: defaultdict(float))

        for name, times in self.records.items():
            total = sum(times)
            parts = name.split('/')
            top = parts[0]
            top_level[top] += total

            if len(parts) > 1:
                child = parts[1]
                children[top][child] += total

        # Print each top-level section with its children
        for top, total in sorted(top_level.items(), key=lambda x: x[1], reverse=True):
            CONSOLE.print(f"\n{top}: {total:.2f} ms (100%)")
            if top in children:
                for child, child_total in sorted(children[top].items(), key=lambda x: x[1], reverse=True):
                    pct = (child_total / total * 100) if total > 0 else 0
                    CONSOLE.print(f"  └─ {child}: {child_total:.2f} ms ({pct:.1f}%)")

    def export_csv(self, filepath: str):
        """Export profiling results to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        if not self.records:
            CONSOLE.print("[Profiler] No data to export.")
            return

        with open(filepath, 'w') as f:
            f.write("section,depth,count,total_ms,mean_ms,min_ms,max_ms,std_ms\n")
            for name, times in sorted(self.records.items()):
                depth = name.count('/')
                count = len(times)
                total_ms = sum(times)
                mean_ms = np.mean(times)
                min_ms = min(times)
                max_ms = max(times)
                std_ms = np.std(times) if len(times) > 1 else 0

                f.write(f'"{name}",{depth},{count},{total_ms:.4f},{mean_ms:.4f},{min_ms:.4f},{max_ms:.4f},{std_ms:.4f}\n')

        CONSOLE.print(f"[Profiler] Results exported to {filepath}")

    def get_summary_dict(self) -> Dict[str, Dict]:
        """Get summary as a dictionary for programmatic access."""
        summary = {}
        for name, times in self.records.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times),
                'mean_ms': np.mean(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'std_ms': np.std(times) if len(times) > 1 else 0,
            }
        return summary

    class _RecordContext:
        """Context manager for timing code sections."""

        def __init__(self, profiler: 'CudaProfiler', name: str):
            self.profiler = profiler
            self.base_name = name
            self.full_name = None
            self.start_time = None

        def __enter__(self):
            # Build hierarchical name
            if self.profiler._stack:
                self.full_name = '/'.join(self.profiler._stack) + '/' + self.base_name
            else:
                self.full_name = self.base_name

            self.profiler._stack.append(self.base_name)

            # Synchronize for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Synchronize and record
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start_time) * 1000  # ms

            self.profiler.records[self.full_name].append(elapsed)
            self.profiler._stack.pop()

    class _DummyContext:
        """Dummy context manager when profiler is disabled."""
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# Global singleton instance
global_profiler = CudaProfiler(enabled=False)

# Auto-enable if environment variable is set
if os.environ.get('RETRACKER_PROFILE', '').lower() in ('1', 'true', 'yes'):
    global_profiler.enable()


# Convenience functions
def enable_profiling():
    """Enable the global profiler."""
    global_profiler.enable()


def disable_profiling():
    """Disable the global profiler."""
    global_profiler.disable()


def print_profile_summary():
    """Print summary of global profiler."""
    global_profiler.print_stats()


def export_profile_csv(filepath: str):
    """Export global profiler results to CSV."""
    global_profiler.export_csv(filepath)


def reset_profiler():
    """Reset the global profiler."""
    global_profiler.reset()

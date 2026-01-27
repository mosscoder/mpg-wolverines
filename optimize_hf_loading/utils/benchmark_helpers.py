"""
Benchmark utilities for measuring HuggingFace dataset loading performance.

Provides timing context managers, result formatting, and comparison utilities.
"""

import time
import json
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional
from datetime import datetime


@contextmanager
def timer(name: str = "Operation"):
    """Context manager to time a block of code.

    Usage:
        with timer("load_dataset") as t:
            dataset = load_dataset(...)
        print(f"Time: {t.elapsed:.2f}s")
    """
    class TimerResult:
        def __init__(self):
            self.elapsed = 0.0
            self.start = 0.0
            self.end = 0.0

    result = TimerResult()
    result.start = time.perf_counter()

    try:
        yield result
    finally:
        result.end = time.perf_counter()
        result.elapsed = result.end - result.start
        print(f"{name}: {result.elapsed:.3f}s")


class BenchmarkTracker:
    """Track timing results for multiple operations."""

    def __init__(self, name: str = "Benchmark"):
        self.name = name
        self.timings = {}
        self.metadata = {}
        self.start_time = time.perf_counter()

    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        @contextmanager
        def _timer():
            start = time.perf_counter()
            try:
                yield
            finally:
                elapsed = time.perf_counter() - start
                self.timings[operation_name] = elapsed
                print(f"  {operation_name}: {elapsed:.3f}s")

        return _timer()

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the benchmark results."""
        self.metadata[key] = value

    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results as a dictionary."""
        total_time = time.perf_counter() - self.start_time

        return {
            'name': self.name,
            'timings': self.timings,
            'total_time': total_time,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }

    def save(self, output_path: str):
        """Save benchmark results to JSON file."""
        results = self.get_results()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to: {output_path}")


def compare_timings(baseline_path: str, optimized_path: str):
    """Compare timing results between baseline and optimized versions.

    Args:
        baseline_path: Path to baseline timing JSON
        optimized_path: Path to optimized timing JSON
    """
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    with open(optimized_path, 'r') as f:
        optimized = json.load(f)

    baseline_timings = baseline['timings']
    optimized_timings = optimized['timings']

    print("\n" + "="*70)
    print("TIMING COMPARISON")
    print("="*70)

    # Compare common operations
    common_ops = set(baseline_timings.keys()) & set(optimized_timings.keys())

    if not common_ops:
        print("No common operations found!")
        return

    print(f"{'Operation':<40} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10}")
    print("-"*70)

    for op in sorted(common_ops):
        baseline_time = baseline_timings[op]
        optimized_time = optimized_timings[op]

        if optimized_time > 0:
            speedup = baseline_time / optimized_time
            reduction_pct = ((baseline_time - optimized_time) / baseline_time) * 100

            print(f"{op:<40} {baseline_time:>10.3f}s {optimized_time:>10.3f}s {speedup:>8.2f}x ({reduction_pct:+.1f}%)")

    # Total comparison
    baseline_total = baseline['total_time']
    optimized_total = optimized['total_time']

    if optimized_total > 0:
        total_speedup = baseline_total / optimized_total
        total_reduction_pct = ((baseline_total - optimized_total) / baseline_total) * 100

        print("-"*70)
        print(f"{'TOTAL':<40} {baseline_total:>10.3f}s {optimized_total:>10.3f}s {total_speedup:>8.2f}x ({total_reduction_pct:+.1f}%)")

    print("="*70)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def print_iteration_header(iteration: int, description: str,
                          prev_time: Optional[float] = None,
                          curr_time: Optional[float] = None,
                          baseline_time: Optional[float] = None):
    """Print formatted header for iteration progress."""
    print("\n" + "="*70)
    print(f"ITERATION {iteration}: {description}")
    print("="*70)

    if prev_time is not None and curr_time is not None:
        abs_reduction = prev_time - curr_time
        pct_reduction = (abs_reduction / prev_time) * 100

        print(f"Previous:   {format_time(prev_time)}")
        print(f"Current:    {format_time(curr_time)}")
        print(f"Reduction:  {format_time(abs_reduction)} ({pct_reduction:.1f}%)")

        if baseline_time is not None:
            cumulative_reduction = ((baseline_time - curr_time) / baseline_time) * 100
            print(f"Cumulative: {cumulative_reduction:.1f}% reduction from baseline")

    print("="*70)

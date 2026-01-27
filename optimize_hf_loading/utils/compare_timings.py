#!/usr/bin/env python3
"""
Compare timing results between baseline and optimized versions.

Usage:
    python optimize_hf_loading/utils/compare_timings.py \\
        optimize_hf_loading/results/baseline_timings.json \\
        optimize_hf_loading/results/iteration_01_timings.json
"""

import sys
import json


def compare_timings(baseline_path: str, optimized_path: str):
    """Compare timing results between baseline and optimized versions."""
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    with open(optimized_path, 'r') as f:
        optimized = json.load(f)

    baseline_timings = baseline['timings']
    optimized_timings = optimized['timings']

    print("\n" + "="*80)
    print("TIMING COMPARISON")
    print("="*80)

    # Compare common operations
    common_ops = set(baseline_timings.keys()) & set(optimized_timings.keys())

    if not common_ops:
        print("No common operations found!")
        return

    print(f"{'Operation':<45} {'Baseline':<12} {'Optimized':<12} {'Speedup':<12}")
    print("-"*80)

    total_speedup_sum = 0
    for op in sorted(common_ops):
        baseline_time = baseline_timings[op]
        optimized_time = optimized_timings[op]

        if optimized_time > 0:
            speedup = baseline_time / optimized_time
            reduction_pct = ((baseline_time - optimized_time) / baseline_time) * 100

            print(f"{op:<45} {baseline_time:>10.3f}s {optimized_time:>10.3f}s {speedup:>8.2f}x ({reduction_pct:+.1f}%)")
            total_speedup_sum += speedup

    # Total comparison
    baseline_total = baseline['total_time']
    optimized_total = optimized['total_time']

    if optimized_total > 0:
        total_speedup = baseline_total / optimized_total
        total_reduction_pct = ((baseline_total - optimized_total) / baseline_total) * 100

        print("-"*80)
        print(f"{'TOTAL':<45} {baseline_total:>10.3f}s {optimized_total:>10.3f}s {total_speedup:>8.2f}x ({total_reduction_pct:+.1f}%)")

    print("="*80)

    # Per-config comparison if available
    if 'metadata' in baseline and 'average_per_config_timings' in baseline['metadata']:
        baseline_avg = baseline['metadata']['average_per_config_timings']
        if 'metadata' in optimized and 'average_per_config_timings' in optimized['metadata']:
            optimized_avg = optimized['metadata']['average_per_config_timings']

            print("\nPER-CONFIG AVERAGES:")
            print("-"*80)
            print(f"{'Operation':<45} {'Baseline':<12} {'Optimized':<12} {'Speedup':<12}")
            print("-"*80)

            common_config_ops = set(baseline_avg.keys()) & set(optimized_avg.keys())
            for op in sorted(common_config_ops):
                b_time = baseline_avg[op]
                o_time = optimized_avg[op]
                if o_time > 0:
                    speedup = b_time / o_time
                    reduction_pct = ((b_time - o_time) / b_time) * 100
                    print(f"{op:<45} {b_time:>10.3f}s {o_time:>10.3f}s {speedup:>8.2f}x ({reduction_pct:+.1f}%)")

            print("="*80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_timings.py <baseline_json> <optimized_json>")
        sys.exit(1)

    compare_timings(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""
Iteration 3: Arrow Batch Access

Uses Arrow columnar operations for metadata extraction instead of row-by-row iteration.

Expected improvement: 60-70% cumulative (10-100x faster metadata cache build)

Usage:
    # Local testing (single config, cached dataset)
    python optimize_hf_loading/03_benchmark_arrow.py --local

    # Cluster testing (full job)
    python optimize_hf_loading/03_benchmark_arrow.py --idx 0
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.append('.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimize_hf_loading.utils.benchmark_helpers import BenchmarkTracker, print_iteration_header
from optimize_hf_loading.utils.arrow_cache import build_metadata_cache_arrow, build_id_to_indices_vectorized
from optimize_hf_loading.utils.optimized_filters import (
    filter_training_pool_by_quality_vectorized,
    create_filtered_gallery_dataset_optimized,
    get_rare_individual_indices_vectorized,
)

# Import from the original script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hygiene_sweep",
    os.path.join(os.path.dirname(__file__), '..', 'reid_openset_tnorm', 'scripts', '00_hygiene_sweep.py')
)
hygiene_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene_sweep)

load_reidentification_dataset = hygiene_sweep.load_reidentification_dataset
get_job_combinations = hygiene_sweep.get_job_combinations
load_feasibility_config = hygiene_sweep.load_feasibility_config


def get_cache_dir(local: bool = False):
    """Get appropriate cache directory based on environment."""
    if local:
        return os.path.expanduser("~/.cache/huggingface/wolverines")
    elif os.path.exists("/data/hf_cache"):
        return "/data/hf_cache"
    else:
        return os.path.expanduser("~/.cache/huggingface/wolverines")


def benchmark_arrow(threshold: float, gallery_size: int, seed: int,
                   dataset, config, metadata_cache) -> dict:
    """Benchmark all operations using Arrow-optimized metadata cache.

    Args:
        threshold: Quality threshold for filtering
        gallery_size: Number of gallery samples per individual
        seed: Random seed
        dataset: HuggingFace dataset
        config: Feasibility configuration
        metadata_cache: Cached metadata from build_metadata_cache_arrow()

    Returns:
        Dictionary of timing results
    """
    tracker = BenchmarkTracker(f"arrow_t{threshold}_g{gallery_size}_s{seed}")

    valid_individuals = config.get('valid_individuals', [])

    # Benchmark: Quality filtering (vectorized version)
    with tracker.time_operation("filter_quality_vectorized"):
        ind_id = valid_individuals[0]
        id_to_indices = metadata_cache['id_to_indices']
        all_ind_indices = id_to_indices.get(ind_id, [])
        val_indices = config['validation_indices'][ind_id]['indices']
        train_candidates = list(set(all_ind_indices) - set(val_indices))
        _ = filter_training_pool_by_quality_vectorized(
            metadata_cache['quality_scores'], train_candidates, threshold
        )

    # Benchmark: Rare individuals lookup (vectorized version)
    with tracker.time_operation("get_rare_individuals_vectorized"):
        _ = get_rare_individual_indices_vectorized(
            metadata_cache, valid_individuals, quality_threshold=threshold
        )

    # Benchmark: Complete filtered gallery (optimized version)
    with tracker.time_operation("create_filtered_gallery_optimized"):
        _, _, _, _ = create_filtered_gallery_dataset_optimized(
            dataset, valid_individuals, gallery_size, threshold, seed, config, metadata_cache
        )

    # Add metadata
    tracker.add_metadata("threshold", threshold)
    tracker.add_metadata("gallery_size", gallery_size)
    tracker.add_metadata("seed", seed)
    tracker.add_metadata("num_individuals", len(valid_individuals))
    tracker.add_metadata("dataset_size", len(dataset))

    return tracker.get_results()


def main():
    parser = argparse.ArgumentParser(description='Iteration 3: Arrow batch access benchmark')
    parser.add_argument('--idx', type=int, default=0, help='Job index (0-23)')
    parser.add_argument('--local', action='store_true', help='Run in local mode (single config)')
    parser.add_argument('--output_dir', type=str, default='optimize_hf_loading/results',
                        help='Output directory for results')
    parser.add_argument('--test_vectorized_mapping', action='store_true',
                        help='Test vectorized ID-to-indices mapping')

    args = parser.parse_args()

    # Load previous results for comparison
    baseline_path = os.path.join(args.output_dir, "baseline_timings.json")
    iter2_path = os.path.join(args.output_dir, "iteration_02_timings.json")

    baseline_total = None
    iter2_total = None

    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        baseline_total = baseline['total_time']

    if os.path.exists(iter2_path):
        with open(iter2_path, 'r') as f:
            iter2 = json.load(f)
        iter2_total = iter2['total_time']

    print_iteration_header(3, "Arrow Batch Access",
                          prev_time=iter2_total, baseline_time=baseline_total)

    # Setup cache directory
    cache_dir = get_cache_dir(args.local)
    os.environ["HF_HOME"] = cache_dir
    print(f"Using HuggingFace cache: {cache_dir}")

    # Get configurations to benchmark
    if args.local:
        combinations = [(0.1, 8, 0)]
        print("\nLocal mode: Testing single configuration")
    else:
        combinations = get_job_combinations(args.idx)
        print(f"\nCluster mode: Job {args.idx} with {len(combinations)} configurations")

    if not combinations:
        print(f"No combinations assigned to job {args.idx}")
        return

    # Overall benchmark tracker
    overall_tracker = BenchmarkTracker("iteration_03_overall")

    # Load dataset (one-time cost)
    print("\n--- Loading Dataset ---")
    with overall_tracker.time_operation("load_dataset"):
        dataset = load_reidentification_dataset()

    # Build metadata cache using Arrow columnar access (NEW - should be much faster!)
    print("\n--- Building Metadata Cache (Arrow Columnar Access) ---")
    with overall_tracker.time_operation("build_metadata_cache_arrow"):
        metadata_cache = build_metadata_cache_arrow(dataset)

    # Optional: test vectorized ID mapping
    if args.test_vectorized_mapping:
        print("\n--- Testing Vectorized ID-to-Indices Mapping ---")
        import time
        ids = metadata_cache['ids']

        # Test original version
        start = time.perf_counter()
        id_to_indices_orig = {}
        for idx, ind_id in enumerate(ids):
            if ind_id not in id_to_indices_orig:
                id_to_indices_orig[ind_id] = []
            id_to_indices_orig[ind_id].append(idx)
        time_orig = time.perf_counter() - start

        # Test vectorized version
        start = time.perf_counter()
        id_to_indices_vec = build_id_to_indices_vectorized(ids)
        time_vec = time.perf_counter() - start

        print(f"  Original: {time_orig:.3f}s")
        print(f"  Vectorized: {time_vec:.3f}s")
        print(f"  Speedup: {time_orig/time_vec:.2f}x")

        # Verify they're the same
        assert set(id_to_indices_orig.keys()) == set(id_to_indices_vec.keys())
        for key in id_to_indices_orig:
            assert id_to_indices_orig[key] == id_to_indices_vec[key]
        print("  ✓ Results match!")

    # Load feasibility config
    print("\n--- Loading Feasibility Config ---")
    with overall_tracker.time_operation("load_feasibility_config"):
        config = load_feasibility_config()

    if not config:
        print("Failed to load feasibility config")
        return

    # Benchmark each configuration
    print(f"\n--- Benchmarking {len(combinations)} Configurations ---")
    all_config_results = []

    for i, (threshold, gallery_size, seed) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] threshold={threshold}, gallery_size={gallery_size}, seed={seed}")

        config_results = benchmark_arrow(threshold, gallery_size, seed, dataset, config, metadata_cache)
        all_config_results.append(config_results)

    # Compute aggregate statistics
    if all_config_results:
        avg_timings = {}
        for key in all_config_results[0]['timings'].keys():
            avg_timings[key] = sum(r['timings'][key] for r in all_config_results) / len(all_config_results)

        overall_tracker.add_metadata("num_configs_benchmarked", len(all_config_results))
        overall_tracker.add_metadata("average_per_config_timings", avg_timings)
        overall_tracker.add_metadata("config_results", all_config_results)

    # Save results
    output_path = os.path.join(args.output_dir, "iteration_03_timings.json")
    overall_tracker.save(output_path)

    # Print summary and comparison
    print("\n" + "="*70)
    print("ITERATION 3 SUMMARY")
    print("="*70)
    print(f"Dataset load:             {overall_tracker.timings['load_dataset']:.3f}s (one-time)")
    print(f"Build metadata (Arrow):   {overall_tracker.timings['build_metadata_cache_arrow']:.3f}s (one-time)")
    print(f"Load config:              {overall_tracker.timings['load_feasibility_config']:.3f}s (one-time)")

    if all_config_results:
        avg_timings = overall_tracker.metadata['average_per_config_timings']
        print(f"\nPer-config averages ({len(all_config_results)} configs):")
        print(f"  Filter quality (vectorized):       {avg_timings.get('filter_quality_vectorized', 0):.3f}s")
        print(f"  Get rare individuals (vectorized): {avg_timings.get('get_rare_individuals_vectorized', 0):.3f}s")
        print(f"  Create filtered gallery (optimized): {avg_timings.get('create_filtered_gallery_optimized', 0):.3f}s")

        # Compare to baseline and previous iterations
        results = overall_tracker.get_results()
        curr_total = results['total_time']

        print(f"\n{'='*70}")
        print(f"COMPARISON:")

        if baseline_total:
            abs_reduction = baseline_total - curr_total
            pct_reduction = (abs_reduction / baseline_total) * 100
            print(f"  Baseline:  {baseline_total:.2f}s")
            print(f"  Current:   {curr_total:.2f}s")
            print(f"  Reduction: {abs_reduction:.2f}s ({pct_reduction:.1f}% from baseline)")

        if iter2_total:
            abs_reduction_iter2 = iter2_total - curr_total
            pct_reduction_iter2 = (abs_reduction_iter2 / iter2_total) * 100
            speedup = iter2_total / curr_total if curr_total > 0 else 0

            # Compare cache build times specifically
            iter2_cache_time = 101.69  # From iteration 2 results
            curr_cache_time = overall_tracker.timings['build_metadata_cache_arrow']
            cache_speedup = iter2_cache_time / curr_cache_time if curr_cache_time > 0 else 0

            print(f"\n  Iteration 2: {iter2_total:.2f}s")
            print(f"  Current:     {curr_total:.2f}s")
            print(f"  Reduction:   {abs_reduction_iter2:.2f}s ({pct_reduction_iter2:.1f}% from iteration 2)")
            print(f"  Speedup:     {speedup:.2f}x")
            print(f"\n  Cache build speedup: {cache_speedup:.2f}x ({iter2_cache_time:.2f}s → {curr_cache_time:.2f}s)")

        print(f"{'='*70}")

    print("\nRun comparison tool:")
    print(f"  python optimize_hf_loading/utils/compare_timings.py \\")
    print(f"    {baseline_path} \\")
    print(f"    {output_path}")


if __name__ == "__main__":
    main()

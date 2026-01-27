#!/usr/bin/env python3
"""
Iteration 1: Pre-cache Quality Scores & ID Mappings

Eliminates repeated sequential dataset access by building metadata cache once.

Expected improvement: 15-25% reduction (50-200x speedup on filtering ops)

Usage:
    # Local testing (single config, cached dataset)
    python optimize_hf_loading/01_benchmark_cached.py --local

    # Cluster testing (full job)
    python optimize_hf_loading/01_benchmark_cached.py --idx 0
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.append('.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimize_hf_loading.utils.benchmark_helpers import BenchmarkTracker, print_iteration_header
from optimize_hf_loading.utils.metadata_cache import (
    build_metadata_cache,
    filter_training_pool_by_quality_cached,
    get_rare_individual_indices_cached,
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
create_filtered_gallery_dataset = hygiene_sweep.create_filtered_gallery_dataset
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


def benchmark_cached(threshold: float, gallery_size: int, seed: int,
                    dataset, config, metadata_cache) -> dict:
    """Benchmark all operations using cached metadata.

    Args:
        threshold: Quality threshold for filtering
        gallery_size: Number of gallery samples per individual
        seed: Random seed
        dataset: HuggingFace dataset
        config: Feasibility configuration
        metadata_cache: Cached metadata from build_metadata_cache()

    Returns:
        Dictionary of timing results
    """
    tracker = BenchmarkTracker(f"cached_t{threshold}_g{gallery_size}_s{seed}")

    valid_individuals = config.get('valid_individuals', [])
    quality_cache = metadata_cache['quality_scores']
    ids_cache = metadata_cache['ids']
    id_to_indices = metadata_cache['id_to_indices']

    # Benchmark: Quality filtering (cached version)
    with tracker.time_operation("filter_quality_cached"):
        # Simulate filtering for one individual
        ind_id = valid_individuals[0]
        all_ind_indices = id_to_indices.get(ind_id, [])
        val_indices = config['validation_indices'][ind_id]['indices']
        train_candidates = list(set(all_ind_indices) - set(val_indices))
        _ = filter_training_pool_by_quality_cached(quality_cache, train_candidates, threshold)

    # Benchmark: Rare individuals lookup (cached version)
    with tracker.time_operation("get_rare_individuals_cached"):
        _ = get_rare_individual_indices_cached(
            quality_cache, ids_cache, valid_individuals, id_to_indices, quality_threshold=threshold
        )

    # Benchmark: Complete filtered gallery (still uses original function)
    # Note: We'll optimize this in next iterations
    with tracker.time_operation("create_filtered_gallery_original"):
        _, _, _, _ = create_filtered_gallery_dataset(
            dataset, valid_individuals, gallery_size, threshold, seed, config, id_to_indices
        )

    # Add metadata
    tracker.add_metadata("threshold", threshold)
    tracker.add_metadata("gallery_size", gallery_size)
    tracker.add_metadata("seed", seed)
    tracker.add_metadata("num_individuals", len(valid_individuals))
    tracker.add_metadata("dataset_size", len(dataset))

    return tracker.get_results()


def main():
    parser = argparse.ArgumentParser(description='Iteration 1: Cached metadata benchmark')
    parser.add_argument('--idx', type=int, default=0, help='Job index (0-23)')
    parser.add_argument('--local', action='store_true', help='Run in local mode (single config)')
    parser.add_argument('--output_dir', type=str, default='optimize_hf_loading/results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load baseline for comparison
    baseline_path = os.path.join(args.output_dir, "baseline_timings.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        baseline_total = baseline['total_time']
    else:
        baseline_total = None

    print_iteration_header(1, "Pre-cache Quality Scores & ID Mappings",
                          prev_time=baseline_total)

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
    overall_tracker = BenchmarkTracker("iteration_01_overall")

    # Load dataset (one-time cost)
    print("\n--- Loading Dataset ---")
    with overall_tracker.time_operation("load_dataset"):
        dataset = load_reidentification_dataset()

    # Build metadata cache (NEW - one-time cost)
    print("\n--- Building Metadata Cache ---")
    with overall_tracker.time_operation("build_metadata_cache"):
        metadata_cache = build_metadata_cache(dataset)

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

        config_results = benchmark_cached(threshold, gallery_size, seed, dataset, config, metadata_cache)
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
    output_path = os.path.join(args.output_dir, "iteration_01_timings.json")
    overall_tracker.save(output_path)

    # Print summary and comparison
    print("\n" + "="*70)
    print("ITERATION 1 SUMMARY")
    print("="*70)
    print(f"Dataset load:       {overall_tracker.timings['load_dataset']:.3f}s (one-time)")
    print(f"Build metadata:     {overall_tracker.timings['build_metadata_cache']:.3f}s (one-time, NEW)")
    print(f"Load config:        {overall_tracker.timings['load_feasibility_config']:.3f}s (one-time)")

    if all_config_results:
        avg_timings = overall_tracker.metadata['average_per_config_timings']
        print(f"\nPer-config averages ({len(all_config_results)} configs):")
        print(f"  Filter quality (cached):       {avg_timings.get('filter_quality_cached', 0):.3f}s")
        print(f"  Get rare individuals (cached): {avg_timings.get('get_rare_individuals_cached', 0):.3f}s")
        print(f"  Create filtered gallery:       {avg_timings.get('create_filtered_gallery_original', 0):.3f}s")

        # Compare to baseline
        if baseline_total:
            results = overall_tracker.get_results()
            curr_total = results['total_time']
            abs_reduction = baseline_total - curr_total
            pct_reduction = (abs_reduction / baseline_total) * 100

            print(f"\n{'='*70}")
            print(f"COMPARISON TO BASELINE:")
            print(f"  Baseline:  {baseline_total:.2f}s")
            print(f"  Current:   {curr_total:.2f}s")
            print(f"  Reduction: {abs_reduction:.2f}s ({pct_reduction:.1f}%)")
            print(f"{'='*70}")

    print("\nRun comparison tool:")
    print(f"  python optimize_hf_loading/utils/compare_timings.py \\")
    print(f"    {baseline_path} \\")
    print(f"    {output_path}")


if __name__ == "__main__":
    main()

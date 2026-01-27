#!/usr/bin/env python3
"""
Iteration 0: Baseline Benchmarking

Establishes performance baseline for HuggingFace dataset loading operations.
Measures time for:
1. Dataset load: load_reidentification_dataset()
2. ID mapping: build_id_to_indices()
3. Quality filtering: filter_training_pool_by_quality()
4. Rare individuals: get_rare_individual_indices()
5. Complete workflow: create_filtered_gallery_dataset()

Usage:
    # Local testing (single config, cached dataset)
    python optimize_hf_loading/00_benchmark_baseline.py --local

    # Cluster testing (full job)
    python optimize_hf_loading/00_benchmark_baseline.py --idx 0
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.append('.')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from optimize_hf_loading.utils.benchmark_helpers import BenchmarkTracker, print_iteration_header

# Import functions from the target script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hygiene_sweep",
    os.path.join(os.path.dirname(__file__), '..', 'reid_openset_tnorm', 'scripts', '00_hygiene_sweep.py')
)
hygiene_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene_sweep)

load_reidentification_dataset = hygiene_sweep.load_reidentification_dataset
build_id_to_indices = hygiene_sweep.build_id_to_indices
filter_training_pool_by_quality = hygiene_sweep.filter_training_pool_by_quality
get_rare_individual_indices = hygiene_sweep.get_rare_individual_indices
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


def benchmark_baseline(threshold: float, gallery_size: int, seed: int, dataset, config, id_to_indices):
    """Benchmark all operations for a single configuration.

    Args:
        threshold: Quality threshold for filtering
        gallery_size: Number of gallery samples per individual
        seed: Random seed
        dataset: HuggingFace dataset
        config: Feasibility configuration
        id_to_indices: Precomputed ID to indices mapping

    Returns:
        Dictionary of timing results
    """
    tracker = BenchmarkTracker(f"baseline_t{threshold}_g{gallery_size}_s{seed}")

    valid_individuals = config.get('valid_individuals', [])

    # Benchmark: Quality filtering (called once per config)
    with tracker.time_operation("filter_quality_single_call"):
        # Simulate filtering for one individual
        ind_id = valid_individuals[0]
        all_ind_indices = id_to_indices.get(ind_id, [])
        val_indices = config['validation_indices'][ind_id]['indices']
        train_candidates = list(set(all_ind_indices) - set(val_indices))
        _ = filter_training_pool_by_quality(dataset, train_candidates, threshold)

    # Benchmark: Rare individuals lookup (called once per config)
    with tracker.time_operation("get_rare_individuals"):
        _ = get_rare_individual_indices(dataset, valid_individuals, id_to_indices, quality_threshold=threshold)

    # Benchmark: Complete filtered gallery creation
    with tracker.time_operation("create_filtered_gallery_complete"):
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
    parser = argparse.ArgumentParser(description='Baseline HuggingFace loading benchmark')
    parser.add_argument('--idx', type=int, default=0, help='Job index (0-23)')
    parser.add_argument('--local', action='store_true', help='Run in local mode (single config, subset data)')
    parser.add_argument('--output_dir', type=str, default='optimize_hf_loading/results',
                        help='Output directory for results')

    args = parser.parse_args()

    print_iteration_header(0, "Baseline Benchmarking")

    # Setup cache directory
    cache_dir = get_cache_dir(args.local)
    os.environ["HF_HOME"] = cache_dir
    print(f"Using HuggingFace cache: {cache_dir}")

    # Get configurations to benchmark
    if args.local:
        # Local: single config for fast testing
        combinations = [(0.1, 8, 0)]
        print("\nLocal mode: Testing single configuration")
    else:
        # Cluster: all configs for this job
        combinations = get_job_combinations(args.idx)
        print(f"\nCluster mode: Job {args.idx} with {len(combinations)} configurations")

    if not combinations:
        print(f"No combinations assigned to job {args.idx}")
        return

    # Overall benchmark tracker
    overall_tracker = BenchmarkTracker("baseline_overall")

    # Load dataset (one-time cost)
    print("\n--- Loading Dataset ---")
    with overall_tracker.time_operation("load_dataset"):
        dataset = load_reidentification_dataset()

    # Build ID mapping (one-time cost)
    print("\n--- Building ID Mapping ---")
    with overall_tracker.time_operation("build_id_to_indices"):
        id_to_indices = build_id_to_indices(dataset)

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

        config_results = benchmark_baseline(threshold, gallery_size, seed, dataset, config, id_to_indices)
        all_config_results.append(config_results)

    # Compute aggregate statistics
    if all_config_results:
        # Average timing across all configs
        avg_timings = {}
        for key in all_config_results[0]['timings'].keys():
            avg_timings[key] = sum(r['timings'][key] for r in all_config_results) / len(all_config_results)

        overall_tracker.add_metadata("num_configs_benchmarked", len(all_config_results))
        overall_tracker.add_metadata("average_per_config_timings", avg_timings)
        overall_tracker.add_metadata("config_results", all_config_results)

    # Save overall results
    output_path = os.path.join(args.output_dir, "baseline_timings.json")
    overall_tracker.save(output_path)

    # Print summary
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"Dataset load:       {overall_tracker.timings['load_dataset']:.3f}s (one-time)")
    print(f"Build ID mapping:   {overall_tracker.timings['build_id_to_indices']:.3f}s (one-time)")
    print(f"Load config:        {overall_tracker.timings['load_feasibility_config']:.3f}s (one-time)")

    if all_config_results:
        avg_timings = overall_tracker.metadata['average_per_config_timings']
        print(f"\nPer-config averages ({len(all_config_results)} configs):")
        print(f"  Filter quality:         {avg_timings.get('filter_quality_single_call', 0):.3f}s")
        print(f"  Get rare individuals:   {avg_timings.get('get_rare_individuals', 0):.3f}s")
        print(f"  Create filtered gallery: {avg_timings.get('create_filtered_gallery_complete', 0):.3f}s")

        # Estimate total experiment time (288 configs)
        avg_per_config = sum(avg_timings.values())
        total_configs = 288
        estimated_total = (overall_tracker.timings['load_dataset'] +
                          overall_tracker.timings['build_id_to_indices'] +
                          avg_per_config * total_configs)

        print(f"\nEstimated total for 288 configs: {estimated_total/60:.1f} minutes")

    print("="*70)


if __name__ == "__main__":
    main()

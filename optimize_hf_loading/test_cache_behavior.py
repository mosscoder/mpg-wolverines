#!/usr/bin/env python3
"""
Test script to verify cache behavior across different scenarios.

Usage:
    # Test cache location detection
    python optimize_hf_loading/test_cache_behavior.py --show_paths

    # Test cache build and load
    python optimize_hf_loading/test_cache_behavior.py --test_build_load

    # Simulate cluster environment
    python optimize_hf_loading/test_cache_behavior.py --simulate_cluster
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimize_hf_loading.utils.persistent_cache import get_cache_path, get_cache_info
from optimize_hf_loading.utils.arrow_cache import build_metadata_cache_arrow
from optimize_hf_loading.utils.persistent_cache import load_or_build_metadata_cache, clear_metadata_cache

# Import dataset loader
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hygiene_sweep",
    "reid_openset_tnorm/scripts/00_hygiene_sweep.py"
)
hygiene_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene_sweep)
load_reidentification_dataset = hygiene_sweep.load_reidentification_dataset


def show_cache_paths():
    """Show where cache will be stored in different environments."""
    print("="*70)
    print("CACHE PATH DETECTION")
    print("="*70)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()

    # Test different scenarios
    scenarios = [
        (None, "Auto-detect (current environment)"),
        ("/data/hf_cache", "Cluster path (simulated)"),
        (os.path.expanduser("~/.cache/huggingface/wolverines"), "Local path"),
    ]

    for cache_dir, description in scenarios:
        print(f"\n{description}:")
        print(f"  cache_dir input: {cache_dir}")

        if cache_dir and not os.path.exists(cache_dir):
            print(f"  ⚠ Directory doesn't exist (would use default)")
            cache_dir = None

        cache_path = get_cache_path(dataset, cache_dir)
        print(f"  Resolved path: {cache_path}")
        print(f"  Directory exists: {os.path.exists(os.path.dirname(cache_path))}")

        cache_info = get_cache_info(dataset, cache_dir)
        if cache_info['exists']:
            print(f"  Cache file exists: Yes ({cache_info['size_mb']:.2f} MB)")
        else:
            print(f"  Cache file exists: No (will be created on first use)")


def test_build_and_load():
    """Test building and loading cache."""
    print("="*70)
    print("CACHE BUILD AND LOAD TEST")
    print("="*70)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()

    # Get cache path
    cache_dir = os.path.expanduser("~/.cache/huggingface/wolverines")
    cache_path = get_cache_path(dataset, cache_dir)
    print(f"\nCache path: {cache_path}")

    # Test 1: Clear cache and build
    print("\n--- Test 1: Fresh Build ---")
    clear_metadata_cache(dataset, cache_dir)

    import time
    start = time.perf_counter()
    metadata_cache = load_or_build_metadata_cache(
        dataset,
        build_metadata_cache_arrow,
        cache_dir=cache_dir
    )
    build_time = time.perf_counter() - start
    print(f"Build time: {build_time:.3f}s")

    # Test 2: Load from disk
    print("\n--- Test 2: Load from Disk ---")
    start = time.perf_counter()
    metadata_cache = load_or_build_metadata_cache(
        dataset,
        build_metadata_cache_arrow,
        cache_dir=cache_dir
    )
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")

    # Test 3: Verify speedup
    print("\n--- Test 3: Speedup ---")
    speedup = build_time / load_time if load_time > 0 else 0
    print(f"Speedup: {speedup:.1f}x faster")

    if speedup > 5:
        print("✅ PASS: Disk cache is significantly faster")
    else:
        print("⚠ WARNING: Disk cache speedup lower than expected")

    # Test 4: Verify cache contents
    print("\n--- Test 4: Cache Validation ---")
    print(f"Dataset size: {metadata_cache['dataset_size']}")
    print(f"Unique individuals: {len(metadata_cache['id_to_indices'])}")
    print(f"Quality scores shape: {metadata_cache['quality_scores'].shape}")

    if metadata_cache['dataset_size'] == len(dataset):
        print("✅ PASS: Cache contains correct dataset size")
    else:
        print("❌ FAIL: Cache size mismatch")


def simulate_cluster():
    """Simulate cluster environment with shared cache."""
    print("="*70)
    print("CLUSTER SIMULATION")
    print("="*70)

    # Create simulated cluster cache directory
    sim_cache_dir = "/tmp/cluster_cache_sim"
    os.makedirs(sim_cache_dir, exist_ok=True)

    print(f"\nSimulated cluster cache: {sim_cache_dir}")
    print("(In real cluster: /data/hf_cache/metadata_cache/)\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_reidentification_dataset()

    # Clear cache
    cache_path = get_cache_path(dataset, sim_cache_dir)
    if os.path.exists(cache_path):
        os.remove(cache_path)
    print(f"Cleared cache: {cache_path}\n")

    import time

    # Simulate Job 0 (builds cache)
    print("--- Simulated Job 0 (first job) ---")
    start = time.perf_counter()
    metadata_cache = load_or_build_metadata_cache(
        dataset,
        build_metadata_cache_arrow,
        cache_dir=sim_cache_dir
    )
    job0_time = time.perf_counter() - start
    print(f"Job 0 time: {job0_time:.3f}s")
    print(f"Cache saved to: {cache_path}\n")

    # Simulate Job 1-3 (load cache)
    for job_id in range(1, 4):
        print(f"--- Simulated Job {job_id} ---")
        start = time.perf_counter()
        metadata_cache = load_or_build_metadata_cache(
            dataset,
            build_metadata_cache_arrow,
            cache_dir=sim_cache_dir
        )
        job_time = time.perf_counter() - start
        print(f"Job {job_id} time: {job_time:.3f}s")
        speedup = job0_time / job_time if job_time > 0 else 0
        print(f"Speedup vs Job 0: {speedup:.1f}x\n")

    # Summary
    print("="*70)
    print("CLUSTER SIMULATION SUMMARY")
    print("="*70)
    print(f"Job 0 (builds):  {job0_time:.3f}s")
    print(f"Jobs 1-3 (load): ~{job_time:.3f}s each")
    print(f"\nIn full cluster (24 jobs):")
    print(f"  Job 0:    {job0_time:.3f}s (builds cache)")
    print(f"  Jobs 1-23: {job_time:.3f}s each (load cache)")
    print(f"  Savings:   {(job0_time - job_time) * 23:.3f}s total")

    # Clean up
    print(f"\nCleaning up: {sim_cache_dir}")
    shutil.rmtree(sim_cache_dir)


def main():
    parser = argparse.ArgumentParser(description='Test cache behavior')
    parser.add_argument('--show_paths', action='store_true',
                        help='Show cache paths in different environments')
    parser.add_argument('--test_build_load', action='store_true',
                        help='Test building and loading cache')
    parser.add_argument('--simulate_cluster', action='store_true',
                        help='Simulate cluster with shared cache')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')

    args = parser.parse_args()

    if args.all or not any([args.show_paths, args.test_build_load, args.simulate_cluster]):
        # Run all tests if no specific test selected
        args.show_paths = True
        args.test_build_load = True
        args.simulate_cluster = True

    if args.show_paths:
        show_cache_paths()
        print()

    if args.test_build_load:
        test_build_and_load()
        print()

    if args.simulate_cluster:
        simulate_cluster()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze gallery/query quality counts by config without running training.
Outputs counts of HQ/LQ samples for each (samples_per_class, seed) combo.

HQ (High Quality): pelage_score >= 0.5
LQ (Low Quality): pelage_score < 0.5
"""

import os
import sys
import json
import random
from collections import defaultdict

# Add project root to path
sys.path.append('.')

# Config
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
THRESHOLD = 0.5
MIN_P = 5


def build_id_to_indices(dataset):
    """Build individual ID to dataset indices mapping."""
    print("Building ID to indices mapping...")
    id_to_indices = {}
    for idx, sample in enumerate(dataset):
        ind_id = sample['id']
        if ind_id not in id_to_indices:
            id_to_indices[ind_id] = []
        id_to_indices[ind_id].append(idx)
    print(f"  Mapped {len(id_to_indices)} individuals")
    return id_to_indices


def compute_counts_for_config(samples_per_class, seed, config, dataset, id_to_indices):
    """
    Compute HQ/LQ counts for gallery and queries for a given config.

    Returns dict with query/gallery counts or None if config not feasible.
    """
    # Determine feasible individuals for this sample_size
    valid_individuals = config['valid_individuals']
    feasible = []

    for ind_id in valid_individuals:
        training_compat = config['training_compatibility'][ind_id]
        compatible_sizes = training_compat['threshold_compatibility']['threshold_0.00']['compatible_training_sizes']
        if samples_per_class in compatible_sizes:
            feasible.append(ind_id)

    if len(feasible) < MIN_P:
        return None

    # Set seed for reproducible train sampling
    random.seed(seed)

    query_hq, query_lq = 0, 0
    gallery_hq, gallery_lq = 0, 0

    # Per-individual breakdown
    per_individual = {}

    for ind_id in feasible:
        val_indices = set(config['validation_indices'][ind_id]['indices'])
        all_indices = set(id_to_indices.get(ind_id, []))
        train_candidates = list(all_indices - val_indices)

        # Sample from training candidates (seed-dependent)
        random.shuffle(train_candidates)
        sampled_train = train_candidates[:samples_per_class]

        ind_query_hq, ind_query_lq = 0, 0
        ind_gallery_hq, ind_gallery_lq = 0, 0

        # Count queries by quality (static per individual)
        for idx in val_indices:
            if dataset[idx]['pelage_score'] >= THRESHOLD:
                query_hq += 1
                ind_query_hq += 1
            else:
                query_lq += 1
                ind_query_lq += 1

        # Count gallery by quality (seed-dependent)
        for idx in sampled_train:
            if dataset[idx]['pelage_score'] >= THRESHOLD:
                gallery_hq += 1
                ind_gallery_hq += 1
            else:
                gallery_lq += 1
                ind_gallery_lq += 1

        per_individual[ind_id] = {
            'query': {'HQ': ind_query_hq, 'LQ': ind_query_lq},
            'gallery': {'HQ': ind_gallery_hq, 'LQ': ind_gallery_lq}
        }

    return {
        'individuals': feasible,
        'query': {'HQ': query_hq, 'LQ': query_lq, 'total': query_hq + query_lq},
        'gallery': {'HQ': gallery_hq, 'LQ': gallery_lq, 'total': gallery_hq + gallery_lq},
        'per_individual': per_individual
    }


def main():
    from datasets import load_dataset

    print("=" * 60)
    print("Gallery/Query Quality Counts Analysis")
    print("=" * 60)

    # Load config
    config_path = 'individual_id/results/feasible_individuals.json'
    print(f"\nLoading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load dataset
    print("Loading wolverines dataset...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    print(f"  Loaded {len(dataset)} samples")

    # Build index mapping
    id_to_indices = build_id_to_indices(dataset)

    # Compute counts for all configs
    results = []

    print("\n" + "=" * 60)
    print("Computing counts by (samples_per_class, seed)...")
    print("=" * 60)

    for samples in SAMPLE_SIZES:
        for seed in SEEDS:
            counts = compute_counts_for_config(samples, seed, config, dataset, id_to_indices)
            if counts:
                result = {
                    'samples_per_class': samples,
                    'seed': seed,
                    **counts
                }
                results.append(result)

                if seed == 0:  # Print first seed for each sample size
                    print(f"\nsamples={samples}, seed={seed}:")
                    print(f"  Query:   HQ={counts['query']['HQ']:3d}, LQ={counts['query']['LQ']:3d}, total={counts['query']['total']}")
                    print(f"  Gallery: HQ={counts['gallery']['HQ']:3d}, LQ={counts['gallery']['LQ']:3d}, total={counts['gallery']['total']}")

    # Save results
    output_path = 'reid_triplet/results/quality_counts_analysis.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY BY SAMPLE SIZE (Query counts same across seeds)")
    print("=" * 70)
    print(f"{'Samples':<10} {'Query HQ':<12} {'Query LQ':<12} {'Gallery HQ':<15} {'Gallery LQ':<15}")
    print("-" * 70)

    for samples in SAMPLE_SIZES:
        sample_results = [r for r in results if r['samples_per_class'] == samples]
        if sample_results:
            # Query counts are the same across seeds
            q = sample_results[0]['query']
            # Gallery counts vary by seed
            g_hq = [r['gallery']['HQ'] for r in sample_results]
            g_lq = [r['gallery']['LQ'] for r in sample_results]

            g_hq_range = f"{min(g_hq)}-{max(g_hq)}" if min(g_hq) != max(g_hq) else str(min(g_hq))
            g_lq_range = f"{min(g_lq)}-{max(g_lq)}" if min(g_lq) != max(g_lq) else str(min(g_lq))

            print(f"{samples:<10} {q['HQ']:<12} {q['LQ']:<12} {g_hq_range:<15} {g_lq_range:<15}")

    # Per-individual breakdown for samples=64, seed=0
    print("\n" + "=" * 70)
    print("PER-INDIVIDUAL BREAKDOWN (samples=64, seed=0)")
    print("=" * 70)

    sample64_seed0 = [r for r in results if r['samples_per_class'] == 64 and r['seed'] == 0]
    if sample64_seed0:
        r = sample64_seed0[0]
        print(f"{'Individual':<12} {'Q_HQ':<8} {'Q_LQ':<8} {'G_HQ':<8} {'G_LQ':<8}")
        print("-" * 50)
        for ind_id in r['individuals']:
            pi = r['per_individual'][ind_id]
            print(f"{ind_id:<12} {pi['query']['HQ']:<8} {pi['query']['LQ']:<8} "
                  f"{pi['gallery']['HQ']:<8} {pi['gallery']['LQ']:<8}")

    print("\n" + "=" * 70)
    print("Analysis complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Select feasible individuals for reid experiments using HuggingFace train/test splits.

Individuals qualify for closed-set experiments if they have:
  - >= min_gallery_samples train images with pelage_score > gallery_quality_threshold
  - >= min_query_samples test images with pelage_score > query_quality_threshold

Usage:
    python preprocessing/create_validation_splits.py
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset


def json_serialize_helper(obj):
    """Convert numpy types to JSON serializable Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def main():
    parser = argparse.ArgumentParser(description='Select feasible individuals for reid experiments')
    parser.add_argument('--output_dir', type=str, default='preprocessing/results',
                       help='Output directory for results')
    parser.add_argument('--min_gallery_samples', type=int, default=64,
                       help='Minimum train images above quality threshold (default: 64)')
    parser.add_argument('--gallery_quality_threshold', type=float, default=0.5,
                       help='Quality threshold for gallery selection (default: 0.5)')
    parser.add_argument('--min_query_samples', type=int, default=1,
                       help='Minimum test images above quality threshold (default: 1)')
    parser.add_argument('--query_quality_threshold', type=float, default=0.5,
                       help='Quality threshold for query selection (default: 0.5)')
    args = parser.parse_args()

    # Manual exclusions
    EXCLUDED_ENTIRELY = ['PA23-M1', 'HLC21-H1']
    PROMOTED_TO_RARE = []

    # Load both splits
    print("Loading wolverines dataset (reidentification config)...")
    train_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    test_dataset = load_dataset("kdoherty/wolverines", "reidentification", split="test")
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    # Build per-individual stats from train split
    print("\nAnalyzing train split...")
    train_stats = {}
    for sample in train_dataset:
        ind_id = sample['id']
        if ind_id not in train_stats:
            train_stats[ind_id] = {'total': 0, 'above_threshold': 0, 'pelage_scores': []}
        train_stats[ind_id]['total'] += 1
        train_stats[ind_id]['pelage_scores'].append(sample['pelage_score'])
        if sample['pelage_score'] > args.gallery_quality_threshold:
            train_stats[ind_id]['above_threshold'] += 1

    # Build per-individual stats from test split
    print("Analyzing test split...")
    test_stats = {}
    for sample in test_dataset:
        ind_id = sample['id']
        if ind_id not in test_stats:
            test_stats[ind_id] = {'total': 0, 'above_threshold': 0}
        test_stats[ind_id]['total'] += 1
        if sample['pelage_score'] > args.query_quality_threshold:
            test_stats[ind_id]['above_threshold'] += 1

    # Determine valid individuals
    all_individuals = set(train_stats.keys()) | set(test_stats.keys())
    excluded_set = set(EXCLUDED_ENTIRELY)

    valid_individuals = []
    excluded_individuals = []

    for ind_id in sorted(all_individuals):
        if ind_id in excluded_set:
            excluded_individuals.append((ind_id, "Manually excluded"))
            continue

        train_above = train_stats.get(ind_id, {}).get('above_threshold', 0)
        test_above = test_stats.get(ind_id, {}).get('above_threshold', 0)

        if train_above >= args.min_gallery_samples and test_above >= args.min_query_samples:
            valid_individuals.append(ind_id)
        else:
            reasons = []
            if train_above < args.min_gallery_samples:
                reasons.append(f"train above {args.gallery_quality_threshold}: {train_above} < {args.min_gallery_samples}")
            if test_above < args.min_query_samples:
                reasons.append(f"test above {args.query_quality_threshold}: {test_above} < {args.min_query_samples}")
            excluded_individuals.append((ind_id, "; ".join(reasons)))

    # Sort by mean pelage score (descending)
    valid_individuals.sort(
        key=lambda x: np.mean(train_stats[x]['pelage_scores']),
        reverse=True
    )

    # Build individual_pelage_scores and individual_stats
    individual_pelage_scores = {}
    individual_stats = {}

    for ind_id in valid_individuals:
        individual_pelage_scores[ind_id] = float(np.mean(train_stats[ind_id]['pelage_scores']))

    for ind_id in sorted(all_individuals - excluded_set):
        t = train_stats.get(ind_id, {})
        s = test_stats.get(ind_id, {})
        individual_stats[ind_id] = {
            'train_total': t.get('total', 0),
            f'train_above_{args.gallery_quality_threshold}': t.get('above_threshold', 0),
            'test_total': s.get('total', 0),
            f'test_above_{args.query_quality_threshold}': s.get('above_threshold', 0),
        }

    # Report
    print(f"\nFound {len(valid_individuals)} valid individuals "
          f"(train >= {args.min_gallery_samples} at > {args.gallery_quality_threshold}, "
          f"test >= {args.min_query_samples} at > {args.query_quality_threshold}):")
    for ind_id in valid_individuals:
        t = individual_stats[ind_id]
        print(f"  {ind_id}: train={t[f'train_above_{args.gallery_quality_threshold}']}/{t['train_total']}, "
              f"test={t[f'test_above_{args.query_quality_threshold}']}/{t['test_total']}, "
              f"mean_pelage={individual_pelage_scores[ind_id]:.3f}")

    if excluded_individuals:
        print(f"\nExcluded {len(excluded_individuals)} individuals:")
        for ind_id, reason in excluded_individuals:
            print(f"  {ind_id}: {reason}")

    # Save config
    config = {
        'valid_individuals': valid_individuals,
        'promoted_to_rare': PROMOTED_TO_RARE,
        'excluded_entirely': EXCLUDED_ENTIRELY,
        'selection_criteria': {
            'min_gallery_samples': args.min_gallery_samples,
            'gallery_quality_threshold': args.gallery_quality_threshold,
            'min_query_samples': args.min_query_samples,
            'query_quality_threshold': args.query_quality_threshold,
        },
        'individual_pelage_scores': individual_pelage_scores,
        'individual_stats': individual_stats,
    }

    output_path = os.path.join(args.output_dir, 'feasible_individuals.json')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=json_serialize_helper)

    print(f"\nConfiguration saved to: {output_path}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Reassign all unknown/rare individual images to the test pool for BA evaluation.

The upstream HuggingFace split has unknown individuals in both train and test.
This script identifies their train-split indices so that downstream code can:
  1. Exclude them from train-time rare sourcing (no val BA needed)
  2. Combine train + test unknown images for a single robust test-time BA estimate

Reads: preprocessing/results/feasible_individuals.json
Writes: preprocessing/results/unknown_assignment.json

Usage:
    python preprocessing/assign_reid_unknowns_all_test.py
"""

import os
import json
import numpy as np
from datasets import load_dataset


def main():
    # Load feasibility config
    config_path = 'preprocessing/results/feasible_individuals.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    qualified = set(config['qualified_individuals'])
    # Only exclude individuals too sparse for even unknown evaluation
    # PA23-M1 was excluded from closed-set but is valid as an unknown
    EXCLUDE_FROM_UNKNOWN = ['HLC21-H1']
    excluded = set(EXCLUDE_FROM_UNKNOWN)

    # Load both splits
    print("Loading wolverines dataset (reidentification config)...")
    train_dataset = load_dataset("mpg-ranch/wolverines", "reidentification", split="train")
    test_dataset = load_dataset("mpg-ranch/wolverines", "reidentification", split="test")
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    train_ids = np.array(train_dataset['id'], dtype=object)
    train_scores = np.array(train_dataset['pelage_score'], dtype=np.float32)
    test_ids = np.array(test_dataset['id'], dtype=object)
    test_scores = np.array(test_dataset['pelage_score'], dtype=np.float32)

    # Identify unknown individuals (not qualified, not excluded)
    all_individuals = set(np.unique(train_ids)) | set(np.unique(test_ids))
    unknown_individuals = sorted(
        ind for ind in all_individuals
        if ind not in qualified and ind not in excluded
    )

    print(f"\nQualified (closed-set): {sorted(qualified)}")
    print(f"Excluded entirely: {sorted(excluded)}")
    print(f"Unknown individuals: {unknown_individuals}")

    # Collect indices per unknown individual in each split
    per_individual = {}
    total_train_unknown = 0
    total_test_unknown = 0

    for ind_id in unknown_individuals:
        train_mask = train_ids == ind_id
        test_mask = test_ids == ind_id
        train_indices = np.where(train_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        train_scores_ind = train_scores[train_mask]
        test_scores_ind = test_scores[test_mask]

        per_individual[ind_id] = {
            'train_indices': train_indices,
            'train_count': len(train_indices),
            'train_mean_pelage': float(np.mean(train_scores_ind)) if len(train_scores_ind) > 0 else 0.0,
            'test_indices': test_indices,
            'test_count': len(test_indices),
            'test_mean_pelage': float(np.mean(test_scores_ind)) if len(test_scores_ind) > 0 else 0.0,
            'total': len(train_indices) + len(test_indices),
        }

        total_train_unknown += len(train_indices)
        total_test_unknown += len(test_indices)

        print(f"  {ind_id}: train={len(train_indices)}, test={len(test_indices)}, "
              f"total={len(train_indices) + len(test_indices)}")

    # Flat list of all train indices to reassign
    all_train_unknown_indices = []
    for ind_id in unknown_individuals:
        all_train_unknown_indices.extend(per_individual[ind_id]['train_indices'])
    all_train_unknown_indices.sort()

    print(f"\nSummary:")
    print(f"  Unknown individuals: {len(unknown_individuals)}")
    print(f"  Train unknown images (to reassign): {total_train_unknown}")
    print(f"  Test unknown images (existing):     {total_test_unknown}")
    print(f"  Combined test unknown pool:          {total_train_unknown + total_test_unknown}")

    # Save config
    result = {
        'unknown_individuals': unknown_individuals,
        'train_unknown_indices': all_train_unknown_indices,
        'per_individual': per_individual,
        'summary': {
            'n_unknown_individuals': len(unknown_individuals),
            'train_unknown_images': total_train_unknown,
            'test_unknown_images': total_test_unknown,
            'combined_unknown_images': total_train_unknown + total_test_unknown,
        },
        'source_config': config_path,
    }

    output_dir = 'preprocessing/results'
    output_path = os.path.join(output_dir, 'unknown_assignment.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

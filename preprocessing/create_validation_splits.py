#!/usr/bin/env python3
"""
Create validation splits and training feasibility config for reid experiments.

This script analyzes the wolverines dataset to:
1. Create temporal validation splits for each individual (using most recent dates)
2. Assess training feasibility at different quality thresholds
3. Output feasible_individuals.json used by reid_hygiene_filter and reid_megadescriptor

Usage:
    python preprocessing/create_validation_splits.py
    python preprocessing/create_validation_splits.py --max_examples_per_class 128
"""

import os
import json
import argparse
import pandas as pd
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


def create_greedy_temporal_validation_split(train_df, individuals):
    """Create temporal validation sets using greedy selection to cover all quality bins"""
    print("Creating greedy temporal validation splits...")

    quality_bins = [
        (0.75, 1.0, '[0.75,1.0]'),
        (0.50, 0.75, '[0.5,0.75)'),
        (0.25, 0.50, '[0.25,0.5)'),
        (0.00, 0.25, '[0,0.25)')
    ]

    validation_indices = {}

    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id].copy()
        ind_data = ind_data.sort_values('ymdh', ascending=False)
        unique_ymdh = ind_data['ymdh'].unique()

        selected_indices = []
        used_ymdh_values = []
        bins_covered = set()

        for ymdh_val in unique_ymdh:
            ymdh_samples = ind_data[ind_data['ymdh'] == ymdh_val]
            selected_indices.extend(ymdh_samples.index.tolist())
            used_ymdh_values.append(int(ymdh_val))

            for min_score, max_score, bin_name in quality_bins:
                if bin_name == '[0.75,1.0]':
                    bin_samples = ymdh_samples[(ymdh_samples['pelage_score'] >= min_score) &
                                             (ymdh_samples['pelage_score'] <= max_score)]
                else:
                    bin_samples = ymdh_samples[(ymdh_samples['pelage_score'] >= min_score) &
                                             (ymdh_samples['pelage_score'] < max_score)]

                if len(bin_samples) > 0:
                    bins_covered.add(bin_name)

            if len(bins_covered) == len(quality_bins):
                break

        missing_bins = set(bin[2] for bin in quality_bins) - bins_covered
        if missing_bins:
            print(f"  {individual_id}: Could not find samples for bins: {missing_bins}")

        validation_indices[individual_id] = {
            'indices': selected_indices,
            'ymdh_values': used_ymdh_values,
            'count': len(selected_indices),
            'bins_covered': list(bins_covered),
            'missing_bins': list(missing_bins)
        }

        print(f"  {individual_id}: {len(selected_indices)} validation samples from {len(used_ymdh_values)} ymdh values")

    return validation_indices


def assess_training_feasibility(train_df, validation_indices, individuals, training_sizes):
    """Assess which individuals can support different training sizes at different thresholds"""
    print("Assessing training feasibility...")

    training_compatibility = {}

    for individual_id in individuals:
        print(f"\nAssessing {individual_id}...")
        ind_data = train_df[train_df['id'] == individual_id]
        val_indices = validation_indices[individual_id]['indices']
        training_data = ind_data[~ind_data.index.isin(val_indices)]

        threshold_compatibility = {}

        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            threshold_key = f'threshold_{threshold:.2f}'
            eligible_samples = training_data[training_data['pelage_score'] >= threshold]
            compatible_sizes = [size for size in training_sizes if len(eligible_samples) >= size]
            max_compatible = max(compatible_sizes) if compatible_sizes else 0

            print(f"  {threshold_key}: {len(eligible_samples)} eligible training -> max training size: {max_compatible}")

            threshold_compatibility[threshold_key] = {
                'eligible_samples': len(eligible_samples),
                'compatible_training_sizes': compatible_sizes,
                'max_training_size': max_compatible
            }

        overall_compatible = []
        for size in training_sizes:
            if any(size in threshold_compatibility[tk]['compatible_training_sizes'] for tk in threshold_compatibility):
                overall_compatible.append(size)

        training_compatibility[individual_id] = {
            'total_samples': len(ind_data),
            'validation_samples': len(val_indices),
            'training_samples_available': len(training_data),
            'threshold_compatibility': threshold_compatibility,
            'overall_compatible_training_sizes': overall_compatible
        }

    return training_compatibility, training_sizes


def compute_individual_stats(train_df, individuals):
    """Compute statistics for each individual"""
    print("Computing individual statistics...")

    individual_stats = {}

    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id]

        stats = {
            'total_samples': len(ind_data),
            'pelage_score_mean': ind_data['pelage_score'].mean(),
            'pelage_score_std': ind_data['pelage_score'].std(),
            'pelage_score_min': ind_data['pelage_score'].min(),
            'pelage_score_max': ind_data['pelage_score'].max(),
            'unique_dates': ind_data['ymdh'].nunique(),
            'date_range': {
                'earliest': ind_data['ymdh'].min(),
                'latest': ind_data['ymdh'].max()
            }
        }
        individual_stats[individual_id] = stats

        print(f"  {individual_id}: {stats['total_samples']} samples, "
              f"pelage_score={stats['pelage_score_mean']:.3f} +/- {stats['pelage_score_std']:.3f}")

    return individual_stats


def save_config(individuals, individual_stats, validation_indices, training_compatibility, training_sizes, output_dir, train_df):
    """Save the configuration needed by reid experiments."""
    print("Saving configuration...")

    # Manual exclusions - these individuals are excluded from all consideration
    EXCLUDED_ENTIRELY = ['PA23-M1', 'HLC21-H1']

    # Manual promotions - these individuals are promoted to novel set (excluded from closed-set)
    PROMOTED_TO_RARE = ['Tex']

    max_size = max(training_sizes)
    valid_individuals = []
    excluded_individuals = []

    for ind_id in individuals:
        # Skip manually excluded individuals
        if ind_id in EXCLUDED_ENTIRELY:
            excluded_individuals.append((ind_id, "Manually excluded from all consideration"))
            continue

        # Skip manually promoted individuals (they go to rare/novel set)
        if ind_id in PROMOTED_TO_RARE:
            continue

        if validation_indices[ind_id]['count'] == 0:
            excluded_individuals.append((ind_id, "No validation samples"))
            continue

        threshold_compatibility = training_compatibility[ind_id]['threshold_compatibility']
        all_thresholds_compatible = True
        failing_thresholds = []

        for threshold in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
            threshold_key = f'threshold_{threshold:.2f}'
            if threshold_key in threshold_compatibility:
                compat_data = threshold_compatibility[threshold_key]
                if max_size not in compat_data['compatible_training_sizes']:
                    all_thresholds_compatible = False
                    failing_thresholds.append((threshold, compat_data['eligible_samples']))

        if all_thresholds_compatible:
            valid_individuals.append(ind_id)
        else:
            reason = f"Insufficient samples at thresholds: {', '.join([f'{t:.2f} ({n} available)' for t, n in failing_thresholds])}"
            excluded_individuals.append((ind_id, reason))

    valid_individuals.sort(key=lambda x: individual_stats[x]['pelage_score_mean'], reverse=True)

    print(f"\nFound {len(valid_individuals)} valid individuals who can support {max_size} training examples at ALL thresholds:")
    for ind_id in valid_individuals:
        score = individual_stats[ind_id]['pelage_score_mean']
        val_count = validation_indices[ind_id]['count']
        print(f"  {ind_id}: pelage_score={score:.3f}, {val_count} validation samples")

    if PROMOTED_TO_RARE:
        print(f"\nPromoted to rare/novel set ({len(PROMOTED_TO_RARE)} individuals):")
        for ind_id in PROMOTED_TO_RARE:
            print(f"  {ind_id}: Manually promoted to novel set for open-set evaluation")

    if excluded_individuals:
        print(f"\nExcluded {len(excluded_individuals)} individuals:")
        for ind_id, reason in excluded_individuals:
            print(f"  {ind_id}: {reason}")

    config = {
        'valid_individuals': valid_individuals,
        'promoted_to_rare': PROMOTED_TO_RARE,
        'excluded_entirely': EXCLUDED_ENTIRELY,
        'max_examples_per_class': max_size,
        'training_sizes': training_sizes,
        'thresholds': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'individual_pelage_scores': {
            ind_id: individual_stats[ind_id]['pelage_score_mean']
            for ind_id in valid_individuals
        },
        'validation_indices': {
            ind_id: validation_indices[ind_id]
            for ind_id in valid_individuals
        },
        'training_compatibility': {
            ind_id: training_compatibility[ind_id]
            for ind_id in valid_individuals
        },
        'validation_strategy': {
            'type': 'temporal_greedy_bins'
        }
    }

    output_path = os.path.join(output_dir, 'feasible_individuals.json')
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=json_serialize_helper)

    print(f"\nConfiguration saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create validation splits for reid experiments')
    parser.add_argument('--output_dir', type=str, default='preprocessing/results',
                       help='Output directory for results')
    parser.add_argument('--max_examples_per_class', type=int, default=64,
                       help='Maximum training examples per class to consider (default: 64)')
    args = parser.parse_args()

    print("Loading wolverines dataset (reidentification config)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")

    print("Converting to DataFrame...")
    df_data = []
    for i, sample in enumerate(dataset):
        df_data.append({
            'index': i,
            'id': sample['id'],
            'ymdh': sample['ymdh'],
            'pelage_score': sample['pelage_score']
        })

    train_df = pd.DataFrame(df_data)

    print("Analyzing individuals...")
    individuals = train_df['id'].unique()
    print(f"Found {len(individuals)} unique individuals in dataset")

    min_samples = 2
    individuals_with_enough_samples = []

    for ind_id in individuals:
        ind_count = len(train_df[train_df['id'] == ind_id])
        if ind_count >= min_samples:
            individuals_with_enough_samples.append(ind_id)
        else:
            print(f"  Excluding {ind_id}: only {ind_count} samples")

    print(f"After filtering: {len(individuals_with_enough_samples)} individuals with >={min_samples} samples")

    training_sizes = []
    size = 1
    while size <= args.max_examples_per_class:
        training_sizes.append(size)
        size *= 2

    print(f"Training sizes to evaluate: {training_sizes}")

    individual_stats = compute_individual_stats(train_df, individuals_with_enough_samples)
    validation_indices = create_greedy_temporal_validation_split(train_df, individuals_with_enough_samples)
    training_compatibility, _ = assess_training_feasibility(
        train_df, validation_indices, individuals_with_enough_samples, training_sizes
    )

    save_config(
        individuals_with_enough_samples, individual_stats, validation_indices,
        training_compatibility, training_sizes, args.output_dir, train_df
    )

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

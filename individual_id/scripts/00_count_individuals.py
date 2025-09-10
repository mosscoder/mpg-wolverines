#!/usr/bin/env python3
"""
Script to assess training feasibility with temporal validation splits for individual ID experiments.
Creates the feasible_individuals.json file needed by script 01.
Uses last unique ymdh for validation and quality thresholds [0, 0.25, 0.5, 0.75].
"""

import os
import json
import argparse
import math
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
    
    # Quality bins to ensure coverage
    quality_bins = [
        (0.75, 1.0, '[0.75,1.0]'),
        (0.50, 0.75, '[0.5,0.75)'),
        (0.25, 0.50, '[0.25,0.5)'),
        (0.00, 0.25, '[0,0.25)')
    ]
    
    validation_indices = {}
    
    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id].copy()
        
        # Sort by ymdh descending (newest first)
        ind_data = ind_data.sort_values('ymdh', ascending=False)
        
        # Get unique ymdh values (newest first)
        unique_ymdh = ind_data['ymdh'].unique()
        
        selected_indices = []
        used_ymdh_values = []
        bins_covered = set()
        
        # Greedily add ymdh values until all quality bins are covered
        for ymdh_val in unique_ymdh:
            ymdh_samples = ind_data[ind_data['ymdh'] == ymdh_val]
            selected_indices.extend(ymdh_samples.index.tolist())
            used_ymdh_values.append(int(ymdh_val))  # Convert to native Python int
            
            # Check which bins are now covered
            for min_score, max_score, bin_name in quality_bins:
                if bin_name == '[0.75,1.0]':
                    bin_samples = ymdh_samples[(ymdh_samples['pelage_score'] >= min_score) & 
                                             (ymdh_samples['pelage_score'] <= max_score)]
                else:
                    bin_samples = ymdh_samples[(ymdh_samples['pelage_score'] >= min_score) & 
                                             (ymdh_samples['pelage_score'] < max_score)]
                
                if len(bin_samples) > 0:
                    bins_covered.add(bin_name)
            
            # Stop if all bins are covered
            if len(bins_covered) == len(quality_bins):
                break
        
        # If we still don't have all bins covered, report what's missing
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
        print(f"    Covered bins: {', '.join(bins_covered)}")
        if missing_bins:
            print(f"    Missing bins: {', '.join(missing_bins)}")
    
    return validation_indices


def assess_training_feasibility(train_df, validation_indices, individuals, training_sizes):
    """Assess which individuals can support different training sizes at different thresholds"""
    print("Assessing training feasibility...")
    
    training_compatibility = {}
    
    for individual_id in individuals:
        print(f"\nAssessing {individual_id}...")
        ind_data = train_df[train_df['id'] == individual_id]
        
        # Get validation indices (same for all thresholds now)
        val_indices = validation_indices[individual_id]['indices']
        
        # Remaining training data (exclude validation)
        training_data = ind_data[~ind_data.index.isin(val_indices)]
        
        # For each threshold, see what training sizes are feasible
        threshold_compatibility = {}
        
        for threshold in [0, 0.25, 0.5, 0.75]:
            threshold_key = f'threshold_{threshold:.2f}'
            
            # Filter by quality threshold
            eligible_samples = training_data[training_data['pelage_score'] >= threshold]
            
            # Check which training sizes are compatible
            compatible_sizes = [size for size in training_sizes if len(eligible_samples) >= size]
            max_compatible = max(compatible_sizes) if compatible_sizes else 0
            
            print(f"  {threshold_key}: {len(eligible_samples)} eligible training -> max training size: {max_compatible}")
            
            threshold_compatibility[threshold_key] = {
                'eligible_samples': len(eligible_samples),
                'compatible_training_sizes': compatible_sizes,
                'max_training_size': max_compatible
            }
        
        # Overall compatibility (any threshold can support these sizes)
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
              f"pelage_score={stats['pelage_score_mean']:.3f} ± {stats['pelage_score_std']:.3f}, "
              f"{stats['unique_dates']} unique dates")
    
    return individual_stats


def report_validation_bin_counts(train_df, validation_indices, valid_individuals):
    """Report validation sample counts by quality bin for each individual"""
    print("\n" + "="*80)
    print("VALIDATION SET QUALITY DISTRIBUTION")
    print("="*80)
    print(f"{'Individual':<15} {'Total':<8} {'[0.75,1.0]':<10} {'[0.5,0.75)':<10} {'[0.25,0.5)':<10} {'[0,0.25)':<10}")
    print("-"*80)
    
    # Quality bins
    quality_bins = [
        (0.75, 1.0, '[0.75,1.0]'),
        (0.50, 0.75, '[0.5,0.75)'),
        (0.25, 0.50, '[0.25,0.5)'),
        (0.00, 0.25, '[0,0.25)')
    ]
    
    total_counts = {bin_name: 0 for _, _, bin_name in quality_bins}
    total_validation_samples = 0
    
    for individual_id in valid_individuals:
        val_indices = validation_indices[individual_id]['indices']
        val_data = train_df.loc[val_indices]
        
        bin_counts = {}
        for min_score, max_score, bin_name in quality_bins:
            if bin_name == '[0.75,1.0]':
                bin_samples = val_data[(val_data['pelage_score'] >= min_score) & (val_data['pelage_score'] <= max_score)]
            else:
                bin_samples = val_data[(val_data['pelage_score'] >= min_score) & (val_data['pelage_score'] < max_score)]
            
            count = len(bin_samples)
            bin_counts[bin_name] = count
            total_counts[bin_name] += count
        
        total_val = len(val_indices)
        total_validation_samples += total_val
        
        print(f"{individual_id:<15} {total_val:<8} {bin_counts['[0.75,1.0]']:<10} "
              f"{bin_counts['[0.5,0.75)']:<10} {bin_counts['[0.25,0.5)']:<10} {bin_counts['[0,0.25)']:<10}")
    
    print("-"*80)
    print(f"{'TOTAL':<15} {total_validation_samples:<8} {total_counts['[0.75,1.0]']:<10} "
          f"{total_counts['[0.5,0.75)']:<10} {total_counts['[0.25,0.5)']:<10} {total_counts['[0,0.25)']:<10}")
    print("="*80)


def save_config(individuals, individual_stats, validation_indices, training_compatibility, training_sizes, output_dir, train_df):
    """Save the configuration needed by script 01"""
    print("Saving configuration...")
    
    # Identify valid individuals who can support max_examples_per_class at ALL thresholds
    max_size = max(training_sizes)
    valid_individuals = []
    excluded_individuals = []
    
    for ind_id in individuals:
        # Must have validation samples
        if validation_indices[ind_id]['count'] == 0:
            excluded_individuals.append((ind_id, "No validation samples"))
            continue
        
        # Must support max training size at ALL thresholds
        threshold_compatibility = training_compatibility[ind_id]['threshold_compatibility']
        all_thresholds_compatible = True
        failing_thresholds = []
        
        for threshold in [0.00, 0.25, 0.50, 0.75]:
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
    
    # Sort valid individuals by pelage score
    valid_individuals.sort(key=lambda x: individual_stats[x]['pelage_score_mean'], reverse=True)
    
    print(f"\nFound {len(valid_individuals)} valid individuals who can support {max_size} training examples at ALL thresholds:")
    for ind_id in valid_individuals:
        score = individual_stats[ind_id]['pelage_score_mean']
        val_data = validation_indices[ind_id]
        val_count = val_data['count']
        num_ymdh = len(val_data.get('ymdh_values', [val_data.get('ymdh', 1)]))
        bins_covered = len(val_data.get('bins_covered', []))
        print(f"  {ind_id}: pelage_score={score:.3f}, {val_count} validation samples from {num_ymdh} ymdh values, {bins_covered}/4 bins covered")
    
    if excluded_individuals:
        print(f"\nExcluded {len(excluded_individuals)} individuals:")
        for ind_id, reason in excluded_individuals:
            print(f"  {ind_id}: {reason}")
    
    # Report validation bin distribution
    report_validation_bin_counts(train_df, validation_indices, valid_individuals)
    
    config = {
        'valid_individuals': valid_individuals,
        'max_examples_per_class': max_size,
        'training_sizes': training_sizes,
        'thresholds': [0, 0.25, 0.5, 0.75],
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
    
    print(f"✓ Configuration saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create feasibility analysis for individual ID experiments')
    parser.add_argument('--output_dir', type=str, default='individual_id/results',
                       help='Output directory for results')
    parser.add_argument('--max_examples_per_class', type=int, default=64,
                       help='Maximum training examples per class to consider (default: 64)')
    args = parser.parse_args()
    
    # Load dataset
    print("Loading wolverines dataset (reidentification config)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    
    # Convert to DataFrame
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
    
    # Find individuals with sufficient samples
    print("Analyzing individuals...")
    individuals = train_df['id'].unique()
    print(f"Found {len(individuals)} unique individuals in dataset")
    
    # Filter individuals with minimum samples (at least 2: 1 for validation + 1 for training)
    min_samples = 2
    individuals_with_enough_samples = []
    
    for ind_id in individuals:
        ind_count = len(train_df[train_df['id'] == ind_id])
        if ind_count >= min_samples:
            individuals_with_enough_samples.append(ind_id)
        else:
            print(f"  Excluding {ind_id}: only {ind_count} samples (need at least {min_samples})")
    
    print(f"After filtering: {len(individuals_with_enough_samples)} individuals with ≥{min_samples} samples")
    
    # Generate training sizes (powers of 2)
    training_sizes = []
    size = 1
    while size <= args.max_examples_per_class:
        training_sizes.append(size)
        size *= 2
    
    print(f"Training sizes to evaluate: {training_sizes}")
    
    # Compute individual stats
    individual_stats = compute_individual_stats(train_df, individuals_with_enough_samples)
    
    # Create greedy temporal validation splits
    validation_indices = create_greedy_temporal_validation_split(train_df, individuals_with_enough_samples)
    
    # Assess training feasibility
    training_compatibility, _ = assess_training_feasibility(
        train_df, validation_indices, individuals_with_enough_samples, training_sizes
    )
    
    # Save configuration
    output_path = save_config(
        individuals_with_enough_samples, individual_stats, validation_indices, 
        training_compatibility, training_sizes, args.output_dir, train_df
    )
    
    print(f"\n✓ Feasibility analysis complete!")
    print(f"Configuration saved to: {output_path}")
    print("Ready to run script 01_individual_id_sweep.py")


if __name__ == "__main__":
    main()
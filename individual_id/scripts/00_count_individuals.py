#!/usr/bin/env python3
"""
Minimal script to assess training feasibility and create validation sets for individual ID experiments.
Creates the feasible_individuals.json file needed by script 01.
"""

import os
import json
import argparse
import math
import pandas as pd
from datasets import load_dataset


def create_validation_indices(train_df, individuals):
    """Create cumulative validation sets for each individual by quality threshold"""
    print("Creating validation indices...")
    
    # Quality bins for validation
    thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]
    quality_ranges = [
        (0.8, 1.0, '[0.8,1.0]'),
        (0.6, 0.8, '[0.6,0.8)'), 
        (0.4, 0.6, '[0.4,0.6)'),
        (0.2, 0.4, '[0.2,0.4)'),
        (0.0, 0.2, '[0.0,0.2)')
    ]
    max_val_per_bin = 10
    
    validation_indices = {}
    
    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id].copy()
        ind_data = ind_data.sort_values('ymdh', ascending=False)  # Newest first
        
        # Create cumulative validation sets
        threshold_validation = {}
        cumulative_indices = []
        cumulative_bins = []
        
        for threshold, (min_score, max_score, bin_name) in zip(thresholds, quality_ranges):
            # Select samples from this quality range
            if bin_name == '[0.8,1.0]':
                range_samples = ind_data[(ind_data['pelage_score'] >= min_score) & (ind_data['pelage_score'] <= max_score)]
            else:
                range_samples = ind_data[(ind_data['pelage_score'] >= min_score) & (ind_data['pelage_score'] < max_score)]
            
            # Add up to max_val_per_bin newest samples
            selected_samples = range_samples.head(max_val_per_bin)
            new_indices = selected_samples.index.tolist()
            new_bins = [bin_name] * len(new_indices)
            
            cumulative_indices.extend(new_indices)
            cumulative_bins.extend(new_bins)
            
            # Store cumulative validation set for this threshold
            threshold_validation[f'threshold_{threshold:.1f}'] = {
                'indices': cumulative_indices.copy(),
                'bin_assignments': cumulative_bins.copy(),
                'count': len(cumulative_indices)
            }
        
        validation_indices[individual_id] = threshold_validation
    
    return validation_indices


def compute_feasibility(train_df, individuals, validation_indices, max_examples_per_class=64):
    """Compute which training sizes are feasible for each individual at each threshold"""
    print(f"Computing feasibility matrix (up to {max_examples_per_class} examples per class)...")
    
    thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]
    # Generate powers of 2 up to max_examples_per_class
    max_power = int(math.log2(max_examples_per_class))
    training_sizes = [2**i for i in range(max_power + 1)]
    validation_compatibility = {}
    
    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id]
        threshold_compatibility = {}
        
        for threshold in thresholds:
            threshold_key = f'threshold_{threshold:.1f}'
            
            # Count eligible samples for training
            eligible_samples = ind_data[ind_data['pelage_score'] >= threshold]
            
            # Get validation samples used
            val_samples_used = len(validation_indices[individual_id][threshold_key]['indices'])
            
            # Check which training sizes are feasible
            compatible_sizes = []
            for size in training_sizes:
                total_needed = val_samples_used + size
                if len(eligible_samples) >= total_needed:
                    compatible_sizes.append(size)
            
            max_compatible = max(compatible_sizes) if compatible_sizes else 0
            
            threshold_compatibility[threshold_key] = {
                'eligible_samples': len(eligible_samples),
                'validation_samples': val_samples_used,
                'compatible_training_sizes': compatible_sizes,
                'max_training_size': max_compatible
            }
        
        # Overall compatibility
        overall_compatible = []
        for size in training_sizes:
            if any(size in threshold_compatibility[tk]['compatible_training_sizes'] for tk in threshold_compatibility):
                overall_compatible.append(size)
        
        validation_compatibility[individual_id] = {
            'total_samples': len(ind_data),
            'max_validation_samples': len(validation_indices[individual_id]['threshold_0.0']['indices']),
            'threshold_compatibility': threshold_compatibility,
            'overall_compatible_training_sizes': overall_compatible
        }
    
    return validation_compatibility, training_sizes


def save_config(individuals, individual_stats, validation_indices, validation_compatibility, training_sizes, output_dir):
    """Save the configuration needed by script 01"""
    print("Saving configuration...")
    
    # Sort individuals by pelage score
    individuals_with_scores = [
        (ind_id, individual_stats[ind_id]['pelage_score_mean']) 
        for ind_id in individuals
    ]
    individuals_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    config = {
        'individuals_sorted_by_pelage': [ind[0] for ind in individuals_with_scores],
        'individual_pelage_scores': {
            ind_id: individual_stats[ind_id]['pelage_score_mean'] 
            for ind_id in individuals
        },
        'individual_total_counts': {
            ind_id: individual_stats[ind_id]['total_samples'] 
            for ind_id in individuals
        },
        'validation_indices': validation_indices,
        'validation_compatibility': validation_compatibility,
        'training_sizes': training_sizes,
        'validation_strategy': {
            'type': 'cumulative_by_threshold',
            'thresholds': [0.8, 0.6, 0.4, 0.2, 0.0],
            'training_sizes': training_sizes,
            'max_per_range': 10
        }
    }
    
    output_path = os.path.join(output_dir, 'feasible_individuals.json')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
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
    print("Loading wolverines dataset (train split only)...")
    dataset = load_dataset('kdoherty/wolverines', 'reidentification')
    train_dataset = dataset['train']
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Convert to DataFrame
    data = []
    for item in train_dataset:
        data.append({
            'id': item['id'],
            'ymdh': item['ymdh'],
            'pelage_score': item['pelage_score']
        })
    train_df = pd.DataFrame(data)
    
    # Calculate individual statistics
    print("Analyzing individuals...")
    individual_stats = {}
    individuals = train_df['id'].unique()
    
    for individual_id in individuals:
        ind_data = train_df[train_df['id'] == individual_id]
        individual_stats[individual_id] = {
            'total_samples': len(ind_data),
            'pelage_score_mean': ind_data['pelage_score'].mean(),
            'pelage_score_std': ind_data['pelage_score'].std(),
            'pelage_score_min': ind_data['pelage_score'].min(),
            'pelage_score_max': ind_data['pelage_score'].max()
        }
    
    print(f"Found {len(individuals)} individuals")
    
    # Create validation indices
    validation_indices = create_validation_indices(train_df, individuals)
    
    # Compute feasibility
    validation_compatibility, training_sizes = compute_feasibility(train_df, individuals, validation_indices, args.max_examples_per_class)
    
    # Save configuration
    config_path = save_config(individuals, individual_stats, validation_indices, 
                             validation_compatibility, training_sizes, args.output_dir)
    
    # Feasibility Summary
    print(f"\nFeasibility Summary (up to {args.max_examples_per_class} examples per class):")
    print("=" * 60)
    
    # Generate powers of 2 for display, starting from 4 to avoid too much detail
    max_power = int(math.log2(args.max_examples_per_class))
    key_training_sizes = [2**i for i in range(2, max_power + 1)]  # Start from 2^2=4
    thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]
    
    for training_size in key_training_sizes:
        print(f"\nTraining size: {training_size} samples")
        
        for threshold in thresholds:
            threshold_key = f'threshold_{threshold:.1f}'
            
            # Find individuals feasible for this training size at this threshold
            feasible_individuals = []
            for ind_id in individuals:
                if training_size in validation_compatibility[ind_id]['threshold_compatibility'][threshold_key]['compatible_training_sizes']:
                    feasible_individuals.append(ind_id)
            
            # Sort by pelage score for consistent ordering
            feasible_individuals.sort(key=lambda x: individual_stats[x]['pelage_score_mean'], reverse=True)
            
            if feasible_individuals:
                print(f"  Threshold ≥{threshold:.1f}: {len(feasible_individuals)} individuals feasible [{', '.join(feasible_individuals)}]")
            else:
                print(f"  Threshold ≥{threshold:.1f}: 0 individuals feasible")
    
    print(f"\n✓ Analysis complete. Configuration saved for script 01.")


if __name__ == "__main__":
    main()
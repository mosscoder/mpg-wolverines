#!/usr/bin/env python3
"""
Script 00: Count Individuals
Count individuals in training set (id col) by unique dates (ymdh) and label.
Analyze data availability for individual ID experiments.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from collections import defaultdict, Counter
from datetime import datetime
import argparse

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.dataset import load_wolverines_dataset

def parse_ymdh(ymdh_int):
    """Parse YMDH integer format to datetime components"""
    ymdh_str = str(ymdh_int)
    if len(ymdh_str) == 12:  # Format: YYYYMMDDHHMM
        year = int(ymdh_str[:4])
        month = int(ymdh_str[4:6])
        day = int(ymdh_str[6:8])
        hour = int(ymdh_str[8:10])
        minute = int(ymdh_str[10:12])
        return year, month, day, hour, minute
    else:
        return None, None, None, None, None

def create_compatibility_matrix(individual_stats):
    """Create compatibility matrix for pelage-only classification"""
    
    sample_sizes = [2, 4, 8, 16, 32, 64]
    individuals = sorted(individual_stats.keys())
    
    # Create matrix
    matrix = []
    for individual_id in individuals:
        stats = individual_stats[individual_id]
        pelage_samples = stats['label_1_samples']
        row = [1 if pelage_samples >= size else 0 for size in sample_sizes]
        matrix.append(row)
    
    # Convert to DataFrame for better display
    matrix_df = pd.DataFrame(matrix, 
                           index=individuals, 
                           columns=[f'{size}_samples' for size in sample_sizes])
    
    return matrix_df, sample_sizes

def create_compatibility_figure(matrix_df, sample_sizes, individual_stats, test_individual_stats, output_dir):
    """Create and save pelage sample count bar graph with train and test data"""
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Get all unique individuals from both datasets
    all_individuals = set(individual_stats.keys()) | set(test_individual_stats.keys())
    
    # Get pelage counts and sort by total count (train + test, descending)
    individual_data = []
    for ind in all_individuals:
        train_count = individual_stats.get(ind, {}).get('label_1_samples', 0)
        test_count = test_individual_stats.get(ind, {}).get('label_1_samples', 0)
        total_count = train_count + test_count
        individual_data.append((ind, train_count, test_count, total_count))
    
    individual_data.sort(key=lambda x: x[3], reverse=True)  # Sort by total count, highest first
    
    individuals = [x[0] for x in individual_data]
    train_counts = [x[1] for x in individual_data]
    test_counts = [x[2] for x in individual_data]
    
    # Create positions for dodged bars
    y_pos = np.arange(len(individuals))
    bar_height = 0.35
    
    # Create horizontal bar chart with dodged bars
    train_bars = plt.barh(y_pos - bar_height/2, train_counts, bar_height, 
                         color='blue', alpha=0.8, label='Train')
    test_bars = plt.barh(y_pos + bar_height/2, test_counts, bar_height, 
                        color='red', alpha=0.8, label='Test')
    
    # Set non-linear x-axis with specific tick marks
    x_ticks = [2, 4, 8, 16, 32, 64, 128]
    plt.xticks(x_ticks)
    
    # Set x-axis limits to show all ticks properly
    max_count = max(max(train_counts + test_counts, default=0), 128)
    plt.xlim(0, max_count * 1.1)
    
    # Add dashed light gray horizontal grid lines
    plt.grid(axis='x', linestyle='--', color='lightgray', alpha=0.7)
    
    # Add value annotations at the right end of each bar
    for i, (individual, train_count, test_count) in enumerate(zip(individuals, train_counts, test_counts)):
        if train_count > 0:
            plt.text(train_count + 1, i - bar_height/2, str(train_count), 
                    va='center', ha='left', fontsize=8, color='blue')
        if test_count > 0:
            plt.text(test_count + 1, i + bar_height/2, str(test_count), 
                    va='center', ha='left', fontsize=8, color='red')
    
    plt.xlabel('Pelage Sample Count')
    plt.ylabel('Individual')
    plt.yticks(y_pos, individuals)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure in individual_id/figures/
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get individual_id directory
    figure_path = os.path.join(script_dir, 'figures', 'pelage_compatibility_matrix.png')
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Pelage sample count figure saved to: {figure_path}")
    return figure_path

def save_feasible_individuals(individual_stats, output_dir):
    """Save list of individuals with sufficient pelage samples for experiments"""
    
    # Find individuals with at least 64 pelage samples (32 train + 32 val)
    feasible_pelage_64 = []
    for ind_id, stats in individual_stats.items():
        if stats['label_1_samples'] >= 64:
            feasible_pelage_64.append((ind_id, stats['label_1_samples']))
    
    # Sort by pelage count (descending) and take top 3
    feasible_pelage_64.sort(key=lambda x: x[1], reverse=True)
    top_3_individuals = [x[0] for x in feasible_pelage_64[:3]]
    
    # Create configuration for script 01
    config = {
        'feasible_individuals_pelage_64': top_3_individuals,
        'individual_pelage_counts': {
            ind: individual_stats[ind]['label_1_samples'] 
            for ind in top_3_individuals
        },
        'individual_total_counts': {
            ind: individual_stats[ind]['total_samples'] 
            for ind in top_3_individuals
        },
        'selection_criteria': {
            'min_pelage_samples': 64,
            'reason': '32 samples for training + 32 for validation',
            'approach': 'top_3_by_pelage_count'
        }
    }
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'feasible_individuals.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Feasible individuals saved to: {output_path}")
    print(f"Selected individuals: {', '.join(top_3_individuals)}")
    
    return output_path

def analyze_individuals(args):
    """Analyze individual wolverine data by date and label"""
    
    print("=" * 80)
    print("INDIVIDUAL WOLVERINE ANALYSIS - TRAIN AND TEST SETS")
    print("=" * 80)
    
    # Load dataset - BOTH TRAINING AND TEST SETS for comprehensive analysis
    print("Loading wolverines dataset...")
    train_dataset, test_dataset = load_wolverines_dataset()
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Convert to DataFrame for analysis
    def dataset_to_df(dataset):
        data = []
        for item in dataset:
            row = {
                'id': item['id'],
                'ymdh': item['ymdh'],
                'label': item['label'],
                'year': item['year'],
                'station': item['station'],
                'marks': item['marks']
            }
            data.append(row)
        return pd.DataFrame(data)
    
    train_df = dataset_to_df(train_dataset)
    test_df = dataset_to_df(test_dataset)
    
    print(f"\\nTraining DataFrame: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test DataFrame: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Analysis 1: Individual counts by label
    print("\\n" + "=" * 80)
    print("INDIVIDUAL COUNTS BY LABEL (TRAINING SET)")
    print("=" * 80)
    
    individual_stats = {}
    
    for individual_id in sorted(train_df['id'].unique()):
        ind_data = train_df[train_df['id'] == individual_id]
        
        total_samples = len(ind_data)
        label_0_count = len(ind_data[ind_data['label'] == 0])
        label_1_count = len(ind_data[ind_data['label'] == 1])
        
        # Count unique dates
        unique_dates = ind_data['ymdh'].nunique()
        
        # Year range
        years = sorted(ind_data['year'].unique())
        year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        
        individual_stats[individual_id] = {
            'total_samples': total_samples,
            'label_0_samples': label_0_count,
            'label_1_samples': label_1_count,
            'unique_dates': unique_dates,
            'year_range': year_range,
            'stations': list(ind_data['station'].unique())
        }
        
        print(f"\\n{individual_id}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Label 0 (no pelage): {label_0_count}")
        print(f"  Label 1 (pelage): {label_1_count}")
        print(f"  Unique dates: {unique_dates}")
        print(f"  Years: {year_range}")
        print(f"  Stations: {', '.join(individual_stats[individual_id]['stations'])}")
    
    # Analysis 1b: Individual counts by label - TEST SET
    print("\\n" + "=" * 80)
    print("INDIVIDUAL COUNTS BY LABEL (TEST SET)")
    print("=" * 80)
    
    test_individual_stats = {}
    
    for individual_id in sorted(test_df['id'].unique()):
        ind_data = test_df[test_df['id'] == individual_id]
        
        total_samples = len(ind_data)
        label_0_count = len(ind_data[ind_data['label'] == 0])
        label_1_count = len(ind_data[ind_data['label'] == 1])
        
        # Count unique dates
        unique_dates = ind_data['ymdh'].nunique()
        
        # Year range
        years = sorted(ind_data['year'].unique())
        year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        
        test_individual_stats[individual_id] = {
            'total_samples': total_samples,
            'label_0_samples': label_0_count,
            'label_1_samples': label_1_count,
            'unique_dates': unique_dates,
            'year_range': year_range,
            'stations': list(ind_data['station'].unique())
        }
        
        print(f"\\n{individual_id}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Label 0 (no pelage): {label_0_count}")
        print(f"  Label 1 (pelage): {label_1_count}")
        print(f"  Unique dates: {unique_dates}")
        print(f"  Years: {year_range}")
        print(f"  Stations: {', '.join(test_individual_stats[individual_id]['stations'])}")
    
    # Analysis 2: Date distribution
    print("\\n" + "=" * 80)
    print("DATE DISTRIBUTION ANALYSIS (TRAINING SET)")
    print("=" * 80)
    
    # Parse YMDH for temporal analysis
    train_df['parsed_year'] = train_df['ymdh'].apply(lambda x: parse_ymdh(x)[0])
    train_df['parsed_month'] = train_df['ymdh'].apply(lambda x: parse_ymdh(x)[1])
    train_df['parsed_day'] = train_df['ymdh'].apply(lambda x: parse_ymdh(x)[2])
    train_df['parsed_hour'] = train_df['ymdh'].apply(lambda x: parse_ymdh(x)[3])
    
    print(f"Total unique YMDH timestamps: {train_df['ymdh'].nunique()}")
    print(f"Year range: {train_df['parsed_year'].min()}-{train_df['parsed_year'].max()}")
    
    # Samples per month
    print("\\nSamples by month:")
    month_dist = train_df['parsed_month'].value_counts().sort_index()
    for month, count in month_dist.items():
        if month is not None:
            print(f"  Month {month:2d}: {count:4d} samples")
    
    # Samples per hour
    print("\\nSamples by hour:")
    hour_dist = train_df['parsed_hour'].value_counts().sort_index()
    for hour, count in hour_dist.items():
        if hour is not None:
            print(f"  Hour {hour:2d}: {count:4d} samples")
    
    # Analysis 2b: Date distribution - TEST SET
    print("\\n" + "=" * 80)
    print("DATE DISTRIBUTION ANALYSIS (TEST SET)")
    print("=" * 80)
    
    # Parse YMDH for temporal analysis
    test_df['parsed_year'] = test_df['ymdh'].apply(lambda x: parse_ymdh(x)[0])
    test_df['parsed_month'] = test_df['ymdh'].apply(lambda x: parse_ymdh(x)[1])
    test_df['parsed_day'] = test_df['ymdh'].apply(lambda x: parse_ymdh(x)[2])
    test_df['parsed_hour'] = test_df['ymdh'].apply(lambda x: parse_ymdh(x)[3])
    
    print(f"Total unique YMDH timestamps: {test_df['ymdh'].nunique()}")
    print(f"Year range: {test_df['parsed_year'].min()}-{test_df['parsed_year'].max()}")
    
    # Samples per month
    print("\\nSamples by month:")
    test_month_dist = test_df['parsed_month'].value_counts().sort_index()
    for month, count in test_month_dist.items():
        if month is not None:
            print(f"  Month {month:2d}: {count:4d} samples")
    
    # Samples per hour
    print("\\nSamples by hour:")
    test_hour_dist = test_df['parsed_hour'].value_counts().sort_index()
    for hour, count in test_hour_dist.items():
        if hour is not None:
            print(f"  Hour {hour:2d}: {count:4d} samples")
    
    # Analysis 3: Pelage-Only Compatibility Matrix
    print("\\n" + "=" * 80)
    print("PELAGE-ONLY COMPATIBILITY MATRIX")
    print("=" * 80)
    
    matrix_df, sample_sizes = create_compatibility_matrix(individual_stats)
    
    # Create and save compatibility matrix figure
    figure_path = create_compatibility_figure(matrix_df, sample_sizes, individual_stats, test_individual_stats, args.output_dir)
    
    # Save feasible individuals for script 01
    feasible_individuals_path = save_feasible_individuals(individual_stats, args.output_dir)
    
    print("\\nCompatibility matrix for pelage-only classification:")
    print("(1 = sufficient pelage samples, 0 = insufficient)")
    print()
    
    # Print header
    header = "Individual".ljust(12) + " | " + " | ".join(f"{size:2d}".center(3) for size in sample_sizes)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for individual_id in matrix_df.index:
        row_values = matrix_df.loc[individual_id].values
        pelage_count = individual_stats[individual_id]['label_1_samples']
        row_str = f"{individual_id:<12} | " + " | ".join(f" {val} ".center(3) for val in row_values)
        row_str += f"  ({pelage_count} pelage samples)"
        print(row_str)
    
    # Summary by sample size
    print("\\n" + "=" * 80)
    print("PELAGE-ONLY FEASIBILITY SUMMARY")
    print("=" * 80)
    
    print("\\nNumber of individuals with sufficient pelage samples:")
    for size in sample_sizes:
        col_name = f'{size}_samples'
        feasible_count = matrix_df[col_name].sum()
        total_individuals = len(matrix_df)
        feasible_individuals = matrix_df[matrix_df[col_name] == 1].index.tolist()
        
        print(f"\\n{size:2d} samples: {feasible_count}/{total_individuals} individuals feasible")
        if feasible_individuals:
            print(f"    Individuals: {', '.join(feasible_individuals)}")
    
    # Analysis 4: Random Sample Feasibility (for comparison)
    print("\\n" + "=" * 80)
    print("RANDOM SAMPLE APPROACH FEASIBILITY")
    print("=" * 80)
    
    print("\\nRandom sample approach (using any samples regardless of label):")
    for size in sample_sizes:
        feasible_individuals = []
        for individual_id, stats in individual_stats.items():
            if stats['total_samples'] >= size:
                feasible_individuals.append(individual_id)
        
        print(f"\\n{size:2d} samples: {len(feasible_individuals)}/{len(individual_stats)} individuals feasible")
        if feasible_individuals:
            print(f"    Individuals: {', '.join(feasible_individuals)}")
    
    # Prepare results for saving
    results = {
        'dataset_summary': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(test_df),
            'train_individuals': len(individual_stats),
            'test_individuals': len(test_individual_stats),
            'total_unique_individuals': len(set(individual_stats.keys()) | set(test_individual_stats.keys())),
            'train_unique_dates': train_df['ymdh'].nunique(),
            'test_unique_dates': test_df['ymdh'].nunique()
        },
        'train_data': {
            'individual_statistics': individual_stats,
            'temporal_distribution': {
                'year_range': f"{train_df['parsed_year'].min()}-{train_df['parsed_year'].max()}",
                'samples_by_month': month_dist.to_dict(),
                'samples_by_hour': hour_dist.to_dict()
            },
            'label_distribution': {
                'label_0': int(train_df[train_df['label'] == 0].shape[0]),
                'label_1': int(train_df[train_df['label'] == 1].shape[0])
            }
        },
        'test_data': {
            'individual_statistics': test_individual_stats,
            'temporal_distribution': {
                'year_range': f"{test_df['parsed_year'].min()}-{test_df['parsed_year'].max()}",
                'samples_by_month': test_month_dist.to_dict(),
                'samples_by_hour': test_hour_dist.to_dict()
            },
            'label_distribution': {
                'label_0': int(test_df[test_df['label'] == 0].shape[0]),
                'label_1': int(test_df[test_df['label'] == 1].shape[0])
            }
        },
        'pelage_only_compatibility_matrix': {
            'sample_sizes': sample_sizes,
            'matrix': matrix_df.to_dict(),
            'feasible_counts': {size: int(matrix_df[f'{size}_samples'].sum()) for size in sample_sizes}
        }
    }
    
    # Save results
    output_path = os.path.join(args.output_dir, 'individual_analysis.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✓ Analysis results saved to: {output_path}")
    
    # Recommendations
    print("\\n" + "=" * 80)
    print("RECOMMENDATIONS FOR INDIVIDUAL ID EXPERIMENTS")
    print("=" * 80)
    
    # Find optimal sample sizes for experiments
    min_individuals_needed = 5  # Minimum for meaningful multi-class classification
    
    print(f"\\nFor meaningful multi-class classification, recommend ≥{min_individuals_needed} individuals:")
    
    print("\\nPELAGE-ONLY APPROACH:")
    for size in sample_sizes:
        col_name = f'{size}_samples'
        feasible_count = matrix_df[col_name].sum()
        if feasible_count >= min_individuals_needed:
            print(f"  ✓ {size:2d} samples: {feasible_count} individuals feasible - RECOMMENDED")
        else:
            print(f"  ✗ {size:2d} samples: {feasible_count} individuals feasible - insufficient")
    
    print("\\nRANDOM SAMPLE APPROACH:")
    for size in sample_sizes:
        feasible_count = sum(1 for stats in individual_stats.values() if stats['total_samples'] >= size)
        if feasible_count >= min_individuals_needed:
            print(f"  ✓ {size:2d} samples: {feasible_count} individuals feasible - RECOMMENDED")
        else:
            print(f"  ✗ {size:2d} samples: {feasible_count} individuals feasible - insufficient")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze individual wolverine data')
    parser.add_argument('--output_dir', type=str,
                       default='results',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    try:
        results = analyze_individuals(args)
        print("\\n🎯 Individual analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
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

# Assume script is run from wolverines root directory
sys.path.append('.')

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

def create_yearly_compatibility_figure(train_df, test_df, individual_stats, test_individual_stats, output_dir):
    """Create and save single-panel stacked pelage sample count bar graph with yearly breakdown"""
    
    # Set up single-panel figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Get all unique individuals from both datasets and sort by total pelage count
    all_individuals = set(individual_stats.keys()) | set(test_individual_stats.keys())
    individual_data = []
    for ind in all_individuals:
        train_count = individual_stats.get(ind, {}).get('label_1_samples', 0)
        test_count = test_individual_stats.get(ind, {}).get('label_1_samples', 0)
        total_count = train_count + test_count
        individual_data.append((ind, train_count, test_count, total_count))
    
    individual_data.sort(key=lambda x: x[3], reverse=True)  # Sort by total count, highest first
    individuals = [x[0] for x in individual_data]
    
    # Get all years present in the data
    all_years = sorted(list(set(train_df['parsed_year'].dropna().astype(int)) | 
                           set(test_df['parsed_year'].dropna().astype(int))))
    n_years = len(all_years)
    
    # Create color palette for years
    colors = plt.cm.Set3(np.linspace(0, 1, n_years))
    year_colors = {year: colors[i] for i, year in enumerate(all_years)}
    
    # Function to get yearly counts for a dataset
    def get_yearly_counts_by_individual(df, individual_list):
        yearly_data = defaultdict(lambda: defaultdict(int))
        for ind in individual_list:
            ind_data = df[(df['id'] == ind) & (df['label'] == 1)]  # Only pelage samples
            if not ind_data.empty:
                year_counts = ind_data['parsed_year'].value_counts()
                for year, count in year_counts.items():
                    if year is not None:
                        yearly_data[ind][int(year)] = count
        return yearly_data
    
    # Get yearly breakdowns
    train_yearly = get_yearly_counts_by_individual(train_df, individuals)
    test_yearly = get_yearly_counts_by_individual(test_df, individuals)
    
    # Calculate positions for dodged bars
    y_pos = np.arange(len(individuals))
    bar_height = 0.8 / n_years if n_years > 0 else 0.8
    
    # Single panel with stacked train+test bars
    max_total_count = 0
    
    # Create legend entries for train/test distinction
    legend_train = None
    legend_test = None
    
    for year_idx, year in enumerate(all_years):
        train_counts = [train_yearly[ind].get(year, 0) for ind in individuals]
        test_counts = [test_yearly[ind].get(year, 0) for ind in individuals]
        
        # Calculate max for this year
        year_totals = [train_counts[i] + test_counts[i] for i in range(len(individuals))]
        max_total_count = max(max_total_count, max(year_totals, default=0))
        
        y_positions = y_pos + (year_idx - n_years/2 + 0.5) * bar_height
        
        # Train bars (base, fully opaque)
        train_bars = ax.barh(y_positions, train_counts, bar_height,
                            color=year_colors[year], alpha=1.0, 
                            label=f'{year}' if year_idx == 0 else "")
        
        # Test bars (stacked on train, semi-transparent)
        test_bars = ax.barh(y_positions, test_counts, bar_height,
                           left=train_counts, color=year_colors[year], alpha=0.5)
        
        # Store legend examples (only need one set)
        if legend_train is None:
            legend_train = train_bars
            legend_test = test_bars
        
        # Add annotations for totals
        for i, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
            total_count = train_count + test_count
            if total_count > 0:
                ax.text(total_count + 0.5, y_positions[i], 
                       f'{train_count}+{test_count}' if test_count > 0 else str(train_count),
                       va='center', ha='left', fontsize=7)
    
    # Formatting
    ax.set_title('Pelage Sample Counts by Year (Train + Test Stacked)', fontsize=14, pad=20)
    ax.set_xlabel('Pelage Sample Count')
    ax.set_ylabel('Individual')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(individuals)
    ax.grid(axis='x', linestyle='--', color='lightgray', alpha=0.7)
    
    # Create custom legend showing year colors and train/test opacity
    import matplotlib.patches as mpatches
    legend_elements = []
    for year in all_years:
        legend_elements.append(mpatches.Rectangle((0,0),1,1, facecolor=year_colors[year], alpha=1.0, label=f'{year}'))
    
    # Add train/test distinction
    legend_elements.extend([
        mpatches.Rectangle((0,0),1,1, facecolor='gray', alpha=1.0, label='Train (Solid)'),
        mpatches.Rectangle((0,0),1,1, facecolor='gray', alpha=0.5, label='Test (Transparent)')
    ])
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set consistent x-axis
    x_ticks = [2, 4, 8, 16, 32, 64, 128]
    max_count = max(max_total_count, 128)
    x_limit = max_count * 1.1
    
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, x_limit)
    
    plt.tight_layout()
    
    # Save figure
    figure_path = 'individual_id/figures/pelage_yearly_counts.png'
    os.makedirs('individual_id/figures', exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Single-panel stacked yearly pelage count figure saved to: {figure_path}")
    return figure_path


def create_compatibility_figure(matrix_df, sample_sizes, individual_stats, test_individual_stats, output_dir):
    """Create and save 2-panel pelage sample count bar graph with yearly breakdown"""
    
    # Set up 2-panel figure
    fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
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
    
    # Get yearly breakdown - need to access the DataFrames from analyze_individuals
    # We'll need to pass the DataFrames to this function
    def get_yearly_counts(dataset_df, individual_list):
        """Get pelage counts by individual and year"""
        yearly_data = defaultdict(lambda: defaultdict(int))
        
        for ind in individual_list:
            ind_data = dataset_df[dataset_df['id'] == ind]
            pelage_data = ind_data[ind_data['label'] == 1]  # Only pelage samples
            
            if 'parsed_year' in pelage_data.columns:
                year_counts = pelage_data['parsed_year'].value_counts()
                for year, count in year_counts.items():
                    if year is not None:
                        yearly_data[ind][year] = count
        
        return yearly_data
    
    # For now, we'll extract years from the existing stats and create a simplified view
    # This is a placeholder - in the actual implementation, we'd pass the DataFrames
    
    # Get all years present in the data
    all_years = set()
    for stats in individual_stats.values():
        year_range = stats['year_range']
        if '-' in year_range:
            start_year, end_year = map(int, year_range.split('-'))
            all_years.update(range(start_year, end_year + 1))
        else:
            all_years.add(int(year_range))
    
    for stats in test_individual_stats.values():
        year_range = stats['year_range']
        if '-' in year_range:
            start_year, end_year = map(int, year_range.split('-'))
            all_years.update(range(start_year, end_year + 1))
        else:
            all_years.add(int(year_range))
    
    all_years = sorted(list(all_years))
    n_years = len(all_years)
    
    # Create color palette for years
    colors = plt.cm.Set3(np.linspace(0, 1, n_years))
    
    # For this version, we'll show aggregated counts and note that yearly breakdown needs DataFrame access
    y_pos = np.arange(len(individuals))
    bar_height = 0.35
    
    # Training panel (top)
    train_counts = [individual_stats.get(ind, {}).get('label_1_samples', 0) for ind in individuals]
    train_bars = ax_train.barh(y_pos, train_counts, bar_height, 
                              color='blue', alpha=0.8, label='Train (All Years)')
    
    # Add annotations
    for i, (individual, count) in enumerate(zip(individuals, train_counts)):
        if count > 0:
            ax_train.text(count + 1, i, str(count), 
                         va='center', ha='left', fontsize=8, color='blue')
    
    ax_train.set_title('Training Set Pelage Sample Counts', fontsize=14, pad=20)
    ax_train.set_ylabel('Individual')
    ax_train.set_yticks(y_pos)
    ax_train.set_yticklabels(individuals)
    ax_train.grid(axis='x', linestyle='--', color='lightgray', alpha=0.7)
    ax_train.legend()
    
    # Test panel (middle)
    test_counts = [test_individual_stats.get(ind, {}).get('label_1_samples', 0) for ind in individuals]
    test_bars = ax_test.barh(y_pos, test_counts, bar_height,
                            color='red', alpha=0.8, label='Test (All Years)')
    
    # Add annotations
    for i, (individual, count) in enumerate(zip(individuals, test_counts)):
        if count > 0:
            ax_test.text(count + 1, i, str(count), 
                        va='center', ha='left', fontsize=8, color='red')
    
    ax_test.set_title('Test Set Pelage Sample Counts', fontsize=14, pad=20)
    ax_test.set_ylabel('Individual')
    ax_test.set_yticks(y_pos)
    ax_test.set_yticklabels(individuals)
    ax_test.grid(axis='x', linestyle='--', color='lightgray', alpha=0.7)
    ax_test.legend()
    
    # Pooled panel (bottom)
    pooled_counts = [train_counts[i] + test_counts[i] for i in range(len(individuals))]
    pooled_bars = ax_pooled.barh(y_pos, pooled_counts, bar_height,
                                color='purple', alpha=0.8, label='Train + Test Pooled')
    
    # Add annotations
    for i, (individual, count) in enumerate(zip(individuals, pooled_counts)):
        if count > 0:
            ax_pooled.text(count + 1, i, str(count), 
                          va='center', ha='left', fontsize=8, color='purple')
    
    ax_pooled.set_title('Pooled (Train + Test) Pelage Sample Counts', fontsize=14, pad=20)
    ax_pooled.set_xlabel('Pelage Sample Count')
    ax_pooled.set_ylabel('Individual')
    ax_pooled.set_yticks(y_pos)
    ax_pooled.set_yticklabels(individuals)
    ax_pooled.grid(axis='x', linestyle='--', color='lightgray', alpha=0.7)
    ax_pooled.legend()
    
    # Set consistent x-axis with non-linear ticks
    x_ticks = [2, 4, 8, 16, 32, 64, 128]
    max_count = max(max(train_counts + test_counts + pooled_counts, default=0), 128)
    x_limit = max_count * 1.1
    
    ax_train.set_xticks(x_ticks)
    ax_train.set_xlim(0, x_limit)
    ax_test.set_xticks(x_ticks)
    ax_test.set_xlim(0, x_limit)
    ax_pooled.set_xticks(x_ticks)
    ax_pooled.set_xlim(0, x_limit)
    
    plt.tight_layout()
    
    # Save figure in individual_id/figures/
    figure_path = 'individual_id/figures/pelage_compatibility_matrix.png'
    os.makedirs('individual_id/figures', exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 3-panel pelage sample count figure saved to: {figure_path}")
    print(f"Note: For detailed yearly breakdown, the function needs access to the DataFrames")
    return figure_path

def save_feasible_individuals(individual_stats, test_individual_stats, train_df, test_df, output_dir):
    """Save list of individuals with sufficient pelage samples and year-by-year data"""
    
    # Get ALL individuals sorted by pelage count (descending)
    all_individuals_sorted = []
    for ind_id, stats in individual_stats.items():
        all_individuals_sorted.append({
            'id': ind_id,
            'pelage_count': stats['label_1_samples'],
            'total_count': stats['total_samples']
        })
    all_individuals_sorted.sort(key=lambda x: x['pelage_count'], reverse=True)
    
    # Top individuals for experiments (script 01 will index into this list)
    top_individuals = [ind['id'] for ind in all_individuals_sorted]
    
    # Create year-by-year data for temporal splits (for top individuals)
    individual_year_data = {}
    for ind_id in top_individuals[:5]:  # Store data for top 5 for flexibility
        # Get train year data
        train_ind_data = train_df[train_df['id'] == ind_id]
        train_years = sorted([int(y) for y in train_ind_data['year'].unique()])
        
        # Get test year data  
        test_ind_data = test_df[test_df['id'] == ind_id]
        test_years = sorted([int(y) for y in test_ind_data['year'].unique()])
        
        # All years (train + test)
        all_years = sorted(set(train_years) | set(test_years))
        val_year = int(max(all_years))  # Use final year as validation
        
        # Year-by-year counts
        year_counts = {}
        pooled_data = pd.concat([train_ind_data, test_ind_data])
        for year in all_years:
            year_data = pooled_data[pooled_data['year'] == year]
            year_counts[str(year)] = {
                'label_0': int(len(year_data[year_data['label'] == 0])),
                'label_1': int(len(year_data[year_data['label'] == 1]))
            }
        
        individual_year_data[ind_id] = {
            'train_years': train_years,
            'test_years': test_years,
            'all_years': all_years,
            'val_year': val_year,
            'year_counts': year_counts
        }
    
    # Create configuration for script 01
    config = {
        'individuals_sorted_by_pelage': top_individuals,
        'individual_pelage_counts': {
            ind['id']: ind['pelage_count'] 
            for ind in all_individuals_sorted
        },
        'individual_total_counts': {
            ind['id']: ind['total_count'] 
            for ind in all_individuals_sorted
        },
        'individual_year_data': individual_year_data,
        'selection_criteria': {
            'sorting': 'pelage_count_descending',
            'year_data_computed_for': f'top_{len(individual_year_data)}_individuals'
        }
    }
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'feasible_individuals.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Individual data saved to: {output_path}")
    print(f"Top 3 individuals by pelage count: {', '.join(top_individuals[:3])}")
    
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
    
    # Create and save compatibility matrix figure with yearly breakdown
    figure_path = create_yearly_compatibility_figure(train_df, test_df, individual_stats, test_individual_stats, args.output_dir)
    
    # Save feasible individuals for script 01
    feasible_individuals_path = save_feasible_individuals(individual_stats, test_individual_stats, train_df, test_df, args.output_dir)
    
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
                       default='individual_id/results',
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
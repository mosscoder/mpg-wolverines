#!/usr/bin/env python3
"""
Script 02: Plot Individual ID Results
Creates line charts showing individual ID performance across quality thresholds.
Analyzes cross-validated best epoch performance for each configuration.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import argparse

def load_results(results_dir):
    """Load all result CSV files from results directory"""
    
    pattern = os.path.join(results_dir, "samples=*_threshold=*_seed=*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No result files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} result files")
    
    results = []
    failed_files = []
    
    for csv_file in csv_files:
        try:
            # Parse filename to extract parameters
            basename = os.path.basename(csv_file)
            # Format: samples=X_threshold=Y.ZZ_seed=N.csv
            parts = basename.replace('.csv', '').split('_')
            
            sample_size = None
            threshold = None
            seed = None
            
            for part in parts:
                if part.startswith('samples='):
                    sample_size = int(part.split('=')[1])
                elif part.startswith('threshold='):
                    threshold = float(part.split('=')[1])  # Now a float
                elif part.startswith('seed='):
                    seed = int(part.split('=')[1])
            
            if sample_size is None or threshold is None or seed is None:
                failed_files.append((csv_file, "Could not parse filename"))
                continue
                
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            result = {
                'sample_size': sample_size,
                'threshold': threshold,
                'seed': seed,
                'epochs_data': df.to_dict('records'),
                'filename': basename
            }
            results.append(result)
            
        except Exception as e:
            failed_files.append((csv_file, str(e)))
            continue
    
    if failed_files:
        print(f"Failed to load {len(failed_files)} files:")
        for file, error in failed_files:
            print(f"  {os.path.basename(file)}: {error}")
    
    print(f"Successfully loaded {len(results)} results")
    return results

def extract_threshold_metrics(results):
    """Extract cross-validated best epoch metrics from CSV results"""
    
    # Group results by (sample_size, threshold)
    grouped = defaultdict(list)
    for result in results:
        key = (result['sample_size'], result['threshold'])
        grouped[key].append(result)
    
    metrics = {}
    performance_data = []
    
    for (sample_size, threshold), group_results in grouped.items():
        if len(group_results) < 8:
            print(f"Warning: Only {len(group_results)} seeds for samples={sample_size}, threshold={threshold}")
        
        # Track metrics for overall F1 and each bin separately
        best_epoch_f1s = []  # Overall F1 best epochs
        best_epochs = []     # Overall F1 best epoch numbers
        bin_metrics_per_seed = defaultdict(lambda: defaultdict(list))  # bin_name -> seed -> [f1_scores, best_epochs]
        
        # Define validation bins
        bin_definitions = [('val_f1_bin_0.75_1.0', '[0.75,1.0]'),
                          ('val_f1_bin_0.5_0.75', '[0.5,0.75)'),
                          ('val_f1_bin_0.25_0.5', '[0.25,0.5)'),
                          ('val_f1_bin_0_0.25', '[0,0.25)')]
        
        for result in group_results:
            epochs_data = result['epochs_data']
            
            if not epochs_data:
                print(f"Warning: No epoch data for {result['filename']}")
                continue
                
            # Find best epoch based on overall validation F1 score (for overall metrics)
            best_epoch_idx_overall = 0
            best_val_f1_overall = 0.0
            
            for i, epoch_data in enumerate(epochs_data):
                val_f1 = epoch_data.get('val_f1_overall', 0.0)
                if val_f1 > best_val_f1_overall:
                    best_val_f1_overall = val_f1
                    best_epoch_idx_overall = i
            
            best_epoch_data_overall = epochs_data[best_epoch_idx_overall]
            best_epoch_f1s.append(best_val_f1_overall)
            best_epochs.append(best_epoch_data_overall['epoch'])
            
            # Add to performance data with overall metrics
            performance_data.append({
                'sample_size': sample_size,
                'threshold': threshold,
                'f1_score': best_val_f1_overall,
                'best_epoch': best_epoch_data_overall['epoch'],
                'seed': result['seed']
            })
            
            # For each bin, find its own best epoch and collect metrics
            for bin_col, bin_name in bin_definitions:
                best_epoch_idx_bin = 0
                best_val_f1_bin = 0.0
                
                # Find epoch with best F1 for this specific bin
                for i, epoch_data in enumerate(epochs_data):
                    bin_f1 = epoch_data.get(bin_col, 0.0)
                    if bin_f1 > best_val_f1_bin:
                        best_val_f1_bin = bin_f1
                        best_epoch_idx_bin = i
                
                # Store the best F1 and epoch for this bin
                if best_val_f1_bin > 0.0:  # Only add if we have real data
                    bin_metrics_per_seed[bin_name]['f1_scores'].append(best_val_f1_bin)
                    bin_metrics_per_seed[bin_name]['best_epochs'].append(epochs_data[best_epoch_idx_bin]['epoch'])
        
        # Store metrics for this configuration
        if best_epoch_f1s:
            # Aggregate bin metrics across seeds (using bin-specific best epochs)
            aggregated_bin_metrics = {}
            for bin_name, bin_data in bin_metrics_per_seed.items():
                f1_scores = bin_data['f1_scores']
                best_epochs_bin = bin_data['best_epochs']
                
                if f1_scores:
                    aggregated_bin_metrics[bin_name] = {
                        'mean_f1': np.mean(f1_scores),
                        'std_f1': np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0,
                        'n_seeds': len(f1_scores),
                        'mean_best_epoch': np.mean(best_epochs_bin),
                        'f1_scores': f1_scores,  # Store individual scores for CI calculation
                        'best_epochs': best_epochs_bin
                    }
            
            metrics[(sample_size, threshold)] = {
                'f1_scores': best_epoch_f1s,
                'mean_f1': np.mean(best_epoch_f1s),
                'std_f1': np.std(best_epoch_f1s, ddof=1),
                'n_seeds': len(best_epoch_f1s),
                'mean_best_epoch': np.mean(best_epochs),
                'best_epochs': best_epochs,
                'bin_metrics': aggregated_bin_metrics
            }
            
            print(f"samples={sample_size}, threshold={threshold}: "
                  f"Overall F1={np.mean(best_epoch_f1s):.4f} (n={len(best_epoch_f1s)} seeds, "
                  f"avg best epoch={np.mean(best_epochs):.1f})")
            
            # Print bin-specific metrics summary
            for bin_name, bin_data in aggregated_bin_metrics.items():
                print(f"  Bin {bin_name}: F1={bin_data['mean_f1']:.4f} "
                      f"(avg best epoch={bin_data['mean_best_epoch']:.1f})")
    
    return metrics, pd.DataFrame(performance_data)

def plot_results(metrics, performance_df, output_path):
    """Create line chart showing performance across quality thresholds"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sample_sizes = sorted(performance_df['sample_size'].unique())
    thresholds = [0.0, 0.25, 0.5, 0.75]  # New threshold values
    
    # Colors for different quality thresholds
    colors = {
        0.0: '#e74c3c',    # Red - lowest quality
        0.25: '#f39c12',   # Orange 
        0.5: '#f1c40f',    # Yellow
        0.75: '#27ae60',   # Green - highest quality
    }
    
    # Plot lines with confidence interval ribbons
    legend_handles = []
    legend_labels = []
    all_ci_lower = []
    all_ci_upper = []
    
    for threshold in thresholds:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []
        
        for sample_size in sample_sizes:
            key = (sample_size, threshold)
            
            if key in metrics:
                data = metrics[key]
                mean_f1 = data['mean_f1']
                f1_scores = data['f1_scores']
                
                # Calculate 95% confidence interval
                if len(f1_scores) > 1:
                    sem = stats.sem(f1_scores)
                    ci_range = stats.t.ppf(0.975, len(f1_scores)-1) * sem
                else:
                    ci_range = 0
                
                x_vals.append(sample_size)
                y_vals.append(mean_f1)
                ci_lower_val = mean_f1 - ci_range
                ci_upper_val = mean_f1 + ci_range
                ci_lower.append(ci_lower_val)
                ci_upper.append(ci_upper_val)
                
                # Collect CI bounds for y-axis scaling
                all_ci_lower.append(ci_lower_val)
                all_ci_upper.append(ci_upper_val)
        
        if x_vals:  # Only plot if we have data
            # Plot main line
            line = ax.plot(x_vals, y_vals, color=colors[threshold], linewidth=2.5, 
                          marker='o', markersize=8, label=f'threshold_{threshold}')[0]
            
            # Add confidence interval ribbon
            ax.fill_between(x_vals, ci_lower, ci_upper, 
                           color=colors[threshold], alpha=0.2)
            
            legend_handles.append(line)
            if threshold == 0.0:
                legend_labels.append(f"Training threshold ≥ {threshold:.0f} (all data)")
            else:
                legend_labels.append(f"Training threshold ≥ {threshold:.2f}")
    
    # Styling
    ax.set_xlabel('Images per Individual', fontsize=14)
    ax.set_ylabel('Cross-Validated Best Epoch F1 Score', fontsize=14)
    ax.set_title('Individual ID Performance by Training Quality Threshold\n(Cross-validated best epoch F1 across seeds)', fontsize=16, pad=20)
    
    # Set x-axis based on actual sample sizes
    if sample_sizes:
        ax.set_xticks(sample_sizes)
        ax.set_xlim(min(sample_sizes) - 1, max(sample_sizes) + 1)
    
    # Add legend
    legend = ax.legend(legend_handles, legend_labels, loc='lower right')
    legend.set_title("Training data quality:", prop={'weight': 'bold'})
    
    # Set dynamic y-axis limits based on confidence interval bounds
    if all_ci_lower and all_ci_upper:
        min_acc = min(all_ci_lower)
        max_acc = max(all_ci_upper)
        y_min = max(0.0, min_acc - 0.02)  # Don't go below 0
        y_max = min(1.0, max_acc + 0.02)  # Don't go above 1
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.0)  # Fallback
    
    # Add horizontal gridlines
    ax.grid(True, alpha=0.3, axis='y', color='lightgray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

def plot_bin_results(metrics, performance_df, output_path):
    """Create single figure with 4 columns showing performance for each validation quality bin"""
    
    sample_sizes = sorted(performance_df['sample_size'].unique())
    thresholds = [0.0, 0.25, 0.5, 0.75]
    
    # Colors for different quality thresholds
    colors = {
        0.0: '#e74c3c',    # Red - lowest quality
        0.25: '#f39c12',   # Orange 
        0.5: '#f1c40f',    # Yellow
        0.75: '#27ae60',   # Green - highest quality
    }
    
    # Quality bins to plot
    bin_names = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']
    bin_display_names = {
        '[0.75,1.0]': 'High Quality\n(0.75-1.0)',
        '[0.5,0.75)': 'Medium-High Quality\n(0.5-0.75)', 
        '[0.25,0.5)': 'Medium-Low Quality\n(0.25-0.5)',
        '[0,0.25)': 'Low Quality\n(0.0-0.25)'
    }
    
    # Create figure with 4 subplots (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
    fig.suptitle('Individual ID Performance by Validation Quality Bin\n(Bin-specific best epoch F1 across seeds)', 
                 fontsize=18, y=0.95)
    
    # Collect all CI bounds for consistent y-axis scaling
    global_ci_lower = []
    global_ci_upper = []
    
    # First pass: collect all CI bounds
    for bin_idx, bin_name in enumerate(bin_names):
        for threshold in thresholds:
            for sample_size in sample_sizes:
                key = (sample_size, threshold)
                
                if key in metrics:
                    data = metrics[key]
                    bin_metrics = data.get('bin_metrics', {})
                    
                    if bin_name in bin_metrics:
                        bin_data = bin_metrics[bin_name]
                        mean_f1 = bin_data['mean_f1']
                        f1_scores = bin_data['f1_scores']
                        
                        # Calculate 95% confidence interval using individual scores
                        if len(f1_scores) > 1:
                            sem = stats.sem(f1_scores)
                            ci_range = stats.t.ppf(0.975, len(f1_scores)-1) * sem
                        else:
                            ci_range = 0
                        
                        global_ci_lower.append(mean_f1 - ci_range)
                        global_ci_upper.append(mean_f1 + ci_range)
    
    # Second pass: create plots
    for bin_idx, bin_name in enumerate(bin_names):
        ax = axes[bin_idx]
        
        legend_handles = []
        legend_labels = []
        
        for threshold in thresholds:
            x_vals = []
            y_vals = []
            ci_lower = []
            ci_upper = []
            
            for sample_size in sample_sizes:
                key = (sample_size, threshold)
                
                if key in metrics:
                    data = metrics[key]
                    bin_metrics = data.get('bin_metrics', {})
                    
                    if bin_name in bin_metrics:
                        bin_data = bin_metrics[bin_name]
                        mean_f1 = bin_data['mean_f1']
                        f1_scores = bin_data['f1_scores']
                        
                        # Calculate 95% confidence interval using individual scores
                        if len(f1_scores) > 1:
                            sem = stats.sem(f1_scores)
                            ci_range = stats.t.ppf(0.975, len(f1_scores)-1) * sem
                        else:
                            ci_range = 0
                        
                        x_vals.append(sample_size)
                        y_vals.append(mean_f1)
                        ci_lower.append(mean_f1 - ci_range)
                        ci_upper.append(mean_f1 + ci_range)
            
            if x_vals:  # Only plot if we have data
                # Plot main line
                line = ax.plot(x_vals, y_vals, color=colors[threshold], linewidth=2.5, 
                              marker='o', markersize=6, label=f'threshold_{threshold}')[0]
                
                # Add confidence interval ribbon
                ax.fill_between(x_vals, ci_lower, ci_upper, 
                               color=colors[threshold], alpha=0.2)
                
                legend_handles.append(line)
                if threshold == 0.0:
                    legend_labels.append(f"≥ {threshold:.0f} (all data)")
                else:
                    legend_labels.append(f"≥ {threshold:.2f}")
        
        # Styling for each subplot
        ax.set_xlabel('Images per Individual', fontsize=12)
        if bin_idx == 0:  # Only leftmost plot gets y-label
            ax.set_ylabel('F1 Score (Bin-Specific Best Epoch)', fontsize=12)
        ax.set_title(bin_display_names[bin_name], fontsize=14, pad=15)
        
        # Set x-axis based on actual sample sizes
        if sample_sizes:
            ax.set_xticks(sample_sizes)
            ax.set_xlim(min(sample_sizes) - 1, max(sample_sizes) + 1)
        
        # Add legend to rightmost plot only
        if bin_idx == len(bin_names) - 1 and legend_handles:
            legend = ax.legend(legend_handles, legend_labels, loc='lower right', fontsize=10)
            legend.set_title("Training threshold:", prop={'size': 10, 'weight': 'bold'})
        
        # Add horizontal gridlines
        ax.grid(True, alpha=0.3, axis='y', color='lightgray')
    
    # Set consistent y-axis limits across all subplots
    if global_ci_lower and global_ci_upper:
        min_f1 = min(global_ci_lower)
        max_f1 = max(global_ci_upper)
        y_min = max(0.0, min_f1 - 0.02)  # Don't go below 0
        y_max = min(1.0, max_f1 + 0.02)  # Don't go above 1
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    else:
        for ax in axes:
            ax.set_ylim(0.0, 1.0)  # Fallback
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bin-specific figure saved to: {output_path}")

def print_summary_table(metrics):
    """Print summary table of results"""
    
    print("\n" + "="*100)
    print("SUMMARY TABLE - Cross-Validated Best Epoch Performance")
    print("="*100)
    print(f"{'Config':<25} {'Mean F1':<10} {'Std F1':<10} {'95% CI':<15} {'N':<3} {'Avg Best Epoch':<15}")
    print("-"*100)
    
    for (sample_size, threshold), data in sorted(metrics.items()):
        config = f"{sample_size} samples ≥{threshold:.2f}"
        mean_f1 = data['mean_f1']
        std_f1 = data['std_f1']
        n_seeds = data['n_seeds']
        mean_best_epoch = data['mean_best_epoch']
        
        # Calculate 95% CI
        if n_seeds > 1:
            sem = std_f1 / np.sqrt(n_seeds)
            ci_range = stats.t.ppf(0.975, n_seeds-1) * sem
            ci_str = f"±{ci_range:.3f}"
        else:
            ci_str = "N/A"
        
        print(f"{config:<25} {mean_f1:.4f}{'':>4} {std_f1:.4f}{'':>4} "
              f"{ci_str:<15} {n_seeds:<3} {mean_best_epoch:.1f}{'':>13}")
    
    # Print bin-specific summary for some configurations
    print("\n" + "="*100)
    print("BIN-SPECIFIC PERFORMANCE (selected configurations)")
    print("="*100)
    
    # Show bin performance for a few key configurations
    key_configs = [(8, 0.0), (16, 0.0), (32, 0.0)] if metrics else []
    
    for sample_size, threshold in key_configs:
        if (sample_size, threshold) in metrics:
            data = metrics[(sample_size, threshold)]
            bin_metrics = data.get('bin_metrics', {})
            
            print(f"\n{sample_size} samples, threshold ≥{threshold:.2f}:")
            print(f"{'Bin':<15} {'Mean F1':<10} {'Std F1':<10} {'N Seeds':<8}")
            print("-"*50)
            
            for bin_name in ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']:
                if bin_name in bin_metrics:
                    bin_data = bin_metrics[bin_name]
                    print(f"{bin_name:<15} {bin_data['mean_f1']:.4f}{'':>4} "
                          f"{bin_data['std_f1']:.4f}{'':>4} {bin_data['n_seeds']:<8}")
                else:
                    print(f"{bin_name:<15} {'No data':<20}")

def main():
    parser = argparse.ArgumentParser(description='Plot Individual ID results')
    parser.add_argument('--results_dir', type=str, default='individual_id/results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Directory to save output figure')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path_overall = os.path.join(args.output_dir, 'individual_id_threshold_f1_performance.png')
    output_path_bins = os.path.join(args.output_dir, 'individual_id_threshold_f1_performance_by_bin.png')
    
    print("="*60)
    print("Individual ID Temporal Validation Analysis")
    print("="*60)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        return
    
    # Extract threshold metrics and create performance dataframe
    metrics, performance_df = extract_threshold_metrics(results)
    
    if not metrics:
        print("No valid results found")
        return
    
    # Create overall plot
    plot_results(metrics, performance_df, output_path_overall)
    
    # Create bin-specific plot (single figure with 4 columns)
    print("\nGenerating bin-specific performance plot...")
    plot_bin_results(metrics, performance_df, output_path_bins)
    
    # Print summary table
    print_summary_table(metrics)
    
    print(f"\nAnalysis complete!")
    print(f"Overall F1 figure saved to: {output_path_overall}")
    print(f"Bin-specific F1 figure saved to: {output_path_bins}")

if __name__ == "__main__":
    main()
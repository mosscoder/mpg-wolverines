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
        
        best_epoch_f1s = []
        best_epochs = []
        bin_metrics_per_seed = defaultdict(list)  # Track bin performance across seeds
        
        for result in group_results:
            epochs_data = result['epochs_data']
            
            if not epochs_data:
                print(f"Warning: No epoch data for {result['filename']}")
                continue
                
            # Find best epoch based on overall validation F1 score
            best_epoch_idx = 0
            best_val_f1 = 0.0
            
            for i, epoch_data in enumerate(epochs_data):
                val_f1 = epoch_data.get('val_f1_overall', 0.0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch_idx = i
            
            best_epoch_data = epochs_data[best_epoch_idx]
            best_epoch_f1s.append(best_val_f1)
            best_epochs.append(best_epoch_data['epoch'])
            
            # Add to performance data with bin-specific metrics
            performance_data.append({
                'sample_size': sample_size,
                'threshold': threshold,
                'f1_score': best_val_f1,
                'best_epoch': best_epoch_data['epoch'],
                'seed': result['seed']
            })
            
            # Collect bin metrics for this seed
            for bin_col, bin_name in [('val_f1_bin_0.75_1.0', '[0.75,1.0]'),
                                    ('val_f1_bin_0.5_0.75', '[0.5,0.75)'),
                                    ('val_f1_bin_0.25_0.5', '[0.25,0.5)'),
                                    ('val_f1_bin_0_0.25', '[0,0.25)')]:
                bin_f1 = best_epoch_data.get(bin_col, 0.0)
                if bin_f1 > 0.0:  # Only add if we have real data
                    bin_metrics_per_seed[bin_name].append(bin_f1)
        
        # Store metrics for this configuration
        if best_epoch_f1s:
            # Aggregate bin metrics across seeds
            aggregated_bin_metrics = {}
            for bin_name, bin_f1s in bin_metrics_per_seed.items():
                if bin_f1s:
                    aggregated_bin_metrics[bin_name] = {
                        'mean_f1': np.mean(bin_f1s),
                        'std_f1': np.std(bin_f1s, ddof=1) if len(bin_f1s) > 1 else 0.0,
                        'n_seeds': len(bin_f1s)
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
                  f"F1={np.mean(best_epoch_f1s):.4f} (n={len(best_epoch_f1s)} seeds, "
                  f"avg best epoch={np.mean(best_epochs):.1f})")
    
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
    output_path = os.path.join(args.output_dir, 'individual_id_threshold_f1_performance.png')
    
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
    
    # Create plot
    plot_results(metrics, performance_df, output_path)
    
    # Print summary table
    print_summary_table(metrics)
    
    print(f"\nAnalysis complete! Figure saved to: {output_path}")

if __name__ == "__main__":
    main()
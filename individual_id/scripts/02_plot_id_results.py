#!/usr/bin/env python3
"""
Script 02: Plot Individual ID Results
Creates bar charts with color-alpha design comparing pelage ratio training approaches.
Analyzes performance across overall, visible, and invisible validation sets.
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
    """Load all result JSON files from results directory"""
    
    pattern = os.path.join(results_dir, "samples=*_approach=*_seed=*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    results = []
    failed_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            failed_files.append((json_file, str(e)))
            continue
    
    if failed_files:
        print(f"Failed to load {len(failed_files)} files:")
        for file, error in failed_files:
            print(f"  {os.path.basename(file)}: {error}")
    
    print(f"Successfully loaded {len(results)} results")
    return results

def extract_pelage_metrics(results):
    """Extract overall, visible, and invisible pelage metrics from results"""
    
    # Group results by (sample_size, approach)
    grouped = defaultdict(list)
    for result in results:
        key = (result['sample_size'], result['approach'])
        grouped[key].append(result)
    
    metrics = {}
    performance_data = []
    
    for (sample_size, approach), group_results in grouped.items():
        if len(group_results) < 5:
            print(f"Warning: Only {len(group_results)} seeds for samples={sample_size}, approach={approach}")
        
        # Extract three types of validation metrics
        overall_accs = []
        visible_accs = []
        invisible_accs = []
        
        for result in group_results:
            # Overall accuracy from performance
            if 'performance' in result and 'final_val_accuracy' in result['performance']:
                overall_accs.append(result['performance']['final_val_accuracy'])
            
            # Pelage-specific accuracies from final metrics
            if 'performance' in result and 'final_pelage_metrics' in result['performance']:
                pelage_metrics = result['performance']['final_pelage_metrics']
                
                if 'visible' in pelage_metrics:
                    visible_accs.append(pelage_metrics['visible']['accuracy'])
                
                if 'invisible' in pelage_metrics:
                    invisible_accs.append(pelage_metrics['invisible']['accuracy'])
        
        # Store metrics for each validation type
        config_key = (sample_size, approach)
        
        if overall_accs:
            metrics[(config_key, 'overall')] = {
                'accuracies': overall_accs,
                'mean_accuracy': np.mean(overall_accs),
                'std_accuracy': np.std(overall_accs, ddof=1),
                'n_seeds': len(overall_accs)
            }
            
            # Add to performance data
            for acc in overall_accs:
                performance_data.append({
                    'sample_size': sample_size,
                    'approach': approach,
                    'val_type': 'overall',
                    'accuracy': acc
                })
        
        if visible_accs:
            metrics[(config_key, 'visible')] = {
                'accuracies': visible_accs,
                'mean_accuracy': np.mean(visible_accs),
                'std_accuracy': np.std(visible_accs, ddof=1),
                'n_seeds': len(visible_accs)
            }
            
            # Add to performance data
            for acc in visible_accs:
                performance_data.append({
                    'sample_size': sample_size,
                    'approach': approach,
                    'val_type': 'visible',
                    'accuracy': acc
                })
        
        if invisible_accs:
            metrics[(config_key, 'invisible')] = {
                'accuracies': invisible_accs,
                'mean_accuracy': np.mean(invisible_accs),
                'std_accuracy': np.std(invisible_accs, ddof=1),
                'n_seeds': len(invisible_accs)
            }
            
            # Add to performance data
            for acc in invisible_accs:
                performance_data.append({
                    'sample_size': sample_size,
                    'approach': approach,
                    'val_type': 'invisible',
                    'accuracy': acc
                })
        
        print(f"samples={sample_size}, approach={approach}: "
              f"Accuracy={np.mean(overall_accs):.4f} (n={len(overall_accs)} seeds)")
    
    return metrics, pd.DataFrame(performance_data)

def plot_results(metrics, performance_df, output_path):
    """Create bar chart with color-alpha design"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sample_sizes = sorted(performance_df['sample_size'].unique())
    approaches = ['pelage', 'random']
    
    # Colors for training approaches
    colors = {
        'pelage': '#27ae60',     # Green - pelage only training
        'random': '#3498db'      # Blue - random sampling training
    }
    
    # Bar width and positioning
    n_groups = len(sample_sizes)
    n_approaches = len(approaches) 
    
    bar_width = 0.35
    group_spacing = 0.8
    
    x_positions = np.arange(n_groups) * group_spacing
    
    # Plot bars
    legend_handles = []
    legend_labels = []
    
    for i, sample_size in enumerate(sample_sizes):
        base_x = x_positions[i]
        
        for j, approach in enumerate(approaches):
            # Use overall accuracy since validation matches training condition
            key = ((sample_size, approach), 'overall')
            
            if key in metrics:
                data = metrics[key]
                mean_acc = data['mean_accuracy']
                accuracies = data['accuracies']
                
                # Calculate 95% confidence interval
                if len(accuracies) > 1:
                    sem = stats.sem(accuracies)
                    ci_range = stats.t.ppf(0.975, len(accuracies)-1) * sem
                else:
                    ci_range = 0
                
                # Plot bar
                x_pos = base_x + (j - 0.5) * bar_width
                bar = ax.bar(x_pos, mean_acc, bar_width, 
                           color=colors[approach],
                           edgecolor='black', linewidth=0.5, alpha=0.8)
                
                # Add error bar
                ax.errorbar(x_pos, mean_acc, yerr=ci_range, fmt='none', 
                          color='black', capsize=3, linewidth=1)
                
                # Add to legend (only once per approach)
                if i == 0:  # Only add legend items from first sample size
                    label = f"{approach.title()} Training"
                    legend_handles.append(bar[0])
                    legend_labels.append(label)
    
    # Styling
    ax.set_xlabel('Training Samples per Individual', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title('Individual ID Performance: Pelage vs Random Training\n'
                'Validation matches training condition (95% CI, n=8 seeds)', 
                fontsize=16)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sample_sizes)
    
    # Add legend
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Dynamic y-axis limits and gridlines
    all_accuracies = []
    for key, data in metrics.items():
        all_accuracies.extend(data['accuracies'])
    
    min_acc = min(all_accuracies) if all_accuracies else 0
    y_min = max(0, min_acc - 0.02)  # 0.02 below min, but not below 0
    
    ax.set_ylim(y_min, 1)
    
    # Add horizontal gridlines every 0.1
    y_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3, axis='y', color='lightgray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

def print_summary_table(metrics):
    """Print summary table of results"""
    
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(f"{'Config':<30} {'Val Type':<12} {'Mean Acc':<10} {'Std Acc':<10} {'95% CI':<15} {'N':<3}")
    print("-"*90)
    
    for ((sample_size, approach), val_type), data in sorted(metrics.items()):
        config = f"{sample_size} {approach}"
        mean_acc = data['mean_accuracy']
        std_acc = data['std_accuracy']
        n_seeds = data['n_seeds']
        
        # Calculate 95% CI
        if n_seeds > 1:
            sem = std_acc / np.sqrt(n_seeds)
            ci_range = stats.t.ppf(0.975, n_seeds-1) * sem
            ci_str = f"±{ci_range:.3f}"
        else:
            ci_str = "N/A"
        
        print(f"{config:<30} {val_type:<12} {mean_acc:.4f}{'':>4} {std_acc:.4f}{'':>4} "
              f"{ci_str:<15} {n_seeds:<3}")

def main():
    parser = argparse.ArgumentParser(description='Plot Individual ID results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Directory to save output figure')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'individual_id_performance.png')
    
    print("="*60)
    print("Individual ID Pelage Ratio Analysis")
    print("="*60)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        return
    
    # Extract pelage metrics and create performance dataframe
    metrics, performance_df = extract_pelage_metrics(results)
    
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
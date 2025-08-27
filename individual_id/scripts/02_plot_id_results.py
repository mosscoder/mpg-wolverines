#!/usr/bin/env python3
"""
Script 02: Plot Individual ID Results
Creates line plots with 95% CI ribbons comparing pelage_only vs random_sample approaches.
Analyzes best cross-validated epoch across 8 seeds for each configuration.
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

def find_best_epochs(results):
    """Find best epoch for each configuration across seeds"""
    
    # Group results by (sample_size, approach)
    grouped = defaultdict(list)
    for result in results:
        key = (result['sample_size'], result['approach'])
        grouped[key].append(result)
    
    best_epochs = {}
    performance_data = []
    
    for (sample_size, approach), group_results in grouped.items():
        if len(group_results) < 8:
            print(f"Warning: Only {len(group_results)} seeds for samples={sample_size}, approach={approach}")
        
        # Check if we have val_history for epoch analysis
        if 'val_history' in group_results[0] and group_results[0]['val_history']:
            # Method 1: Find best epoch across all seeds using epoch-wise data
            all_epoch_accuracies = []
            
            for result in group_results:
                val_history = result['val_history']
                if val_history:
                    epoch_accs = [epoch_data['accuracy'] for epoch_data in val_history]
                    all_epoch_accuracies.append(epoch_accs)
            
            if all_epoch_accuracies:
                # Find epoch with highest mean accuracy across seeds
                min_epochs = min(len(accs) for accs in all_epoch_accuracies)
                mean_accuracies = []
                for epoch in range(min_epochs):
                    epoch_values = [accs[epoch] for accs in all_epoch_accuracies]
                    mean_accuracies.append(np.mean(epoch_values))
                
                best_epoch_idx = np.argmax(mean_accuracies)
                best_epoch = best_epoch_idx + 1  # Convert to 1-indexed
                
                # Get accuracies at best epoch for all seeds
                best_epoch_accuracies = [accs[best_epoch_idx] for accs in all_epoch_accuracies]
                
                print(f"samples={sample_size}, approach={approach}: Best epoch {best_epoch}, "
                      f"mean acc={np.mean(best_epoch_accuracies):.4f}")
                
                best_epochs[(sample_size, approach)] = {
                    'best_epoch': best_epoch,
                    'accuracies': best_epoch_accuracies,
                    'mean_accuracy': np.mean(best_epoch_accuracies),
                    'std_accuracy': np.std(best_epoch_accuracies, ddof=1),
                    'n_seeds': len(best_epoch_accuracies)
                }
                
                # Store for plotting
                for acc in best_epoch_accuracies:
                    performance_data.append({
                        'sample_size': sample_size,
                        'approach': approach,
                        'accuracy': acc,
                        'best_epoch': best_epoch
                    })
        else:
            # Method 2: Use best_val_accuracy from performance dict
            print(f"No val_history found for samples={sample_size}, approach={approach}, using best_val_accuracy")
            accuracies = []
            epochs = []
            
            for result in group_results:
                if 'performance' in result and 'best_val_accuracy' in result['performance']:
                    accuracies.append(result['performance']['best_val_accuracy'])
                    epochs.append(result['performance'].get('best_epoch', 'unknown'))
                
            if accuracies:
                best_epochs[(sample_size, approach)] = {
                    'best_epoch': 'mixed',  # Different epochs per seed
                    'accuracies': accuracies,
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies, ddof=1),
                    'n_seeds': len(accuracies),
                    'epochs': epochs
                }
                
                # Store for plotting
                for acc in accuracies:
                    performance_data.append({
                        'sample_size': sample_size,
                        'approach': approach,
                        'accuracy': acc,
                        'best_epoch': 'mixed'
                    })
    
    return best_epochs, pd.DataFrame(performance_data)

def plot_results(best_epochs, performance_df, output_path):
    """Create line plot with 95% CI ribbons"""
    
    plt.figure(figsize=(10, 6))
    
    sample_sizes = sorted(performance_df['sample_size'].unique())
    approaches = ['pelage_only', 'random_sample']
    colors = {'pelage_only': '#1f77b4', 'random_sample': '#ff7f0e'}
    
    for approach in approaches:
        means = []
        lower_cis = []
        upper_cis = []
        
        for sample_size in sample_sizes:
            key = (sample_size, approach)
            if key in best_epochs:
                data = best_epochs[key]
                mean_acc = data['mean_accuracy']
                accuracies = data['accuracies']
                
                # Calculate 95% confidence interval
                if len(accuracies) > 1:
                    sem = stats.sem(accuracies)  # Standard error of the mean
                    ci = stats.t.interval(0.95, len(accuracies)-1, loc=mean_acc, scale=sem)
                    lower_ci, upper_ci = ci
                else:
                    lower_ci = upper_ci = mean_acc
                
                means.append(mean_acc)
                lower_cis.append(lower_ci)
                upper_cis.append(upper_ci)
            else:
                # Fill with NaN if missing
                means.append(np.nan)
                lower_cis.append(np.nan)
                upper_cis.append(np.nan)
        
        # Plot line and confidence interval
        plt.plot(sample_sizes, means, 'o-', color=colors[approach], 
                label=approach.replace('_', ' ').title(), linewidth=2, markersize=6)
        plt.fill_between(sample_sizes, lower_cis, upper_cis, 
                        color=colors[approach], alpha=0.2)
    
    plt.xlabel('Samples per Individual (Training)', fontsize=12)
    plt.ylabel('Best Validation Accuracy', fontsize=12)
    plt.title('Individual ID Classification Performance\n(95% Confidence Intervals, n=8 seeds)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Set x-axis ticks
    plt.xticks(sample_sizes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

def print_summary_table(best_epochs):
    """Print summary table of results"""
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Config':<25} {'Best Epoch':<12} {'Mean Acc':<10} {'Std Acc':<10} {'95% CI':<15} {'N':<3}")
    print("-"*80)
    
    for (sample_size, approach), data in sorted(best_epochs.items()):
        config = f"{sample_size} {approach}"
        best_epoch = data['best_epoch']
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
        
        print(f"{config:<25} {str(best_epoch):<12} {mean_acc:.4f}{'':>4} {std_acc:.4f}{'':>4} "
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
    print("Individual ID Results Analysis")
    print("="*60)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        return
    
    # Find best epochs and create performance dataframe
    best_epochs, performance_df = find_best_epochs(results)
    
    if not best_epochs:
        print("No valid results found")
        return
    
    # Create plot
    plot_results(best_epochs, performance_df, output_path)
    
    # Print summary table
    print_summary_table(best_epochs)
    
    print(f"\nAnalysis complete! Figure saved to: {output_path}")

if __name__ == "__main__":
    main()
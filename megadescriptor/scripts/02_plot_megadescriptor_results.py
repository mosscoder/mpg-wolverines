#!/usr/bin/env python3
"""
Script 02: Plot MegaDescriptor Results
Creates visualizations comparing MegaDescriptor nearest neighbor performance
across pelage training approaches with similarity score analysis.
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

# Assume script is run from wolverines root directory
sys.path.append('.')


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


def extract_megadescriptor_metrics(results):
    """Extract F1 scores and similarity metrics from MegaDescriptor results"""
    
    # Group results by (sample_size, approach)
    grouped = defaultdict(list)
    for result in results:
        key = (result['sample_size'], result['approach'])
        grouped[key].append(result)
    
    metrics = {}
    performance_data = []
    
    for (sample_size, approach), group_results in grouped.items():
        if len(group_results) < 8:
            print(f"Warning: Only {len(group_results)} seeds for samples={sample_size}, approach={approach}")
        
        # Extract metrics
        overall_f1s = []
        visible_f1s = []
        invisible_f1s = []
        similarity_scores = []
        
        for result in group_results:
            # Overall F1 score from performance
            if 'performance' in result and 'final_val_f1_macro' in result['performance']:
                overall_f1s.append(result['performance']['final_val_f1_macro'])
            
            # Similarity scores
            if 'performance' in result and 'avg_similarity_score' in result['performance']:
                similarity_scores.append(result['performance']['avg_similarity_score'])
            
            # Pelage-specific F1 scores from final metrics
            if 'performance' in result and 'final_pelage_metrics' in result['performance']:
                pelage_metrics = result['performance']['final_pelage_metrics']
                
                if 'visible' in pelage_metrics and 'f1_score' in pelage_metrics['visible']:
                    visible_f1s.append(pelage_metrics['visible']['f1_score'])
                
                if 'invisible' in pelage_metrics and 'f1_score' in pelage_metrics['invisible']:
                    invisible_f1s.append(pelage_metrics['invisible']['f1_score'])
        
        # Store metrics for each validation type
        config_key = (sample_size, approach)
        
        if overall_f1s:
            metrics[(config_key, 'overall')] = {
                'f1_scores': overall_f1s,
                'mean_f1': np.mean(overall_f1s),
                'std_f1': np.std(overall_f1s, ddof=1),
                'n_seeds': len(overall_f1s)
            }
            
            # Add to performance data
            for f1 in overall_f1s:
                performance_data.append({
                    'sample_size': sample_size,
                    'approach': approach,
                    'val_type': 'overall',
                    'f1_score': f1
                })
        
        if visible_f1s:
            metrics[(config_key, 'visible')] = {
                'f1_scores': visible_f1s,
                'mean_f1': np.mean(visible_f1s),
                'std_f1': np.std(visible_f1s, ddof=1),
                'n_seeds': len(visible_f1s)
            }
        
        if invisible_f1s:
            metrics[(config_key, 'invisible')] = {
                'f1_scores': invisible_f1s,
                'mean_f1': np.mean(invisible_f1s),
                'std_f1': np.std(invisible_f1s, ddof=1),
                'n_seeds': len(invisible_f1s)
            }
        
        # Store similarity metrics
        if similarity_scores:
            metrics[(config_key, 'similarity')] = {
                'scores': similarity_scores,
                'mean_score': np.mean(similarity_scores),
                'std_score': np.std(similarity_scores, ddof=1),
                'n_seeds': len(similarity_scores)
            }
        
        print(f"samples={sample_size}, approach={approach}: "
              f"F1={np.mean(overall_f1s):.4f}, Sim={np.mean(similarity_scores):.4f} (n={len(overall_f1s)} seeds)")
    
    return metrics, pd.DataFrame(performance_data)


def plot_results(metrics, performance_df, output_path):
    """Create dual subplot with F1 scores and similarity scores"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    sample_sizes = sorted(performance_df['sample_size'].unique())
    approaches = ['pelage', 'pelage_abs']
    
    # Colors for training approaches
    colors = {
        'pelage': '#27ae60',     # Green - pelage visible training
        'pelage_abs': '#e74c3c'  # Red - pelage absent training
    }
    
    # Plot F1 scores (top subplot) and collect CI bounds for y-axis scaling
    all_ci_lower = []
    all_ci_upper = []
    
    for approach in approaches:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []
        
        for sample_size in sample_sizes:
            key = ((sample_size, approach), 'overall')
            
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
        
        if x_vals:
            # Plot F1 line with confidence interval
            ax1.plot(x_vals, y_vals, color=colors[approach], linewidth=2.5, 
                    marker='o', markersize=8, label=approach)
            ax1.fill_between(x_vals, ci_lower, ci_upper, 
                           color=colors[approach], alpha=0.2)
    
    # Plot similarity scores (bottom subplot)
    for approach in approaches:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []
        
        for sample_size in sample_sizes:
            key = ((sample_size, approach), 'similarity')
            
            if key in metrics:
                data = metrics[key]
                mean_sim = data['mean_score']
                sim_scores = data['scores']
                
                # Calculate 95% confidence interval
                if len(sim_scores) > 1:
                    sem = stats.sem(sim_scores)
                    ci_range = stats.t.ppf(0.975, len(sim_scores)-1) * sem
                else:
                    ci_range = 0
                
                x_vals.append(sample_size)
                y_vals.append(mean_sim)
                ci_lower.append(mean_sim - ci_range)
                ci_upper.append(mean_sim + ci_range)
        
        if x_vals:
            # Plot similarity line with confidence interval
            ax2.plot(x_vals, y_vals, color=colors[approach], linewidth=2.5, 
                    marker='s', markersize=8, label=approach)
            ax2.fill_between(x_vals, ci_lower, ci_upper, 
                           color=colors[approach], alpha=0.2)
    
    # Styling for F1 subplot with dynamic y-limits
    ax1.set_ylabel('Validation F1 Score (Macro)', fontsize=14)
    ax1.set_xticks(range(2, 34, 2))  # Every 2 from 2 to 32
    ax1.set_xlim(0, 34)
    
    # Set dynamic y-limits based on confidence interval bounds
    if all_ci_lower and all_ci_upper:
        min_f1 = min(all_ci_lower)
        max_f1 = max(all_ci_upper)
        y_min = max(0.0, min_f1 - 0.02)  # Don't go below 0
        y_max = min(1.0, max_f1 + 0.02)  # Don't go above 1
        ax1.set_ylim(y_min, y_max)
    else:
        ax1.set_ylim(0.0, 1.0)  # Fallback
    
    ax1.grid(True, alpha=0.3, axis='y', color='lightgray')
    ax1.legend(title="Training image quality:", loc='lower right')
    ax1.set_title('MegaDescriptor Nearest Neighbor Classification Performance', fontsize=16)
    
    # Styling for similarity subplot
    ax2.set_xlabel('Image Count per Individual', fontsize=14)
    ax2.set_ylabel('Average Cosine Similarity', fontsize=14)
    ax2.set_xticks(range(2, 34, 2))  # Every 2 from 2 to 32
    ax2.set_xlim(0, 34)
    ax2.grid(True, alpha=0.3, axis='y', color='lightgray')
    ax2.legend(title="Training image quality:", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")


def print_summary_table(metrics):
    """Print summary table of F1 and similarity results"""
    
    print("\n" + "="*100)
    print("MEGADESCRIPTOR SUMMARY TABLE")
    print("="*100)
    print(f"{'Config':<25} {'Mean F1':<10} {'Std F1':<10} {'95% CI':<15} {'Mean Sim':<10} {'Std Sim':<10} {'N':<3}")
    print("-"*100)
    
    # Collect data for organized display
    config_data = {}
    for ((sample_size, approach), val_type), data in metrics.items():
        config_key = (sample_size, approach)
        if config_key not in config_data:
            config_data[config_key] = {}
        config_data[config_key][val_type] = data
    
    # Display overall results
    for (sample_size, approach) in sorted(config_data.keys()):
        config = f"{sample_size} {approach}"
        data = config_data[(sample_size, approach)]
        
        if 'overall' in data:
            f1_data = data['overall']
            mean_f1 = f1_data['mean_f1']
            std_f1 = f1_data['std_f1']
            n_seeds = f1_data['n_seeds']
            
            # Calculate F1 CI
            if n_seeds > 1:
                sem_f1 = std_f1 / np.sqrt(n_seeds)
                ci_f1_range = stats.t.ppf(0.975, n_seeds-1) * sem_f1
                ci_f1_str = f"±{ci_f1_range:.3f}"
            else:
                ci_f1_str = "N/A"
            
            # Similarity data
            if 'similarity' in data:
                sim_data = data['similarity']
                mean_sim = sim_data['mean_score']
                std_sim = sim_data['std_score']
            else:
                mean_sim, std_sim = 0.0, 0.0
            
            print(f"{config:<25} {mean_f1:.4f}{'':>4} {std_f1:.4f}{'':>4} "
                  f"{ci_f1_str:<15} {mean_sim:.4f}{'':>4} {std_sim:.4f}{'':>4} {n_seeds:<3}")


def main():
    parser = argparse.ArgumentParser(description='Plot MegaDescriptor results')
    parser.add_argument('--results_dir', type=str, default='megadescriptor/results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='megadescriptor/figures',
                       help='Directory to save output figure')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'megadescriptor_performance.png')
    
    print("="*60)
    print("MegaDescriptor Nearest Neighbor Analysis")
    print("="*60)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        return
    
    # Extract metrics and create performance dataframe
    metrics, performance_df = extract_megadescriptor_metrics(results)
    
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
#!/usr/bin/env python3
"""
Script 07: Compare LoRA vs Linear Model Performance
Analyze and compare performance between LoRA fine-tuning and linear-only training
for individual ID classification with 32 samples per class (label==1 only).
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(wolverines_root)


def find_best_epoch(val_history, metric='accuracy'):
    """Find best epoch from validation history
    
    Args:
        val_history: List of epoch results with accuracy/loss
        metric: Metric to optimize ('accuracy' or 'loss')
        
    Returns:
        Dict with best epoch info
    """
    if not val_history:
        return None
    
    if metric == 'accuracy':
        best_idx = max(range(len(val_history)), key=lambda i: val_history[i]['accuracy'])
        best_value = val_history[best_idx]['accuracy']
    else:  # loss
        best_idx = min(range(len(val_history)), key=lambda i: val_history[i]['loss'])
        best_value = val_history[best_idx]['loss']
    
    return {
        'epoch': val_history[best_idx]['epoch'],
        'accuracy': val_history[best_idx]['accuracy'],
        'loss': val_history[best_idx]['loss'],
        'value': best_value
    }


def load_linear_results(results_dir='results'):
    """Load linear model results for 32 samples, pelage approach
    
    Returns:
        List of result dicts
    """
    pattern = os.path.join(results_dir, "samples=32_approach=pelage_seed=*.json")
    files = glob.glob(pattern)
    
    results = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Verify this is the right experiment
            if (result.get('sample_size') == 32 and 
                result.get('approach') == 'pelage'):
                results.append(result)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(results)} linear model results (32 samples, pelage)")
    return results


def load_lora_results(results_dir='results/lora'):
    """Load LoRA model results
    
    Returns:
        List of result dicts
    """
    pattern = os.path.join(results_dir, "lora_r=*_alpha=*_seed=*.json")
    files = glob.glob(pattern)
    
    results = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            results.append(result)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(results)} LoRA model results")
    return results


def calculate_statistics(values, confidence=0.95):
    """Calculate mean and confidence interval
    
    Args:
        values: List of numeric values
        confidence: Confidence level (0.95 for 95% CI)
        
    Returns:
        Dict with mean, std, and CI bounds
    """
    if not values:
        return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0, 'n': 0}
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    n = len(values)
    
    # Calculate 95% CI using t-distribution
    try:
        from scipy.stats import t
        alpha = 1 - confidence
        df = n - 1 if n > 1 else 1
        t_critical = t.ppf(1 - alpha/2, df)
        margin = t_critical * (std / np.sqrt(n))
    except ImportError:
        # Fallback to normal approximation if scipy not available
        z_critical = 1.96  # 95% CI
        margin = z_critical * (std / np.sqrt(n)) if n > 0 else 0
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'n': n,
        'margin': margin
    }


def analyze_linear_results(linear_results):
    """Analyze linear model results
    
    Returns:
        Dict with statistics for best epoch and final epoch
    """
    final_accuracies = []
    best_epoch_accuracies = []
    best_epochs = []
    training_times = []
    
    for result in linear_results:
        # Final epoch accuracy
        final_acc = result['performance']['final_val_accuracy']
        final_accuracies.append(final_acc)
        
        # Best epoch accuracy
        val_history = result.get('val_history', [])
        best_epoch = find_best_epoch(val_history)
        if best_epoch:
            best_epoch_accuracies.append(best_epoch['accuracy'])
            best_epochs.append(best_epoch['epoch'])
        
        # Training time
        training_times.append(result.get('training_time', 0) / 60)  # Convert to minutes
    
    return {
        'final_epoch': calculate_statistics(final_accuracies),
        'best_epoch': calculate_statistics(best_epoch_accuracies),
        'best_epoch_nums': calculate_statistics(best_epochs),
        'training_time': calculate_statistics(training_times),
        'n_runs': len(linear_results)
    }


def analyze_lora_results(lora_results):
    """Analyze LoRA model results, grouped by (r, alpha) configuration
    
    Returns:
        Dict with overall statistics and per-config breakdown
    """
    # Group by (r, alpha) configuration
    grouped_results = defaultdict(list)
    all_final_accuracies = []
    all_best_epoch_accuracies = []
    all_best_epochs = []
    all_training_times = []
    
    for result in lora_results:
        lora_r = result.get('lora_r')
        lora_alpha = result.get('lora_alpha')
        config_key = f"r={lora_r}_alpha={lora_alpha}"
        
        # Final epoch accuracy
        final_acc = result['performance']['final_val_accuracy']
        all_final_accuracies.append(final_acc)
        
        # Best epoch accuracy
        val_history = result.get('val_history', [])
        best_epoch = find_best_epoch(val_history)
        if best_epoch:
            all_best_epoch_accuracies.append(best_epoch['accuracy'])
            all_best_epochs.append(best_epoch['epoch'])
        
        # Training time
        training_time = result.get('training_time', 0) / 60  # Convert to minutes
        all_training_times.append(training_time)
        
        # Store in grouped results
        grouped_results[config_key].append({
            'final_acc': final_acc,
            'best_epoch_acc': best_epoch['accuracy'] if best_epoch else final_acc,
            'best_epoch_num': best_epoch['epoch'] if best_epoch else 50,
            'training_time': training_time,
            'seed': result.get('seed'),
            'lora_r': lora_r,
            'lora_alpha': lora_alpha
        })
    
    # Calculate per-config statistics
    config_stats = {}
    for config_key, config_results in grouped_results.items():
        final_accs = [r['final_acc'] for r in config_results]
        best_accs = [r['best_epoch_acc'] for r in config_results]
        best_epochs = [r['best_epoch_num'] for r in config_results]
        times = [r['training_time'] for r in config_results]
        
        config_stats[config_key] = {
            'final_epoch': calculate_statistics(final_accs),
            'best_epoch': calculate_statistics(best_accs),
            'best_epoch_nums': calculate_statistics(best_epochs),
            'training_time': calculate_statistics(times),
            'n_runs': len(config_results),
            'lora_r': config_results[0]['lora_r'],
            'lora_alpha': config_results[0]['lora_alpha']
        }
    
    # Overall LoRA statistics
    overall = {
        'final_epoch': calculate_statistics(all_final_accuracies),
        'best_epoch': calculate_statistics(all_best_epoch_accuracies),
        'best_epoch_nums': calculate_statistics(all_best_epochs),
        'training_time': calculate_statistics(all_training_times),
        'n_runs': len(lora_results),
        'n_configs': len(config_stats)
    }
    
    return overall, config_stats


def print_comparison_table(linear_stats, lora_overall, lora_configs):
    """Print formatted comparison table"""
    
    print("=" * 90)
    print("Individual ID: LoRA vs Linear Model Comparison (32 samples/class, label==1 only)")
    print("=" * 90)
    print()
    
    # Main comparison table
    print(f"{'Model Type':<25} | {'Best Epoch Acc':<15} | {'Final Epoch Acc':<16} | {'Best Epoch':<12} | {'Time (min)':<12}")
    print("-" * 90)
    
    # Linear baseline
    linear_best = linear_stats['best_epoch']
    linear_final = linear_stats['final_epoch']
    linear_epoch = linear_stats['best_epoch_nums']
    linear_time = linear_stats['training_time']
    
    print(f"{'Linear (baseline)':<25} | "
          f"{linear_best['mean']:.3f} ± {linear_best['margin']:.3f} | "
          f"{linear_final['mean']:.3f} ± {linear_final['margin']:.3f} | "
          f"{linear_epoch['mean']:.1f} ± {linear_epoch['margin']:.1f} | "
          f"{linear_time['mean']:.1f} ± {linear_time['margin']:.1f}")
    
    # LoRA overall
    lora_best = lora_overall['best_epoch']
    lora_final = lora_overall['final_epoch']
    lora_epoch = lora_overall['best_epoch_nums']
    lora_time = lora_overall['training_time']
    
    print(f"{'LoRA (all configs)':<25} | "
          f"{lora_best['mean']:.3f} ± {lora_best['margin']:.3f} | "
          f"{lora_final['mean']:.3f} ± {lora_final['margin']:.3f} | "
          f"{lora_epoch['mean']:.1f} ± {lora_epoch['margin']:.1f} | "
          f"{lora_time['mean']:.1f} ± {lora_time['margin']:.1f}")
    
    print()
    
    # Best LoRA configurations (top 5 by best epoch accuracy)
    sorted_configs = sorted(lora_configs.items(), 
                           key=lambda x: x[1]['best_epoch']['mean'], 
                           reverse=True)
    
    print("Top LoRA Configurations (by best epoch accuracy):")
    print(f"{'Config':<15} | {'Best Epoch Acc':<15} | {'Final Epoch Acc':<16} | {'Best Epoch':<12} | {'Time (min)':<12}")
    print("-" * 90)
    
    for config_key, stats in sorted_configs[:5]:
        config_best = stats['best_epoch']
        config_final = stats['final_epoch']
        config_epoch = stats['best_epoch_nums']
        config_time = stats['training_time']
        
        print(f"{config_key:<15} | "
              f"{config_best['mean']:.3f} ± {config_best['margin']:.3f} | "
              f"{config_final['mean']:.3f} ± {config_final['margin']:.3f} | "
              f"{config_epoch['mean']:.1f} ± {config_epoch['margin']:.1f} | "
              f"{config_time['mean']:.1f} ± {config_time['margin']:.1f}")
    
    print()
    
    # Performance improvement
    improvement = lora_best['mean'] - linear_best['mean']
    print(f"Performance Summary:")
    print(f"  Linear best epoch accuracy: {linear_best['mean']:.4f} ± {linear_best['margin']:.4f} (n={linear_best['n']})")
    print(f"  LoRA best epoch accuracy:   {lora_best['mean']:.4f} ± {lora_best['margin']:.4f} (n={lora_best['n']})")
    print(f"  Improvement: {improvement:+.4f} accuracy points")
    print(f"  Best LoRA config: {sorted_configs[0][0]}")


def save_comparison_results(linear_stats, lora_overall, lora_configs, output_path):
    """Save detailed comparison to JSON file"""
    
    comparison_data = {
        'experiment_type': 'individual_id_lora_vs_linear',
        'experiment_config': {
            'samples_per_individual': 32,
            'approach': 'pelage_only',
            'label_filter': 'label==1',
            'epochs': 50
        },
        'linear_model': linear_stats,
        'lora_overall': lora_overall, 
        'lora_configs': lora_configs,
        'summary': {
            'linear_best_accuracy': linear_stats['best_epoch']['mean'],
            'lora_best_accuracy': lora_overall['best_epoch']['mean'],
            'improvement': lora_overall['best_epoch']['mean'] - linear_stats['best_epoch']['mean'],
            'best_lora_config': max(lora_configs.items(), 
                                  key=lambda x: x[1]['best_epoch']['mean'])[0]
        },
        'timestamp': time.time()
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"✓ Detailed comparison saved to: {output_path}")


def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Compare LoRA vs Linear model performance')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Base results directory')
    parser.add_argument('--output_path', type=str, 
                       default='results/lora_vs_linear_comparison.json',
                       help='Output path for comparison results')
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("LoRA vs Linear Model Performance Comparison")
    print("=" * 90)
    
    # Load results
    print("Loading experiment results...")
    linear_results = load_linear_results(args.results_dir)
    lora_results = load_lora_results(os.path.join(args.results_dir, 'lora'))
    
    if not linear_results:
        print("Error: No linear model results found. Run script 01 with 32 samples first.")
        return
    
    if not lora_results:
        print("Error: No LoRA results found. Run script 06 first.")
        return
    
    # Analyze results
    print("Analyzing linear model performance...")
    linear_stats = analyze_linear_results(linear_results)
    
    print("Analyzing LoRA model performance...")
    lora_overall, lora_configs = analyze_lora_results(lora_results)
    
    # Print comparison table
    print_comparison_table(linear_stats, lora_overall, lora_configs)
    
    # Save detailed results
    save_comparison_results(linear_stats, lora_overall, lora_configs, args.output_path)
    
    print(f"\nComparison complete!")
    print(f"Linear runs: {linear_stats['n_runs']}")
    print(f"LoRA runs: {lora_overall['n_runs']} (across {lora_overall['n_configs']} configs)")


if __name__ == "__main__":
    main()
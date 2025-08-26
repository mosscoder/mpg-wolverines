#!/usr/bin/env python3
"""
Script 03: Make Figures
Generate visualizations for the pelage sorting experiments:
- Bar charts with 95% CI for resize size sweep (script 00)
- Line plots with CI ribbons for learning rate sweep (script 01)
- Performance curves for final test (script 02)
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd
from typing import List, Dict, Any

# Add utils to path (assumes we cd to pelage_sorting in sbatch)
script_dir = os.path.dirname(os.path.abspath(__file__))
wolverines_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to wolverines root
sys.path.append(wolverines_root)

from utils.training import load_results


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for a list of values"""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    
    sem = stats.sem(values)  # Standard error of mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    
    return mean, ci, sem


def load_and_aggregate_results(results_dir: str, group_by: List[str]) -> Dict[tuple, List[Dict]]:
    """Load results and group by specified parameters"""
    results = load_results(results_dir)
    
    grouped = {}
    for result in results:
        # Create grouping key
        key_values = []
        for param in group_by:
            if param in result['params']:
                key_values.append(result['params'][param])
            elif param in result:
                key_values.append(result[param])
            else:
                key_values.append(None)
        
        key = tuple(key_values)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    return grouped


def make_resize_figure(results_dir: str, output_path: str):
    """Create bar chart for resize size sweep results"""
    print("Creating resize size figure...")
    
    # Load and group results by resize_size
    grouped = load_and_aggregate_results(results_dir, ['resize_size'])
    
    if not grouped:
        print(f"No results found in {results_dir}")
        return
    
    # Extract data for plotting
    resize_sizes = []
    f1_means = []
    f1_cis = []
    
    for (resize_size,), results in sorted(grouped.items()):
        if resize_size is None:
            continue
        
        f1_scores = [r['final_val_f1'] for r in results]
        mean, ci, _ = calculate_confidence_interval(f1_scores)
        
        resize_sizes.append(resize_size)
        f1_means.append(mean)
        f1_cis.append(ci)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(resize_sizes)), f1_means, yerr=f1_cis, 
                   capsize=5, alpha=0.7, color='steelblue')
    
    plt.xlabel('Resize Size', fontsize=14)
    plt.ylabel('Validation F1 Score', fontsize=14)
    plt.title('Resize Size Sweep: Effect of Image Resize Size on Performance', fontsize=16)
    plt.xticks(range(len(resize_sizes)), resize_sizes)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, ci) in enumerate(zip(bars, f1_means, f1_cis)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Resize size figure saved to: {output_path}")



def make_learning_rate_figure(results_dir: str, output_path: str):
    """Create line plots for learning rate sweep with epoch progression"""
    print("Creating learning rate figure...")
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    # Group by learning rate
    lr_groups = {}
    for result in results:
        lr = result.get('learning_rate')
        if lr is None:
            continue
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(result)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Learning curves for each LR
    colors = plt.cm.Set3(np.linspace(0, 1, len(lr_groups)))
    
    for (lr, lr_results), color in zip(sorted(lr_groups.items()), colors):
        # Collect all validation histories
        all_val_f1s = []
        max_epochs = 0
        
        for result in lr_results:
            if 'val_history' in result and result['val_history']:
                val_f1s = [epoch['f1_score'] for epoch in result['val_history']]
                all_val_f1s.append(val_f1s)
                max_epochs = max(max_epochs, len(val_f1s))
        
        if not all_val_f1s:
            continue
        
        # Pad sequences to same length
        padded_f1s = []
        for f1s in all_val_f1s:
            padded = f1s + [f1s[-1]] * (max_epochs - len(f1s))
            padded_f1s.append(padded)
        
        # Calculate mean and CI across folds
        epochs = range(1, max_epochs + 1)
        mean_f1s = np.mean(padded_f1s, axis=0)
        std_f1s = np.std(padded_f1s, axis=0)
        
        # Plot mean line
        ax1.plot(epochs, mean_f1s, label=f'LR={lr:.0e}', color=color, linewidth=2)
        
        # Plot confidence ribbon
        ax1.fill_between(epochs, mean_f1s - std_f1s, mean_f1s + std_f1s, 
                        alpha=0.2, color=color)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation F1 Score', fontsize=12)
    ax1.set_title('Learning Rate Sweep: Validation F1 vs Epochs', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Best F1 score for each learning rate
    lrs = []
    best_f1_means = []
    best_f1_cis = []
    
    for lr, lr_results in sorted(lr_groups.items()):
        best_f1s = [r['best_val_f1'] for r in lr_results]
        mean, ci, _ = calculate_confidence_interval(best_f1s)
        
        lrs.append(lr)
        best_f1_means.append(mean)
        best_f1_cis.append(ci)
    
    ax2.errorbar(range(len(lrs)), best_f1_means, yerr=best_f1_cis, 
                fmt='o-', capsize=5, linewidth=2, markersize=8, color='darkblue')
    
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Best Validation F1 Score', fontsize=12)
    ax2.set_title('Learning Rate Sweep: Best Performance', fontsize=14)
    ax2.set_xticks(range(len(lrs)))
    ax2.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
    ax2.grid(alpha=0.3)
    
    # Add value labels
    for i, (mean, ci) in enumerate(zip(best_f1_means, best_f1_cis)):
        ax2.text(i, mean + ci + 0.005, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning rate figure saved to: {output_path}")


def make_final_performance_figure(results_dir: str, output_path: str):
    """Create performance curve for final test results"""
    print("Creating final performance figure...")
    
    # Load final results
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    # Should only be one result
    final_result = results[0] if results else None
    
    if not final_result or 'test_history' not in final_result:
        print("No final test results found")
        return
    
    # Extract training and test histories
    train_history = final_result.get('train_history', [])
    test_history = final_result.get('test_history', [])
    
    epochs = range(1, len(train_history) + 1)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Loss curves
    train_losses = [epoch['loss'] for epoch in train_history]
    test_losses = [epoch['loss'] for epoch in test_history]
    
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(epochs, test_losses, label='Test Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: F1 Score
    train_f1s = [epoch['f1_score'] for epoch in train_history]
    test_f1s = [epoch['f1_score'] for epoch in test_history]
    
    ax2.plot(epochs, train_f1s, label='Training F1', color='blue', linewidth=2)
    ax2.plot(epochs, test_f1s, label='Test F1', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Test F1 Score')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Accuracy
    train_accs = [epoch['accuracy'] for epoch in train_history]
    test_accs = [epoch['accuracy'] for epoch in test_history]
    
    ax3.plot(epochs, train_accs, label='Training Accuracy', color='blue', linewidth=2)
    ax3.plot(epochs, test_accs, label='Test Accuracy', color='red', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training and Test Accuracy')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Final metrics summary
    final_metrics = final_result['final_performance']
    metrics = ['final_test_f1', 'final_test_precision', 'final_test_recall', 'final_test_accuracy']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
    values = [final_metrics[metric] for metric in metrics]
    
    bars = ax4.bar(metric_names, values, color=['steelblue', 'darkgreen', 'orange', 'darkred'], alpha=0.7)
    ax4.set_ylabel('Score')
    ax4.set_title('Final Test Performance')
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Final performance figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate figures for pelage sorting experiments')
    parser.add_argument('--results_base_dir', type=str,
                       default='results',
                       help='Base directory containing all results')
    parser.add_argument('--output_dir', type=str,
                       default='figures',
                       help='Output directory for figures')
    parser.add_argument('--experiments', nargs='+',
                       choices=['00_resize', '01_learning_rate', '02_test'],
                       default=['00_resize', '01_learning_rate', '02_test'],
                       help='Which experiments to create figures for')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING PELAGE SORTING FIGURES")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Generate figures for each experiment
    if '00_resize' in args.experiments:
        results_dir = os.path.join(args.results_base_dir, '00_resize')
        output_path = os.path.join(args.output_dir, '00_resize_results.png')
        make_resize_figure(results_dir, output_path)
    
    if '01_learning_rate' in args.experiments:
        results_dir = os.path.join(args.results_base_dir, '01_learning_rate')
        output_path = os.path.join(args.output_dir, '01_learning_rate_results.png')
        make_learning_rate_figure(results_dir, output_path)
    
    if '02_test' in args.experiments:
        results_dir = os.path.join(args.results_base_dir, '02_test')
        output_path = os.path.join(args.output_dir, '02_final_performance.png')
        make_final_performance_figure(results_dir, output_path)
    
    print(f"\n✓ All figures generated successfully!")
    print(f"✓ Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
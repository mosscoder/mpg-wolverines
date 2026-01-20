#!/usr/bin/env python3
"""
Script 01: Plot Gallery Hygiene Sweep Results

Generates publication-quality figures from hygiene filter experiment.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) Line plot: Recall@1 vs gallery_size, lines by threshold value (viridis)
B) Heatmap: threshold × gallery_size → Recall@1 parameter space
C) Bar chart: % improvement over baseline (threshold=0.0)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

sys.path.append('.')
from utils.results import ResultsCollection


def load_hygiene_results(results_dir: str) -> ResultsCollection:
    """Load hygiene sweep results from JSON files."""
    pattern = os.path.join(results_dir, "threshold=*_gallery=*_seed=*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No result files found matching {pattern}")
        return ResultsCollection([])

    print(f"Found {len(json_files)} result files")

    results = []
    failed = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Extract key fields for easier access
                data['threshold'] = data['config']['threshold']
                data['gallery_size'] = data['config']['gallery_size']
                data['seed'] = data['config']['seed']
                results.append(data)
        except Exception as e:
            failed.append((json_file, str(e)))

    if failed:
        print(f"Failed to load {len(failed)} files")

    print(f"Successfully loaded {len(results)} results")
    return ResultsCollection(results)


def find_best_epoch_for_overall_recall(all_histories: List[List[dict]]) -> Tuple[int, float]:
    """Find epoch with best mean overall recall across seeds.

    Supports both old format (val_recall_at_1) and new format (query_quality_metrics).
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0

    # Determine which format we have
    sample_entry = all_histories[0][0]
    has_new_format = 'query_quality_metrics' in sample_entry
    has_old_format = 'val_recall_at_1' in sample_entry

    if has_new_format:
        eval_epochs = [h['epoch'] for h in all_histories[0] if 'query_quality_metrics' in h]
    elif has_old_format:
        eval_epochs = [h['epoch'] for h in all_histories[0] if 'val_recall_at_1' in h]
    else:
        return 50, 0.0

    best_epoch = None
    best_mean = -1

    for epoch in eval_epochs:
        recalls = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch:
                    if has_new_format and 'query_quality_metrics' in h:
                        # New format: use q>=0.0 as overall recall
                        recalls.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                    elif has_old_format and 'val_recall_at_1' in h:
                        # Old format
                        recalls.append(h['val_recall_at_1'])
                    break

        if recalls:
            mean_recall = np.mean(recalls)
            if mean_recall > best_mean:
                best_mean = mean_recall
                best_epoch = epoch

    return best_epoch or 50, best_mean


def extract_metrics(results: ResultsCollection) -> dict:
    """
    Extract aggregated metrics for plotting using best-epoch selection.

    For each (threshold, gallery_size) config, finds the epoch with best mean Recall@1
    across all 8 seeds, then reports that epoch's metrics.

    Supports both old format (val_recall_at_1) and new format (query_quality_metrics).
    """
    groups = results.group_by('threshold', 'gallery_size')

    metrics = {}
    for (threshold, gallery_size), group in groups.items():
        all_histories = [r.get('epoch_history', []) for r in group]

        if not all_histories or not all_histories[0]:
            continue

        # Determine format
        sample_entry = all_histories[0][0]
        has_new_format = 'query_quality_metrics' in sample_entry
        has_old_format = 'val_recall_at_1' in sample_entry

        # Find best epoch for overall recall
        best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

        # Extract recall values at best epoch
        recall_values = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == best_epoch:
                    if has_new_format and 'query_quality_metrics' in h:
                        recall_values.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                    elif has_old_format and 'val_recall_at_1' in h:
                        recall_values.append(h['val_recall_at_1'])
                    break

        if recall_values:
            metrics[(threshold, gallery_size)] = {
                'recall_at_1_values': recall_values,
                'mean_recall_at_1': np.mean(recall_values),
                'std_recall_at_1': np.std(recall_values, ddof=1) if len(recall_values) > 1 else 0,
                'n_seeds': len(recall_values),
                'best_epoch': best_epoch
            }

    return metrics


def plot_recall_vs_gallery_size(metrics: dict, output_path: str):
    """
    Panel A: Line plot of Recall@1 vs gallery_size.
    Lines colored by threshold value using viridis palette (0.0=dark, 0.5=bright).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique values
    thresholds = sorted(set(k[0] for k in metrics.keys()))
    gallery_sizes = sorted(set(k[1] for k in metrics.keys()))

    # Viridis palette: darker for lower thresholds, brighter for higher
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))
    color_map = {t: colors[i] for i, t in enumerate(thresholds)}

    legend_handles = []
    legend_labels = []

    for threshold in thresholds:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []

        for gsize in gallery_sizes:
            key = (threshold, gsize)
            if key in metrics:
                data = metrics[key]
                mean = data['mean_recall_at_1']
                values = data['recall_at_1_values']

                # 95% CI
                if len(values) > 1:
                    sem = stats.sem(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * sem
                else:
                    ci = 0

                x_vals.append(gsize)
                y_vals.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if x_vals:
            line, = ax.plot(x_vals, y_vals, color=color_map[threshold], linewidth=2.5,
                           marker='o', markersize=8)
            ax.fill_between(x_vals, ci_lower, ci_upper, color=color_map[threshold], alpha=0.2)

            legend_handles.append(line)
            if threshold == 0.0:
                legend_labels.append(f't = {threshold} (baseline)')
            else:
                legend_labels.append(f't = {threshold}')

    ax.set_xlabel('Gallery Size (samples per individual)', fontsize=14)
    ax.set_ylabel('Recall@1', fontsize=14)
    ax.set_title('Gallery Hygiene: Recall@1 vs Gallery Size by Quality Threshold\n'
                 '(Best epoch by cross-seed validation, 95% CI from 8 seeds)', fontsize=16, pad=20)
    ax.set_xticks(gallery_sizes)
    ax.grid(True, alpha=0.3, axis='y')

    legend = ax.legend(legend_handles, legend_labels, loc='lower right')
    legend.set_title('Threshold', prop={'weight': 'bold'})

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_parameter_heatmap(metrics: dict, output_path: str):
    """
    Panel B: Heatmap of threshold × gallery_size → Recall@1.
    Shows parameter space overview.
    """
    thresholds = sorted(set(k[0] for k in metrics.keys()))
    gallery_sizes = sorted(set(k[1] for k in metrics.keys()))

    # Create matrix
    recall_matrix = np.zeros((len(thresholds), len(gallery_sizes)))

    for i, threshold in enumerate(thresholds):
        for j, gsize in enumerate(gallery_sizes):
            key = (threshold, gsize)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=gallery_sizes, yticklabels=[f'{t:.1f}' for t in thresholds],
                cbar_kws={'label': 'Recall@1'}, ax=ax)

    ax.set_xlabel('Gallery Size (samples per individual)', fontsize=12)
    ax.set_ylabel('Quality Threshold', fontsize=12)
    ax.set_title('Parameter Space: Mean Recall@1 at Best Epoch', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_improvement_over_baseline(metrics: dict, output_path: str):
    """
    Panel C: Bar chart showing % improvement over baseline (threshold=0.0).
    For each gallery_size, shows improvement from filtering.
    """
    thresholds = sorted(set(k[0] for k in metrics.keys()))
    gallery_sizes = sorted(set(k[1] for k in metrics.keys()))

    # Compute improvement for each (threshold, gallery_size) relative to baseline
    improvements = {}
    for threshold in thresholds:
        if threshold == 0.0:
            continue
        for gsize in gallery_sizes:
            baseline_key = (0.0, gsize)
            current_key = (threshold, gsize)
            if baseline_key in metrics and current_key in metrics:
                baseline = metrics[baseline_key]['mean_recall_at_1']
                current = metrics[current_key]['mean_recall_at_1']
                if baseline > 0:
                    pct_improvement = ((current - baseline) / baseline) * 100
                    improvements[(threshold, gsize)] = pct_improvement

    if not improvements:
        print("No improvement data available")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    non_baseline_thresholds = [t for t in thresholds if t > 0]
    x = np.arange(len(gallery_sizes))
    width = 0.8 / len(non_baseline_thresholds)

    # Viridis colors for non-baseline thresholds
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(non_baseline_thresholds)))

    for i, threshold in enumerate(non_baseline_thresholds):
        offset = (i - len(non_baseline_thresholds) / 2 + 0.5) * width
        values = [improvements.get((threshold, gs), 0) for gs in gallery_sizes]
        ax.bar(x + offset, values, width, label=f't = {threshold}', color=colors[i])

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(gallery_sizes)
    ax.set_xlabel('Gallery Size (samples per individual)', fontsize=12)
    ax.set_ylabel('% Improvement over Baseline (t=0.0)', fontsize=12)
    ax.set_title('Improvement from Gallery Hygiene Filtering', fontsize=14, pad=15)
    ax.legend(title='Threshold', loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_learning_curves(results: ResultsCollection, output_path: str, gallery_size: int = 64):
    """
    Optional: Learning curves showing training progress for a specific gallery size.
    Supports both old format (val_recall_at_1) and new format (query_quality_metrics).
    """
    thresholds = sorted(results.get_unique('threshold'))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

    for i, threshold in enumerate(thresholds):
        # Get all seeds for this threshold and gallery_size
        filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
        if len(filtered) == 0:
            continue

        # Aggregate epoch histories
        all_histories = [r.get('epoch_history', []) for r in filtered]
        if not all_histories or not all_histories[0]:
            continue

        # Determine format
        sample_entry = all_histories[0][0]
        has_new_format = 'query_quality_metrics' in sample_entry
        has_old_format = 'val_recall_at_1' in sample_entry

        epochs = [h['epoch'] for h in all_histories[0]]
        mean_recalls = []
        ci_lower = []
        ci_upper = []

        for epoch in epochs:
            recalls = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == epoch:
                        if has_new_format and 'query_quality_metrics' in h:
                            recalls.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                        elif has_old_format and 'val_recall_at_1' in h:
                            recalls.append(h['val_recall_at_1'])
                        break

            if recalls:
                mean = np.mean(recalls)
                if len(recalls) > 1:
                    sem = stats.sem(recalls)
                    ci = stats.t.ppf(0.975, len(recalls) - 1) * sem
                else:
                    ci = 0
                mean_recalls.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if mean_recalls:
            label = f't={threshold} (baseline)' if threshold == 0.0 else f't={threshold}'
            ax.plot(epochs[:len(mean_recalls)], mean_recalls, color=colors[i],
                   linewidth=2, label=label)
            ax.fill_between(epochs[:len(mean_recalls)], ci_lower, ci_upper,
                           color=colors[i], alpha=0.15)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Recall@1', fontsize=12)
    ax.set_title(f'Learning Curves by Quality Threshold (gallery_size={gallery_size})\n'
                 f'Mean ± 95% CI across 8 seeds', fontsize=14)
    ax.legend(title='Threshold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_main_figure(metrics: dict, output_dir: str):
    """Create combined 3-panel figure."""
    fig = plt.figure(figsize=(18, 6))

    # Get unique values
    thresholds = sorted(set(k[0] for k in metrics.keys()))
    gallery_sizes = sorted(set(k[1] for k in metrics.keys()))

    # Viridis color map
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))
    color_map = {t: colors[i] for i, t in enumerate(thresholds)}

    # Panel A: Line plot
    ax1 = fig.add_subplot(131)

    for threshold in thresholds:
        x_vals, y_vals, ci_lower, ci_upper = [], [], [], []
        for gsize in gallery_sizes:
            key = (threshold, gsize)
            if key in metrics:
                data = metrics[key]
                mean = data['mean_recall_at_1']
                values = data['recall_at_1_values']
                ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                x_vals.append(gsize)
                y_vals.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if x_vals:
            label = f't={threshold}' if threshold > 0 else f't={threshold} (baseline)'
            ax1.plot(x_vals, y_vals, color=color_map[threshold], linewidth=2, marker='o', label=label)
            ax1.fill_between(x_vals, ci_lower, ci_upper, color=color_map[threshold], alpha=0.2)

    ax1.set_xlabel('Gallery Size')
    ax1.set_ylabel('Recall@1')
    ax1.set_title('A) Recall@1 vs Gallery Size\n(Best Epoch)')
    ax1.set_xticks(gallery_sizes)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Heatmap
    ax2 = fig.add_subplot(132)
    recall_matrix = np.zeros((len(thresholds), len(gallery_sizes)))
    for i, threshold in enumerate(thresholds):
        for j, gsize in enumerate(gallery_sizes):
            key = (threshold, gsize)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=gallery_sizes, yticklabels=[f'{t:.1f}' for t in thresholds], ax=ax2)
    ax2.set_xlabel('Gallery Size')
    ax2.set_ylabel('Threshold')
    ax2.set_title('B) Parameter Space Heatmap\n(Best Epoch R@1)')

    # Panel C: Improvement bars
    ax3 = fig.add_subplot(133)

    non_baseline_thresholds = [t for t in thresholds if t > 0]
    x = np.arange(len(gallery_sizes))
    width = 0.8 / max(len(non_baseline_thresholds), 1)

    improvement_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(non_baseline_thresholds)))

    for i, threshold in enumerate(non_baseline_thresholds):
        offset = (i - len(non_baseline_thresholds) / 2 + 0.5) * width
        values = []
        for gsize in gallery_sizes:
            baseline_key = (0.0, gsize)
            current_key = (threshold, gsize)
            if baseline_key in metrics and current_key in metrics:
                baseline = metrics[baseline_key]['mean_recall_at_1']
                current = metrics[current_key]['mean_recall_at_1']
                pct = ((current - baseline) / baseline) * 100 if baseline > 0 else 0
            else:
                pct = 0
            values.append(pct)
        ax3.bar(x + offset, values, width, label=f't={threshold}', color=improvement_colors[i])

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(gallery_sizes)
    ax3.set_xlabel('Gallery Size')
    ax3.set_ylabel('% Improvement')
    ax3.set_title('C) Improvement over Baseline\n(t=0.0)')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hygiene_sweep_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved main figure: {output_path}")


def print_summary_table(metrics: dict):
    """Print summary statistics with best epoch information."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean Recall@1 by Configuration (Best Epoch Selection)")
    print("=" * 70)
    print(f"{'Thresh':<8} {'Gallery':<10} {'Best Ep':<10} {'R@1 Mean':<12} {'R@1 Std':<12} {'N Seeds':<8}")
    print("-" * 70)

    for (threshold, gallery_size), data in sorted(metrics.items()):
        best_ep = data.get('best_epoch', 50)
        print(f"{threshold:<8.2f} {gallery_size:<10} {best_ep:<10} {data['mean_recall_at_1']:.4f}{'':>6} "
              f"{data['std_recall_at_1']:.4f}{'':>6} {data['n_seeds']:<8}")

    # Find best configuration
    if metrics:
        best_key = max(metrics.keys(), key=lambda k: metrics[k]['mean_recall_at_1'])
        best_data = metrics[best_key]
        print("-" * 70)
        print(f"Best: threshold={best_key[0]}, gallery_size={best_key[1]}, epoch={best_data.get('best_epoch', 50)} → "
              f"R@1={best_data['mean_recall_at_1']:.4f}")

    # Show improvement over baseline
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER BASELINE (threshold=0.0)")
    print("=" * 70)

    thresholds = sorted(set(k[0] for k in metrics.keys()))
    gallery_sizes = sorted(set(k[1] for k in metrics.keys()))

    print(f"{'Gallery':<10}", end="")
    for t in thresholds:
        print(f"t={t:.1f}{'':>6}", end="")
    print()
    print("-" * 70)

    for gsize in gallery_sizes:
        print(f"{gsize:<10}", end="")
        baseline = metrics.get((0.0, gsize), {}).get('mean_recall_at_1', 0)
        for t in thresholds:
            current = metrics.get((t, gsize), {}).get('mean_recall_at_1', 0)
            if baseline > 0:
                pct = ((current - baseline) / baseline) * 100
                print(f"{pct:+.1f}%{'':>4}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description='Plot hygiene sweep results')
    parser.add_argument('--results_dir', type=str,
                        default='reid_hygiene_filter/results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_hygiene_filter/figures',
                        help='Directory to save figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Gallery Hygiene Sweep Results Analysis")
    print("=" * 60)

    # Load results
    results = load_hygiene_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Extract metrics from epoch history
    metrics = extract_metrics(results)

    if not metrics:
        print("No valid metrics extracted. Exiting.")
        return

    # Print summary
    print_summary_table(metrics)

    # Generate individual plots
    print("\nGenerating figures...")

    plot_recall_vs_gallery_size(
        metrics,
        os.path.join(args.output_dir, 'panel_a_recall_vs_gallery_size.png')
    )

    plot_parameter_heatmap(
        metrics,
        os.path.join(args.output_dir, 'panel_b_heatmap.png')
    )

    plot_improvement_over_baseline(
        metrics,
        os.path.join(args.output_dir, 'panel_c_improvement.png')
    )

    # Generate combined main figure
    create_main_figure(metrics, args.output_dir)

    # Learning curves for each gallery size
    for gsize in sorted(results.get_unique('gallery_size')):
        try:
            plot_learning_curves(
                results,
                os.path.join(args.output_dir, f'learning_curves_gallery={gsize}.png'),
                gallery_size=gsize
            )
        except Exception as e:
            print(f"Could not generate learning curves for gallery_size={gsize}: {e}")

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

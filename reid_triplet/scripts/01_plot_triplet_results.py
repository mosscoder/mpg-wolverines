#!/usr/bin/env python3
"""
Script 01: Plot Quality-Weighted Triplet Loss Results
Generates publication-quality figures from triplet sweep experiment.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) Line plot: Recall@1 vs samples_per_class, lines by min_weight value
B) Heatmap: min_weight × samples → Recall@1 parameter space
C) Grouped bars: Recall@1 by query×gallery quality combo, all min_weight values
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


def load_triplet_results(results_dir: str) -> ResultsCollection:
    """Load triplet sweep results from JSON files."""
    pattern = os.path.join(results_dir, "min_weight=*_samples=*_seed=*.json")
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
                data['min_weight'] = data['config']['min_weight']
                data['samples_per_class'] = data['config']['samples_per_class']
                data['seed'] = data['config']['seed']
                results.append(data)
        except Exception as e:
            failed.append((json_file, str(e)))

    if failed:
        print(f"Failed to load {len(failed)} files")

    print(f"Successfully loaded {len(results)} results")
    return ResultsCollection(results)


def find_best_epoch_for_combo(all_histories: List[List[dict]], combo: str) -> Tuple[int, float]:
    """
    Find the epoch that maximizes mean recall for a specific query/gallery combo.

    Args:
        all_histories: List of epoch_history lists (one per seed)
        combo: Quality combo name (e.g., 'HQ_HG', 'LQ_LG')

    Returns:
        Tuple of (best_epoch, best_mean_recall)
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0

    # Get all epochs that have query_gallery_matrix
    epochs_with_matrix = []
    for h in all_histories[0]:
        if 'query_gallery_matrix' in h and combo in h.get('query_gallery_matrix', {}):
            epochs_with_matrix.append(h['epoch'])

    if not epochs_with_matrix:
        # Fallback to overall recall if no matrix data
        return find_best_epoch_for_overall_recall(all_histories)

    best_epoch = None
    best_mean = -1

    for epoch in epochs_with_matrix:
        recalls = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch:
                    qg_matrix = h.get('query_gallery_matrix', {})
                    if combo in qg_matrix:
                        recalls.append(qg_matrix[combo]['recall_at_1'])
                    break

        if recalls:
            mean_recall = np.mean(recalls)
            if mean_recall > best_mean:
                best_mean = mean_recall
                best_epoch = epoch

    return best_epoch or 50, best_mean


def find_best_epoch_for_overall_recall(all_histories: List[List[dict]]) -> Tuple[int, float]:
    """Find epoch with best mean val_recall_at_1 across seeds."""
    if not all_histories or not all_histories[0]:
        return 50, 0.0

    eval_epochs = [h['epoch'] for h in all_histories[0] if 'val_recall_at_1' in h]

    best_epoch = None
    best_mean = -1

    for epoch in eval_epochs:
        recalls = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch and 'val_recall_at_1' in h:
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

    For each (min_weight, samples) config, finds the epoch with best mean Recall@1
    across all 8 seeds, then reports that epoch's metrics.
    """
    groups = results.group_by('min_weight', 'samples_per_class')

    metrics = {}
    for (min_weight, samples), group in groups.items():
        all_histories = [r.get('epoch_history', []) for r in group]

        if not all_histories or not all_histories[0]:
            continue

        # Find best epoch for overall recall
        best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

        # Extract recall values at best epoch
        recall_values = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == best_epoch and 'val_recall_at_1' in h:
                    recall_values.append(h['val_recall_at_1'])
                    break

        if recall_values:
            metrics[(min_weight, samples)] = {
                'recall_at_1_values': recall_values,
                'mean_recall_at_1': np.mean(recall_values),
                'std_recall_at_1': np.std(recall_values, ddof=1) if len(recall_values) > 1 else 0,
                'n_seeds': len(recall_values),
                'best_epoch': best_epoch
            }

    return metrics


def extract_metrics_per_combo(results: ResultsCollection) -> dict:
    """
    Extract metrics with per-combo best epoch selection.

    For each (min_weight, samples) config and each quality combo (HQ_HG, etc.),
    finds the best epoch for that specific combo and extracts per-seed values.

    Returns dict keyed by (min_weight, samples) with:
        - 'recall_at_1': overall best epoch metrics
        - 'per_combo': {combo: {'best_epoch': int, 'mean': float, 'std': float, 'values': list}}
    """
    groups = results.group_by('min_weight', 'samples_per_class')
    combo_names = ['HQ_HG', 'HQ_LG', 'LQ_HG', 'LQ_LG']

    metrics = {}
    for (min_weight, samples), group in groups.items():
        all_histories = [r.get('epoch_history', []) for r in group]

        if not all_histories or not all_histories[0]:
            continue

        per_combo = {}
        for combo in combo_names:
            # Find best epoch for this combo (mean across seeds)
            best_epoch, best_mean = find_best_epoch_for_combo(all_histories, combo)

            # Extract per-seed values at best epoch
            values = []
            counts = []
            gallery_sizes = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch:
                        qg_matrix = h.get('query_gallery_matrix', {})
                        if combo in qg_matrix:
                            values.append(qg_matrix[combo]['recall_at_1'])
                            counts.append(qg_matrix[combo].get('count', 0))
                            gallery_sizes.append(qg_matrix[combo].get('gallery_size', 0))
                        break

            per_combo[combo] = {
                'best_epoch': best_epoch,
                'mean': np.mean(values) if values else 0,
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'values': values,
                'avg_count': np.mean(counts) if counts else 0,
                'avg_gallery_size': np.mean(gallery_sizes) if gallery_sizes else 0
            }

        metrics[(min_weight, samples)] = {'per_combo': per_combo}

    return metrics


def plot_recall_vs_samples(metrics: dict, output_path: str):
    """
    Panel A: Line plot of Recall@1 vs samples_per_class.
    Lines colored by min_weight value with 95% CI ribbons.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique values
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Color scheme: gray for min_weight=1.0 (baseline), blue gradient for others
    colors = {1.0: '#888888'}
    other_weights = [mw for mw in min_weights if mw < 1.0]
    if other_weights:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(other_weights)))
        for i, mw in enumerate(other_weights):
            colors[mw] = blues[i]

    legend_handles = []
    legend_labels = []

    for min_weight in min_weights:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []

        for sample_size in samples:
            key = (min_weight, sample_size)
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

                x_vals.append(sample_size)
                y_vals.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if x_vals:
            line, = ax.plot(x_vals, y_vals, color=colors[min_weight], linewidth=2.5,
                           marker='o', markersize=8)
            ax.fill_between(x_vals, ci_lower, ci_upper, color=colors[min_weight], alpha=0.2)

            legend_handles.append(line)
            if min_weight == 1.0:
                legend_labels.append(f'mw = {min_weight} (baseline)')
            else:
                legend_labels.append(f'mw = {min_weight}')

    ax.set_xlabel('Samples per Individual', fontsize=14)
    ax.set_ylabel('Recall@1', fontsize=14)
    ax.set_title('Confidence-Weighted Triplet Loss: Recall@1 vs Training Samples\n'
                 '(Best epoch by cross-seed validation, 95% CI from 8 seeds)', fontsize=16, pad=20)
    ax.set_xticks(samples)
    ax.grid(True, alpha=0.3, axis='y')

    legend = ax.legend(legend_handles, legend_labels, loc='lower right')
    legend.set_title('Min Weight', prop={'weight': 'bold'})

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_parameter_heatmap(metrics: dict, output_path: str):
    """
    Panel B: Heatmap of min_weight × samples → Recall@1.
    Shows parameter space overview.
    """
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Create matrix
    recall_matrix = np.zeros((len(min_weights), len(samples)))

    for i, min_weight in enumerate(min_weights):
        for j, sample in enumerate(samples):
            key = (min_weight, sample)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=min_weights,
                cbar_kws={'label': 'Recall@1'}, ax=ax)

    ax.set_xlabel('Samples per Individual', fontsize=12)
    ax.set_ylabel('Min Weight', fontsize=12)
    ax.set_title('Parameter Space: Mean Recall@1 at Best Epoch', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_query_gallery_matrix(combo_metrics: dict, output_path: str, samples_filter: int = 64):
    """
    Panel C: Grouped bars showing Recall@1 by query×gallery quality combinations.
    Each combo uses its own best epoch for selection.

    Args:
        combo_metrics: Dict from extract_metrics_per_combo()
        output_path: Where to save the figure
        samples_filter: Which samples_per_class to use (default 64)
    """
    min_weights = sorted(set(k[0] for k in combo_metrics.keys()), reverse=True)
    combo_names = ['HQ_HG', 'HQ_LG', 'LQ_HG', 'LQ_LG']
    combo_display = ['HQ→HG', 'HQ→LG', 'LQ→HG', 'LQ→LG']

    if len(min_weights) == 0:
        print("No min_weight values found, skipping query-gallery matrix plot")
        return

    # Color scheme: gray for baseline, blues for others
    colors = {1.0: '#888888'}
    other_weights = [mw for mw in min_weights if mw < 1.0]
    if other_weights:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(other_weights)))
        for i, mw in enumerate(other_weights):
            colors[mw] = blues[i]

    # Check if we have data for the specified samples_filter
    has_data = any((mw, samples_filter) in combo_metrics for mw in min_weights)
    if not has_data:
        print(f"No data found for samples={samples_filter}")
        return

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(combo_names))
    width = 0.8 / len(min_weights)

    # Track best epochs used for subtitle
    best_epochs_info = {}

    for i, mw in enumerate(min_weights):
        key = (mw, samples_filter)
        if key not in combo_metrics:
            continue

        per_combo = combo_metrics[key]['per_combo']
        offset = (i - len(min_weights) / 2 + 0.5) * width

        means = [per_combo[c]['mean'] for c in combo_names]
        stds = [per_combo[c]['std'] for c in combo_names]

        label = f'mw={mw} (baseline)' if mw == 1.0 else f'mw={mw}'
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=colors[mw], capsize=3)

        # Track best epochs
        for c in combo_names:
            if c not in best_epochs_info:
                best_epochs_info[c] = []
            best_epochs_info[c].append(per_combo[c]['best_epoch'])

    ax.set_xticks(x)
    ax.set_xticklabels(combo_display)
    ax.set_ylabel('Recall@1')
    ax.set_xlabel('Query Quality → Gallery Quality')
    ax.set_title(f'Performance by Query×Gallery Quality\n(samples={samples_filter}, per-combo best epoch)')
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_learning_curves(results: ResultsCollection, output_path: str, configs=None):
    """
    Optional: Learning curves showing training progress.
    """
    if configs is None:
        configs = [
            {'min_weight': 1.0, 'samples_per_class': 16, 'seed': 0},
            {'min_weight': 0.1, 'samples_per_class': 16, 'seed': 0}
        ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#888888', '#2980b9', '#27ae60', '#e74c3c']

    for i, cfg in enumerate(configs):
        filtered = results.filter(**cfg)
        if len(filtered) == 0:
            continue

        result = filtered[0]
        history = result.get('epoch_history', [])

        epochs = []
        recall_values = []
        for h in history:
            if 'val_recall_at_1' in h:
                epochs.append(h['epoch'])
                recall_values.append(h['val_recall_at_1'])

        if epochs:
            label = f"mw={cfg['min_weight']}, n={cfg['samples_per_class']}"
            ax.plot(epochs, recall_values, color=colors[i % len(colors)],
                   linewidth=2, marker='o', markersize=4, label=label)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Recall@1', fontsize=12)
    ax.set_title('Learning Curves: Validation Recall@1 over Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_main_figure(metrics: dict, combo_metrics: dict, output_dir: str, samples_filter: int = 64):
    """Create combined 3-panel figure."""
    fig = plt.figure(figsize=(18, 6))

    # Get unique values
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Panel A: Line plot
    ax1 = fig.add_subplot(131)

    colors = {1.0: '#888888'}
    other_weights = [mw for mw in min_weights if mw < 1.0]
    if other_weights:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(other_weights)))
        for i, mw in enumerate(other_weights):
            colors[mw] = blues[i]

    for min_weight in min_weights:
        x_vals, y_vals, ci_lower, ci_upper = [], [], [], []
        for sample in samples:
            key = (min_weight, sample)
            if key in metrics:
                data = metrics[key]
                mean = data['mean_recall_at_1']
                values = data['recall_at_1_values']
                ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                x_vals.append(sample)
                y_vals.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if x_vals:
            label = f'mw={min_weight}' if min_weight < 1.0 else f'mw={min_weight} (baseline)'
            ax1.plot(x_vals, y_vals, color=colors[min_weight], linewidth=2, marker='o', label=label)
            ax1.fill_between(x_vals, ci_lower, ci_upper, color=colors[min_weight], alpha=0.2)

    ax1.set_xlabel('Samples per Individual')
    ax1.set_ylabel('Recall@1')
    ax1.set_title('A) Recall@1 vs Training Samples\n(Best Epoch)')
    ax1.set_xticks(samples)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Heatmap
    ax2 = fig.add_subplot(132)
    recall_matrix = np.zeros((len(min_weights), len(samples)))
    for i, min_weight in enumerate(min_weights):
        for j, sample in enumerate(samples):
            key = (min_weight, sample)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=min_weights, ax=ax2)
    ax2.set_xlabel('Samples per Individual')
    ax2.set_ylabel('Min Weight')
    ax2.set_title('B) Parameter Space Heatmap\n(Best Epoch R@1)')

    # Panel C: Query×Gallery Quality Matrix with per-combo best epochs
    ax3 = fig.add_subplot(133)

    combo_names = ['HQ_HG', 'HQ_LG', 'LQ_HG', 'LQ_LG']
    combo_display = ['HQ→HG', 'HQ→LG', 'LQ→HG', 'LQ→LG']

    x = np.arange(len(combo_names))
    width = 0.8 / len(min_weights)

    has_panel_c_data = any((mw, samples_filter) in combo_metrics for mw in min_weights)

    for i, mw in enumerate(min_weights):
        key = (mw, samples_filter)
        if key not in combo_metrics:
            continue

        per_combo = combo_metrics[key]['per_combo']
        offset = (i - len(min_weights) / 2 + 0.5) * width
        means = [per_combo[c]['mean'] for c in combo_names]
        stds = [per_combo[c]['std'] for c in combo_names]

        label = f'mw={mw} (baseline)' if mw == 1.0 else f'mw={mw}'
        ax3.bar(x + offset, means, width, yerr=stds, label=label, color=colors[mw], capsize=3)

    ax3.set_xticks(x)
    ax3.set_xticklabels(combo_display)
    ax3.set_ylabel('Recall@1')
    title_suffix = '' if has_panel_c_data else '\n(no data)'
    ax3.set_title(f'C) Query×Gallery Quality\n(samples={samples_filter}, per-combo best epoch){title_suffix}')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.set_ylim(bottom=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'triplet_quality_weighting_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved main figure: {output_path}")


def print_summary_table(metrics: dict, combo_metrics: dict):
    """Print summary statistics with best epoch information."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean Recall@1 by Configuration (Best Epoch Selection)")
    print("=" * 70)
    print(f"{'MinWt':<8} {'Samples':<10} {'Best Ep':<10} {'R@1 Mean':<12} {'R@1 Std':<12} {'N Seeds':<8}")
    print("-" * 70)

    for (min_weight, samples), data in sorted(metrics.items(), reverse=True):
        best_ep = data.get('best_epoch', 50)
        print(f"{min_weight:<8} {samples:<10} {best_ep:<10} {data['mean_recall_at_1']:.4f}{'':>6} "
              f"{data['std_recall_at_1']:.4f}{'':>6} {data['n_seeds']:<8}")

    # Find best configuration
    if metrics:
        best_key = max(metrics.keys(), key=lambda k: metrics[k]['mean_recall_at_1'])
        best_data = metrics[best_key]
        print("-" * 70)
        print(f"Best: min_weight={best_key[0]}, samples={best_key[1]}, epoch={best_data.get('best_epoch', 50)} → "
              f"R@1={best_data['mean_recall_at_1']:.4f}")

    # Show per-combo best epochs for a sample configuration
    print("\n" + "=" * 70)
    print("PER-COMBO BEST EPOCHS (showing samples=64)")
    print("=" * 70)

    combo_names = ['HQ_HG', 'HQ_LG', 'LQ_HG', 'LQ_LG']
    print(f"{'MinWt':<8} {'HQ_HG':<12} {'HQ_LG':<12} {'LQ_HG':<12} {'LQ_LG':<12}")
    print("-" * 70)

    for mw in sorted(set(k[0] for k in combo_metrics.keys()), reverse=True):
        key = (mw, 64)
        if key in combo_metrics:
            per_combo = combo_metrics[key]['per_combo']
            epochs = [str(per_combo[c]['best_epoch']) for c in combo_names]
            recalls = [f"{per_combo[c]['mean']:.3f}" for c in combo_names]
            print(f"{mw:<8} " + " ".join(f"ep{e}:{r:<5}" for e, r in zip(epochs, recalls)))


def main():
    parser = argparse.ArgumentParser(description='Plot triplet sweep results')
    parser.add_argument('--results_dir', type=str,
                        default='reid_triplet/results/triplet_sweep',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_triplet/figures',
                        help='Directory to save figures')
    parser.add_argument('--samples', type=int, default=64,
                        help='samples_per_class to use for quality matrix plot (default: 64)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Quality-Weighted Triplet Loss Results Analysis")
    print("=" * 60)

    # Load results
    results = load_triplet_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Extract metrics from epoch history
    metrics = extract_metrics(results)
    combo_metrics = extract_metrics_per_combo(results)

    if not metrics:
        print("No valid metrics extracted. Exiting.")
        return

    # Print summary
    print_summary_table(metrics, combo_metrics)

    # Generate individual plots
    print("\nGenerating figures...")

    plot_recall_vs_samples(
        metrics,
        os.path.join(args.output_dir, 'panel_a_recall_vs_samples.png')
    )

    plot_parameter_heatmap(
        metrics,
        os.path.join(args.output_dir, 'panel_b_heatmap.png')
    )

    plot_query_gallery_matrix(
        combo_metrics,
        os.path.join(args.output_dir, 'panel_c_quality_bins.png'),
        samples_filter=args.samples
    )

    # Generate query×gallery plots for all sample sizes
    qg_plot_dir = os.path.join(args.output_dir, 'query_gallery_plots')
    os.makedirs(qg_plot_dir, exist_ok=True)
    all_samples = sorted(set(k[1] for k in combo_metrics.keys()))
    print(f"\nGenerating query×gallery plots for sample sizes: {all_samples}")
    for sample_size in all_samples:
        plot_query_gallery_matrix(
            combo_metrics,
            os.path.join(qg_plot_dir, f'query_gallery_samples={sample_size}.png'),
            samples_filter=sample_size
        )

    # Generate combined main figure
    create_main_figure(metrics, combo_metrics, args.output_dir, samples_filter=args.samples)

    # Learning curves (optional)
    try:
        plot_learning_curves(
            results,
            os.path.join(args.output_dir, 'learning_curves.png')
        )
    except Exception as e:
        print(f"Could not generate learning curves: {e}")

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

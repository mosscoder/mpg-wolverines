#!/usr/bin/env python3
"""
Script 01: Plot Quality-Weighted Triplet Loss Results
Generates publication-quality figures from triplet sweep experiment.

Figures:
A) Line plot: Recall@1 vs samples_per_class, lines by alpha value
B) Heatmap: alpha × samples → Recall@1 parameter space
C) Grouped bars: Recall@1 by validation quality bin, α=0 vs best α>0
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

sys.path.append('.')
from utils.results import ResultsCollection, load_json_results


def load_triplet_results(results_dir: str) -> ResultsCollection:
    """Load triplet sweep results from JSON files."""
    pattern = os.path.join(results_dir, "alpha=*_samples=*_seed=*.json")
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
                data['alpha'] = data['config']['alpha']
                data['samples_per_class'] = data['config']['samples_per_class']
                data['seed'] = data['config']['seed']
                # Note: recall_at_1 will be computed from best epoch
                # in extract_metrics(), not from final_metrics
                results.append(data)
        except Exception as e:
            failed.append((json_file, str(e)))

    if failed:
        print(f"Failed to load {len(failed)} files")

    print(f"Successfully loaded {len(results)} results")
    return ResultsCollection(results)


def extract_metrics(results: ResultsCollection) -> dict:
    """
    Extract aggregated metrics for plotting using best-epoch selection.

    For each (alpha, samples) config, finds the epoch with best mean Recall@1
    across all 8 seeds, then reports that epoch's metrics instead of epoch 50.
    """
    # Group by (alpha, samples_per_class)
    groups = results.group_by('alpha', 'samples_per_class')

    metrics = {}
    for (alpha, samples), group in groups.items():
        # Get epoch history from all seeds
        all_histories = [r.get('epoch_history', []) for r in group]

        if not all_histories or not all_histories[0]:
            # Fallback to final_metrics if no epoch history
            recall_values = [r.get('final_metrics', {}).get('recall_at_1', 0) for r in group]
            mAP_values = [r.get('final_metrics', {}).get('mean_avg_precision', 0) for r in group]
            best_epoch = 50
        else:
            # Find epochs where we have val_recall_at_1 (every 5th epoch)
            eval_epochs = [h['epoch'] for h in all_histories[0] if 'val_recall_at_1' in h]

            # For each eval epoch, compute mean recall across seeds
            best_epoch = None
            best_mean_recall = -1

            for epoch in eval_epochs:
                epoch_recalls = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == epoch and 'val_recall_at_1' in h:
                            epoch_recalls.append(h['val_recall_at_1'])
                            break

                if epoch_recalls:
                    mean_recall = np.mean(epoch_recalls)
                    if mean_recall > best_mean_recall:
                        best_mean_recall = mean_recall
                        best_epoch = epoch

            # Extract metrics at best epoch for each seed
            recall_values = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch and 'val_recall_at_1' in h:
                        recall_values.append(h['val_recall_at_1'])
                        break

            # mAP not tracked per epoch, use final
            mAP_values = [r.get('final_metrics', {}).get('mean_avg_precision', 0) for r in group]

        if recall_values:
            metrics[(alpha, samples)] = {
                'recall_at_1_values': recall_values,
                'mean_recall_at_1': np.mean(recall_values),
                'std_recall_at_1': np.std(recall_values, ddof=1) if len(recall_values) > 1 else 0,
                'mean_mAP': np.mean(mAP_values) if mAP_values else 0,
                'n_seeds': len(recall_values),
                'best_epoch': best_epoch
            }

            # Extract bin-specific metrics from final_metrics
            # (bin metrics not tracked per-epoch, so use final)
            bin_metrics = defaultdict(list)
            for r in group:
                by_bin = r.get('final_metrics', {}).get('by_quality_bin', {})
                for bin_name, bin_data in by_bin.items():
                    if bin_data.get('count', 0) > 0:
                        bin_metrics[bin_name].append(bin_data.get('recall_at_1', 0))

            metrics[(alpha, samples)]['bin_metrics'] = {
                bin_name: {
                    'mean': np.mean(vals),
                    'std': np.std(vals, ddof=1) if len(vals) > 1 else 0,
                    'values': vals
                }
                for bin_name, vals in bin_metrics.items()
            }

    return metrics


def plot_recall_vs_samples(metrics: dict, output_path: str):
    """
    Panel A: Line plot of Recall@1 vs samples_per_class.
    Lines colored by alpha value with 95% CI ribbons.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique values
    alphas = sorted(set(k[0] for k in metrics.keys()))
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Color scheme: gray for alpha=0, blue gradient for others
    colors = {0: '#888888'}
    blue_alphas = [a for a in alphas if a > 0]
    if blue_alphas:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(blue_alphas)))
        for i, alpha in enumerate(blue_alphas):
            colors[alpha] = blues[i]

    legend_handles = []
    legend_labels = []

    for alpha in alphas:
        x_vals = []
        y_vals = []
        ci_lower = []
        ci_upper = []

        for sample_size in samples:
            key = (alpha, sample_size)
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
            line, = ax.plot(x_vals, y_vals, color=colors[alpha], linewidth=2.5,
                           marker='o', markersize=8)
            ax.fill_between(x_vals, ci_lower, ci_upper, color=colors[alpha], alpha=0.2)

            legend_handles.append(line)
            if alpha == 0:
                legend_labels.append(f'α = {alpha} (baseline)')
            else:
                legend_labels.append(f'α = {alpha}')

    ax.set_xlabel('Samples per Individual', fontsize=14)
    ax.set_ylabel('Recall@1', fontsize=14)
    ax.set_title('Quality-Weighted Triplet Loss: Recall@1 vs Training Samples\n'
                 '(Best epoch by cross-seed validation, 95% CI from 8 seeds)', fontsize=16, pad=20)
    ax.set_xticks(samples)
    ax.grid(True, alpha=0.3, axis='y')

    legend = ax.legend(legend_handles, legend_labels, loc='lower right')
    legend.set_title('Quality weight α', prop={'weight': 'bold'})

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_parameter_heatmap(metrics: dict, output_path: str):
    """
    Panel B: Heatmap of alpha × samples → Recall@1.
    Shows parameter space overview.
    """
    alphas = sorted(set(k[0] for k in metrics.keys()))
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Create matrix
    recall_matrix = np.zeros((len(alphas), len(samples)))

    for i, alpha in enumerate(alphas):
        for j, sample in enumerate(samples):
            key = (alpha, sample)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=alphas,
                cbar_kws={'label': 'Recall@1'}, ax=ax)

    ax.set_xlabel('Samples per Individual', fontsize=12)
    ax.set_ylabel('Quality Weight α', fontsize=12)
    ax.set_title('Parameter Space: Mean Recall@1 at Best Epoch', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_quality_bin_comparison(metrics: dict, output_path: str):
    """
    Panel C: Grouped bars showing Recall@1 by validation quality bin.
    Compares α=0 (baseline) vs best α>0.
    """
    # Find best non-zero alpha across all sample sizes
    alphas = sorted(set(k[0] for k in metrics.keys()))
    samples = sorted(set(k[1] for k in metrics.keys()))
    non_zero_alphas = [a for a in alphas if a > 0]

    if not non_zero_alphas:
        print("No non-zero alpha values found, skipping bin comparison plot")
        return

    # Find best alpha by averaging across sample sizes
    alpha_means = {}
    for alpha in non_zero_alphas:
        recalls = []
        for sample in samples:
            key = (alpha, sample)
            if key in metrics:
                recalls.append(metrics[key]['mean_recall_at_1'])
        if recalls:
            alpha_means[alpha] = np.mean(recalls)

    best_alpha = max(alpha_means, key=alpha_means.get) if alpha_means else non_zero_alphas[0]
    print(f"Best non-zero alpha: {best_alpha}")

    # Quality bins
    bin_names = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']
    bin_display = ['High\n(0.75-1.0)', 'Med-High\n(0.5-0.75)',
                   'Med-Low\n(0.25-0.5)', 'Low\n(0-0.25)']

    # Aggregate across sample sizes for cleaner comparison
    baseline_bins = defaultdict(list)
    best_bins = defaultdict(list)

    for sample in samples:
        # Baseline (alpha=0)
        key_base = (0, sample)
        if key_base in metrics:
            for bin_name in bin_names:
                bin_data = metrics[key_base].get('bin_metrics', {}).get(bin_name, {})
                if 'mean' in bin_data:
                    baseline_bins[bin_name].append(bin_data['mean'])

        # Best alpha
        key_best = (best_alpha, sample)
        if key_best in metrics:
            for bin_name in bin_names:
                bin_data = metrics[key_best].get('bin_metrics', {}).get(bin_name, {})
                if 'mean' in bin_data:
                    best_bins[bin_name].append(bin_data['mean'])

    # Compute means and stds
    baseline_means = [np.mean(baseline_bins.get(b, [0])) for b in bin_names]
    baseline_stds = [np.std(baseline_bins.get(b, [0]), ddof=1) if len(baseline_bins.get(b, [])) > 1 else 0 for b in bin_names]
    best_means = [np.mean(best_bins.get(b, [0])) for b in bin_names]
    best_stds = [np.std(best_bins.get(b, [0]), ddof=1) if len(best_bins.get(b, [])) > 1 else 0 for b in bin_names]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bin_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label=f'α = 0 (baseline)', color='#888888', capsize=5)
    bars2 = ax.bar(x + width/2, best_means, width, yerr=best_stds,
                   label=f'α = {best_alpha} (best)', color='#2980b9', capsize=5)

    ax.set_ylabel('Recall@1', fontsize=12)
    ax.set_xlabel('Validation Image Quality Bin', fontsize=12)
    ax.set_title(f'Performance by Query Quality: Baseline vs Quality-Weighted (α={best_alpha})\n'
                 f'(averaged across sample sizes)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_display)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_learning_curves(results: ResultsCollection, output_path: str, configs=None):
    """
    Optional: Learning curves showing training progress.
    """
    if configs is None:
        # Default: show curves for alpha=0 and alpha=4 at samples=16, seed=0
        configs = [
            {'alpha': 0, 'samples_per_class': 16, 'seed': 0},
            {'alpha': 4, 'samples_per_class': 16, 'seed': 0}
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
            label = f"α={cfg['alpha']}, n={cfg['samples_per_class']}"
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


def create_main_figure(metrics: dict, output_dir: str):
    """Create combined 3-panel figure."""
    fig = plt.figure(figsize=(18, 6))

    # Get unique values
    alphas = sorted(set(k[0] for k in metrics.keys()))
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Panel A: Line plot
    ax1 = fig.add_subplot(131)

    colors = {0: '#888888'}
    blue_alphas = [a for a in alphas if a > 0]
    if blue_alphas:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(blue_alphas)))
        for i, alpha in enumerate(blue_alphas):
            colors[alpha] = blues[i]

    for alpha in alphas:
        x_vals, y_vals, ci_lower, ci_upper = [], [], [], []
        for sample in samples:
            key = (alpha, sample)
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
            label = f'α={alpha}' if alpha > 0 else f'α={alpha} (baseline)'
            ax1.plot(x_vals, y_vals, color=colors[alpha], linewidth=2, marker='o', label=label)
            ax1.fill_between(x_vals, ci_lower, ci_upper, color=colors[alpha], alpha=0.2)

    ax1.set_xlabel('Samples per Individual')
    ax1.set_ylabel('Recall@1')
    ax1.set_title('A) Recall@1 vs Training Samples\n(Best Epoch)')
    ax1.set_xticks(samples)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Heatmap
    ax2 = fig.add_subplot(132)
    recall_matrix = np.zeros((len(alphas), len(samples)))
    for i, alpha in enumerate(alphas):
        for j, sample in enumerate(samples):
            key = (alpha, sample)
            if key in metrics:
                recall_matrix[i, j] = metrics[key]['mean_recall_at_1']

    sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=alphas, ax=ax2)
    ax2.set_xlabel('Samples per Individual')
    ax2.set_ylabel('Quality Weight α')
    ax2.set_title('B) Parameter Space Heatmap\n(Best Epoch R@1)')

    # Panel C: Quality bin comparison
    ax3 = fig.add_subplot(133)

    non_zero_alphas = [a for a in alphas if a > 0]
    if non_zero_alphas:
        alpha_means = {}
        for alpha in non_zero_alphas:
            recalls = [metrics[(alpha, s)]['mean_recall_at_1'] for s in samples if (alpha, s) in metrics]
            alpha_means[alpha] = np.mean(recalls) if recalls else 0
        best_alpha = max(alpha_means, key=alpha_means.get)

        bin_names = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']
        bin_display = ['High', 'Med-High', 'Med-Low', 'Low']

        baseline_means, best_means = [], []
        for bin_name in bin_names:
            base_vals = [metrics[(0, s)]['bin_metrics'].get(bin_name, {}).get('mean', 0)
                        for s in samples if (0, s) in metrics]
            best_vals = [metrics[(best_alpha, s)]['bin_metrics'].get(bin_name, {}).get('mean', 0)
                        for s in samples if (best_alpha, s) in metrics]
            baseline_means.append(np.mean(base_vals) if base_vals else 0)
            best_means.append(np.mean(best_vals) if best_vals else 0)

        x = np.arange(len(bin_names))
        width = 0.35
        ax3.bar(x - width/2, baseline_means, width, label='α=0', color='#888888')
        ax3.bar(x + width/2, best_means, width, label=f'α={best_alpha}', color='#2980b9')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bin_display)
        ax3.set_ylabel('Recall@1')
        ax3.set_title('C) Performance by Query Quality')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'triplet_quality_weighting_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved main figure: {output_path}")


def print_summary_table(metrics: dict):
    """Print summary statistics with best epoch information."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean Recall@1 by Configuration (Best Epoch Selection)")
    print("=" * 70)
    print(f"{'Alpha':<8} {'Samples':<10} {'Best Ep':<10} {'R@1 Mean':<12} {'R@1 Std':<12} {'N Seeds':<8}")
    print("-" * 70)

    for (alpha, samples), data in sorted(metrics.items()):
        best_ep = data.get('best_epoch', 50)
        print(f"{alpha:<8} {samples:<10} {best_ep:<10} {data['mean_recall_at_1']:.4f}{'':>6} "
              f"{data['std_recall_at_1']:.4f}{'':>6} {data['n_seeds']:<8}")

    # Find best configuration
    best_key = max(metrics.keys(), key=lambda k: metrics[k]['mean_recall_at_1'])
    best_data = metrics[best_key]
    print("-" * 70)
    print(f"Best: alpha={best_key[0]}, samples={best_key[1]}, epoch={best_data.get('best_epoch', 50)} → "
          f"R@1={best_data['mean_recall_at_1']:.4f}")

    # Show epoch distribution
    epochs_used = [data.get('best_epoch', 50) for data in metrics.values()]
    unique_epochs = sorted(set(epochs_used))
    print(f"\nBest epochs distribution: {dict((e, epochs_used.count(e)) for e in unique_epochs)}")


def main():
    parser = argparse.ArgumentParser(description='Plot triplet sweep results')
    parser.add_argument('--results_dir', type=str,
                        default='reid_triplet/results/triplet_sweep',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_triplet/figures',
                        help='Directory to save figures')

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

    # Extract metrics
    metrics = extract_metrics(results)

    if not metrics:
        print("No valid metrics extracted. Exiting.")
        return

    # Print summary
    print_summary_table(metrics)

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

    plot_quality_bin_comparison(
        metrics,
        os.path.join(args.output_dir, 'panel_c_quality_bins.png')
    )

    # Generate combined main figure
    create_main_figure(metrics, args.output_dir)

    # Learning curves (optional - requires specific configs to exist)
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

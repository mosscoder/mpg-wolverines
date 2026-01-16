#!/usr/bin/env python3
"""
Script 01: Plot Confidence-Weighted Classification Results
Generates publication-quality figures from classifier sweep experiment.

Figures:
A) Line plot: F1 vs samples_per_class, lines by min_weight value
B) Heatmap: min_weight × samples → F1 parameter space
C) Grouped bars: F1 by validation quality bin, baseline vs best min_weight
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


def load_classifier_results(results_dir: str) -> ResultsCollection:
    """Load classifier sweep results from JSON files."""
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


def extract_metrics(results: ResultsCollection) -> dict:
    """
    Extract aggregated metrics for plotting using best-epoch selection.

    For each (min_weight, samples) config, finds the epoch with best mean F1
    across all 8 seeds, then reports that epoch's metrics.
    """
    # Group by (min_weight, samples_per_class)
    groups = results.group_by('min_weight', 'samples_per_class')

    metrics = {}
    for (min_weight, samples), group in groups.items():
        # Get epoch history from all seeds
        all_histories = [r.get('epoch_history', []) for r in group]

        if not all_histories or not all_histories[0]:
            # Fallback to final_metrics if no epoch history
            f1_values = [r.get('final_metrics', {}).get('f1_macro', 0) for r in group]
            acc_values = [r.get('final_metrics', {}).get('accuracy', 0) for r in group]
            best_epoch = 50
        else:
            # Find epochs where we have val_f1_macro
            eval_epochs = [h['epoch'] for h in all_histories[0] if 'val_f1_macro' in h]

            # For each eval epoch, compute mean F1 across seeds
            best_epoch = None
            best_mean_f1 = -1

            for epoch in eval_epochs:
                epoch_f1s = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == epoch and 'val_f1_macro' in h:
                            epoch_f1s.append(h['val_f1_macro'])
                            break

                if epoch_f1s:
                    mean_f1 = np.mean(epoch_f1s)
                    if mean_f1 > best_mean_f1:
                        best_mean_f1 = mean_f1
                        best_epoch = epoch

            # Extract metrics at best epoch for each seed
            f1_values = []
            acc_values = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch and 'val_f1_macro' in h:
                        f1_values.append(h['val_f1_macro'])
                        acc_values.append(h.get('val_accuracy', 0))
                        break

        if f1_values:
            metrics[(min_weight, samples)] = {
                'f1_values': f1_values,
                'mean_f1': np.mean(f1_values),
                'std_f1': np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0,
                'acc_values': acc_values,
                'mean_accuracy': np.mean(acc_values) if acc_values else 0,
                'n_seeds': len(f1_values),
                'best_epoch': best_epoch
            }

            # Extract bin-specific metrics from final_metrics
            bin_metrics = defaultdict(list)
            for r in group:
                by_bin = r.get('final_metrics', {}).get('by_quality_bin', {})
                for bin_name, bin_data in by_bin.items():
                    if bin_data.get('count', 0) > 0:
                        bin_metrics[bin_name].append(bin_data.get('f1_macro', 0))

            metrics[(min_weight, samples)]['bin_metrics'] = {
                bin_name: {
                    'mean': np.mean(vals),
                    'std': np.std(vals, ddof=1) if len(vals) > 1 else 0,
                    'values': vals
                }
                for bin_name, vals in bin_metrics.items()
            }

    return metrics


def plot_f1_vs_samples(metrics: dict, output_path: str):
    """
    Panel A: Line plot of F1 vs samples_per_class.
    Lines colored by min_weight value with 95% CI ribbons.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique values
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)  # 1.0 first (baseline)
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
                mean = data['mean_f1']
                values = data['f1_values']

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
    ax.set_ylabel('F1 Macro', fontsize=14)
    ax.set_title('Confidence-Weighted Classification: F1 vs Training Samples\n'
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
    Panel B: Heatmap of min_weight × samples → F1.
    Shows parameter space overview.
    """
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)
    samples = sorted(set(k[1] for k in metrics.keys()))

    # Create matrix
    f1_matrix = np.zeros((len(min_weights), len(samples)))

    for i, min_weight in enumerate(min_weights):
        for j, sample in enumerate(samples):
            key = (min_weight, sample)
            if key in metrics:
                f1_matrix[i, j] = metrics[key]['mean_f1']

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=min_weights,
                cbar_kws={'label': 'F1 Macro'}, ax=ax)

    ax.set_xlabel('Samples per Individual', fontsize=12)
    ax.set_ylabel('Min Weight', fontsize=12)
    ax.set_title('Parameter Space: Mean F1 at Best Epoch', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_quality_bin_comparison(metrics: dict, output_path: str):
    """
    Panel C: Grouped bars showing F1 by validation quality bin.
    Compares baseline (min_weight=1.0) vs best min_weight<1.0.
    """
    # Find best non-baseline min_weight across all sample sizes
    min_weights = sorted(set(k[0] for k in metrics.keys()), reverse=True)
    samples = sorted(set(k[1] for k in metrics.keys()))
    non_baseline_weights = [mw for mw in min_weights if mw < 1.0]

    if not non_baseline_weights:
        print("No non-baseline min_weight values found, skipping bin comparison plot")
        return

    # Find best min_weight by averaging across sample sizes
    weight_means = {}
    for mw in non_baseline_weights:
        f1s = []
        for sample in samples:
            key = (mw, sample)
            if key in metrics:
                f1s.append(metrics[key]['mean_f1'])
        if f1s:
            weight_means[mw] = np.mean(f1s)

    best_weight = max(weight_means, key=weight_means.get) if weight_means else non_baseline_weights[0]
    print(f"Best non-baseline min_weight: {best_weight}")

    # Quality bins
    bin_names = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']
    bin_display = ['High\n(0.75-1.0)', 'Med-High\n(0.5-0.75)',
                   'Med-Low\n(0.25-0.5)', 'Low\n(0-0.25)']

    # Aggregate across sample sizes for cleaner comparison
    baseline_bins = defaultdict(list)
    best_bins = defaultdict(list)

    for sample in samples:
        # Baseline (min_weight=1.0)
        key_base = (1.0, sample)
        if key_base in metrics:
            for bin_name in bin_names:
                bin_data = metrics[key_base].get('bin_metrics', {}).get(bin_name, {})
                if 'mean' in bin_data:
                    baseline_bins[bin_name].append(bin_data['mean'])

        # Best min_weight
        key_best = (best_weight, sample)
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
                   label=f'mw = 1.0 (baseline)', color='#888888', capsize=5)
    bars2 = ax.bar(x + width/2, best_means, width, yerr=best_stds,
                   label=f'mw = {best_weight} (best)', color='#2980b9', capsize=5)

    ax.set_ylabel('F1 Macro', fontsize=12)
    ax.set_xlabel('Validation Image Quality Bin', fontsize=12)
    ax.set_title(f'Performance by Query Quality: Baseline vs Confidence-Weighted (mw={best_weight})\n'
                 f'(averaged across sample sizes)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_display)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_main_figure(metrics: dict, output_dir: str):
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
                mean = data['mean_f1']
                values = data['f1_values']
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
    ax1.set_ylabel('F1 Macro')
    ax1.set_title('A) F1 vs Training Samples\n(Best Epoch)')
    ax1.set_xticks(samples)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Heatmap
    ax2 = fig.add_subplot(132)
    f1_matrix = np.zeros((len(min_weights), len(samples)))
    for i, min_weight in enumerate(min_weights):
        for j, sample in enumerate(samples):
            key = (min_weight, sample)
            if key in metrics:
                f1_matrix[i, j] = metrics[key]['mean_f1']

    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=samples, yticklabels=min_weights, ax=ax2)
    ax2.set_xlabel('Samples per Individual')
    ax2.set_ylabel('Min Weight')
    ax2.set_title('B) Parameter Space Heatmap\n(Best Epoch F1)')

    # Panel C: Quality bin comparison
    ax3 = fig.add_subplot(133)

    non_baseline_weights = [mw for mw in min_weights if mw < 1.0]
    if non_baseline_weights:
        weight_means = {}
        for mw in non_baseline_weights:
            f1s = [metrics[(mw, s)]['mean_f1'] for s in samples if (mw, s) in metrics]
            weight_means[mw] = np.mean(f1s) if f1s else 0
        best_weight = max(weight_means, key=weight_means.get)

        bin_names = ['[0.75,1.0]', '[0.5,0.75)', '[0.25,0.5)', '[0,0.25)']
        bin_display = ['High', 'Med-High', 'Med-Low', 'Low']

        baseline_means, best_means = [], []
        for bin_name in bin_names:
            base_vals = [metrics[(1.0, s)]['bin_metrics'].get(bin_name, {}).get('mean', 0)
                        for s in samples if (1.0, s) in metrics]
            best_vals = [metrics[(best_weight, s)]['bin_metrics'].get(bin_name, {}).get('mean', 0)
                        for s in samples if (best_weight, s) in metrics]
            baseline_means.append(np.mean(base_vals) if base_vals else 0)
            best_means.append(np.mean(best_vals) if best_vals else 0)

        x = np.arange(len(bin_names))
        width = 0.35
        ax3.bar(x - width/2, baseline_means, width, label='mw=1.0', color='#888888')
        ax3.bar(x + width/2, best_means, width, label=f'mw={best_weight}', color='#2980b9')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bin_display)
        ax3.set_ylabel('F1 Macro')
        ax3.set_title('C) Performance by Query Quality')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'classifier_quality_weighting_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved main figure: {output_path}")


def print_summary_table(metrics: dict):
    """Print summary statistics with best epoch information."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean F1 by Configuration (Best Epoch Selection)")
    print("=" * 70)
    print(f"{'MinWt':<8} {'Samples':<10} {'Best Ep':<10} {'F1 Mean':<12} {'F1 Std':<12} {'N Seeds':<8}")
    print("-" * 70)

    for (min_weight, samples), data in sorted(metrics.items(), reverse=True):
        best_ep = data.get('best_epoch', 50)
        print(f"{min_weight:<8} {samples:<10} {best_ep:<10} {data['mean_f1']:.4f}{'':>6} "
              f"{data['std_f1']:.4f}{'':>6} {data['n_seeds']:<8}")

    # Find best configuration
    best_key = max(metrics.keys(), key=lambda k: metrics[k]['mean_f1'])
    best_data = metrics[best_key]
    print("-" * 70)
    print(f"Best: min_weight={best_key[0]}, samples={best_key[1]}, epoch={best_data.get('best_epoch', 50)} -> "
          f"F1={best_data['mean_f1']:.4f}")

    # Show epoch distribution
    epochs_used = [data.get('best_epoch', 50) for data in metrics.values()]
    unique_epochs = sorted(set(epochs_used))
    print(f"\nBest epochs distribution: {dict((e, epochs_used.count(e)) for e in unique_epochs)}")


def main():
    parser = argparse.ArgumentParser(description='Plot classifier sweep results')
    parser.add_argument('--results_dir', type=str,
                        default='reid_classifier/results/classifier_sweep',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_classifier/figures',
                        help='Directory to save figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Confidence-Weighted Classification Results Analysis")
    print("=" * 60)

    # Load results
    results = load_classifier_results(args.results_dir)

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

    plot_f1_vs_samples(
        metrics,
        os.path.join(args.output_dir, 'panel_a_f1_vs_samples.png')
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

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

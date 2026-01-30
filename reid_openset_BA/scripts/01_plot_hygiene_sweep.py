#!/usr/bin/env python3
"""
Script 01: Plot Open-Set Hygiene Sweep Results (Raw Cosine Similarity)

Generates publication-quality figures from open-set evaluation experiment
using DINOv3 backbone with ArcFace loss and raw cosine similarity scores.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) Line plot: Recall@1 for known wolverines - baseline vs optimal filtration
B) Bar chart: Optimal quality thresholds for maximizing R@1
C) Open-set performance (Balanced Accuracy) - baseline vs optimal
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple

sys.path.append('.')
from utils.results import ResultsCollection


# Settings for diagnostic functions (print_summary_table, plot_cosine_threshold).
# Note: The main 3-panel figure uses hardcoded 'recall' criterion.
DEFAULT_BEST_EPOCH_CRITERION = 'recall'
DEFAULT_QUALITY_THRESHOLD = 'q>=0.0'


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


def compute_epoch_metric(h: dict, criterion: str, query_thresh: str = "q>=0.0") -> float:
    """
    Compute a single metric value for one epoch record.

    Args:
        h: Single epoch history record
        criterion: One of 'harmonic_mean', 'ba', 'recall', 'recall_only', 'ba_only'
        query_thresh: Quality threshold key (e.g., "q>=0.0")

    Returns:
        Metric value, or None if not computable
    """
    recall = None
    ba = None

    if 'query_quality_metrics' in h and query_thresh in h['query_quality_metrics']:
        recall = h['query_quality_metrics'][query_thresh].get('recall_at_1')

    if 'open_set' in h:
        by_quality = h['open_set'].get('by_quality', {})
        if query_thresh in by_quality:
            ba = by_quality[query_thresh].get('balanced_accuracy')

    if criterion == 'recall' or criterion == 'recall_only':
        return recall
    elif criterion == 'ba' or criterion == 'ba_only':
        return ba
    elif criterion == 'harmonic_mean':
        if recall is not None and ba is not None and (recall + ba) > 0:
            return 2 * recall * ba / (recall + ba)
        return None
    elif criterion == 'arithmetic_mean':
        if recall is not None and ba is not None:
            return (recall + ba) / 2
        return None
    elif criterion == 'geometric_mean':
        if recall is not None and ba is not None and recall > 0 and ba > 0:
            return np.sqrt(recall * ba)
        return None
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def find_best_epoch(all_histories: List[List[dict]],
                    criterion: str = "harmonic_mean",
                    query_thresh: str = "q>=0.0") -> Tuple[int, float, dict]:
    """
    Find epoch with best mean metric across seeds.

    Args:
        all_histories: List of epoch histories (one per seed)
        criterion: Optimization criterion - one of:
            'harmonic_mean' (default): 2*R@1*BA / (R@1+BA) - balances both metrics
            'ba': Balanced accuracy only
            'recall': Recall@1 only
            'arithmetic_mean': (R@1 + BA) / 2
            'geometric_mean': sqrt(R@1 * BA)
        query_thresh: Quality threshold key for evaluation (e.g., "q>=0.0")

    Returns:
        Tuple of (best_epoch, best_mean_metric, details_dict)
        details_dict contains 'recall', 'ba', and 'criterion_value' at best epoch
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0, {}

    # Get epochs that have metrics
    eval_epochs = []
    for h in all_histories[0]:
        if 'query_quality_metrics' in h or 'open_set' in h:
            eval_epochs.append(h['epoch'])

    if not eval_epochs:
        return 50, 0.0, {}

    best_epoch = None
    best_mean = -1
    best_details = {}

    for epoch in eval_epochs:
        metric_values = []
        recall_values = []
        ba_values = []

        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch:
                    metric = compute_epoch_metric(h, criterion, query_thresh)
                    if metric is not None:
                        metric_values.append(metric)

                    # Also collect individual metrics for details
                    r = compute_epoch_metric(h, 'recall', query_thresh)
                    b = compute_epoch_metric(h, 'ba', query_thresh)
                    if r is not None:
                        recall_values.append(r)
                    if b is not None:
                        ba_values.append(b)
                    break

        if metric_values:
            mean_metric = np.mean(metric_values)
            if mean_metric > best_mean:
                best_mean = mean_metric
                best_epoch = epoch
                best_details = {
                    'recall': np.mean(recall_values) if recall_values else 0.0,
                    'ba': np.mean(ba_values) if ba_values else 0.0,
                    'criterion': criterion,
                    'criterion_value': mean_metric
                }

    return best_epoch or 50, best_mean, best_details


def find_optimal_epoch(all_histories: List[List[dict]],
                       query_thresh: str = None) -> Tuple[int, float, dict]:
    """
    Find optimal epoch using the configured default criterion.

    This is the primary function for determining the "best" epoch,
    using the criterion defined in DEFAULT_BEST_EPOCH_CRITERION.

    Args:
        all_histories: List of epoch histories (one per seed)
        query_thresh: Quality threshold key (defaults to DEFAULT_QUALITY_THRESHOLD)

    Returns:
        Tuple of (best_epoch, criterion_value, details_dict)
        details_dict contains 'recall', 'ba', 'criterion', and 'criterion_value'
    """
    if query_thresh is None:
        query_thresh = DEFAULT_QUALITY_THRESHOLD
    return find_best_epoch(all_histories, criterion=DEFAULT_BEST_EPOCH_CRITERION, query_thresh=query_thresh)


def create_combined_figure(results: ResultsCollection, output_dir: str):
    """
    Create 3-panel figure:
    A) R@1 by quality filtering strategy (baseline vs optimal)
    B) Optimal quality thresholds for R@1 (grouped bars: gallery + query)
    C) Balanced Accuracy (baseline vs optimal R@1-optimized model)

    Key principle: Model selection is based on R@1 only. Panel C shows
    open-set capability of those same models with the same quality filters.
    """
    fig = plt.figure(figsize=(18, 6))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    colors = {'baseline': '#1f77b4', 'optimal': '#2ca02c', 'optimal_ba': '#2ca02c'}

    # =========================================================================
    # Collect metrics for both strategies
    # =========================================================================
    strategies = {
        'baseline': {
            'label': 'Baseline (no filter)',
            'x': [], 'r1': [], 'r1_ci_lower': [], 'r1_ci_upper': [],
            'ba': [], 'ba_ci_lower': [], 'ba_ci_upper': []
        },
        'optimal': {
            'label': 'Optimal (R@1)',
            'x': [], 'r1': [], 'r1_ci_lower': [], 'r1_ci_upper': [],
            'ba': [], 'ba_ci_lower': [], 'ba_ci_upper': [],
            'best_gal': [], 'best_q': []
        },
        'optimal_ba': {
            'label': 'Optimal (BA)',
            'x': [], 'ba': [], 'ba_ci_lower': [], 'ba_ci_upper': [],
            'best_gal': [], 'best_q': []
        }
    }

    for gsize in gallery_sizes:
        # --- Baseline: gallery_threshold=0.0, eval q>=0.0 ---
        baseline_filtered = results.filter(threshold=0.0, gallery_size=gsize)
        if len(baseline_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in baseline_filtered]
            if all_histories and all_histories[0] and 'query_quality_metrics' in all_histories[0][0]:
                # Find best epoch by R@1 at q>=0.0
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='recall', query_thresh='q>=0.0')

                # Collect R@1 and balanced_accuracy at that epoch with q>=0.0
                r1_values = []
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            # R@1
                            if 'query_quality_metrics' in h and 'q>=0.0' in h['query_quality_metrics']:
                                r1_values.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                            # Balanced accuracy
                            if 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if 'q>=0.0' in by_quality:
                                    ba_values.append(by_quality['q>=0.0']['balanced_accuracy'])
                            break

                if r1_values:
                    mean_r1 = np.mean(r1_values)
                    ci_r1 = stats.t.ppf(0.975, len(r1_values) - 1) * stats.sem(r1_values) if len(r1_values) > 1 else 0
                    strategies['baseline']['x'].append(gsize)
                    strategies['baseline']['r1'].append(mean_r1)
                    strategies['baseline']['r1_ci_lower'].append(mean_r1 - ci_r1)
                    strategies['baseline']['r1_ci_upper'].append(mean_r1 + ci_r1)

                    if ba_values:
                        mean_ba = np.mean(ba_values)
                        ci_ba = stats.t.ppf(0.975, len(ba_values) - 1) * stats.sem(ba_values) if len(ba_values) > 1 else 0
                        strategies['baseline']['ba'].append(mean_ba)
                        strategies['baseline']['ba_ci_lower'].append(mean_ba - ci_ba)
                        strategies['baseline']['ba_ci_upper'].append(mean_ba + ci_ba)

        # --- Optimal: find best gallery × eval quality combo for R@1 ---
        best_mean_r1 = -1
        best_r1_values = None
        best_ba_values = None
        best_gal_thresh = None
        best_q_thresh = None

        for gal_thresh in gallery_thresholds:
            filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
            if len(filtered) == 0:
                continue
            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue
            if 'query_quality_metrics' not in all_histories[0][0]:
                continue

            for q_thresh in query_thresholds:
                # Find best epoch by R@1 for this combo
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='recall', query_thresh=q_thresh)

                # Collect R@1 values at that epoch
                r1_values = []
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            if 'query_quality_metrics' in h and q_thresh in h['query_quality_metrics']:
                                r1_values.append(h['query_quality_metrics'][q_thresh]['recall_at_1'])
                            # Also collect balanced accuracy at same epoch with same quality filter
                            if 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if q_thresh in by_quality:
                                    ba_values.append(by_quality[q_thresh]['balanced_accuracy'])
                            break

                if r1_values:
                    mean_r1 = np.mean(r1_values)
                    if mean_r1 > best_mean_r1:
                        best_mean_r1 = mean_r1
                        best_r1_values = r1_values
                        best_ba_values = ba_values
                        best_gal_thresh = gal_thresh
                        best_q_thresh = q_thresh

        if best_r1_values:
            mean_r1 = np.mean(best_r1_values)
            ci_r1 = stats.t.ppf(0.975, len(best_r1_values) - 1) * stats.sem(best_r1_values) if len(best_r1_values) > 1 else 0
            strategies['optimal']['x'].append(gsize)
            strategies['optimal']['r1'].append(mean_r1)
            strategies['optimal']['r1_ci_lower'].append(mean_r1 - ci_r1)
            strategies['optimal']['r1_ci_upper'].append(mean_r1 + ci_r1)
            strategies['optimal']['best_gal'].append(best_gal_thresh)
            strategies['optimal']['best_q'].append(float(best_q_thresh.replace('q>=', '')))

            if best_ba_values:
                mean_ba = np.mean(best_ba_values)
                ci_ba = stats.t.ppf(0.975, len(best_ba_values) - 1) * stats.sem(best_ba_values) if len(best_ba_values) > 1 else 0
                strategies['optimal']['ba'].append(mean_ba)
                strategies['optimal']['ba_ci_lower'].append(mean_ba - ci_ba)
                strategies['optimal']['ba_ci_upper'].append(mean_ba + ci_ba)

        # --- Optimal BA: find best gallery × eval quality combo for BA ---
        best_mean_ba = -1
        best_ba_values_ba = None
        best_gal_thresh_ba = None
        best_q_thresh_ba = None

        for gal_thresh in gallery_thresholds:
            filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
            if len(filtered) == 0:
                continue
            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue
            if 'open_set' not in all_histories[0][0]:
                continue

            for q_thresh in query_thresholds:
                # Find best epoch by BA for this combo
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='ba', query_thresh=q_thresh)

                # Collect BA values at that epoch
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            if 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if q_thresh in by_quality:
                                    ba_values.append(by_quality[q_thresh]['balanced_accuracy'])
                            break

                if ba_values:
                    mean_ba = np.mean(ba_values)
                    if mean_ba > best_mean_ba:
                        best_mean_ba = mean_ba
                        best_ba_values_ba = ba_values
                        best_gal_thresh_ba = gal_thresh
                        best_q_thresh_ba = q_thresh

        if best_ba_values_ba:
            mean_ba = np.mean(best_ba_values_ba)
            ci_ba = stats.t.ppf(0.975, len(best_ba_values_ba) - 1) * stats.sem(best_ba_values_ba) if len(best_ba_values_ba) > 1 else 0
            strategies['optimal_ba']['x'].append(gsize)
            strategies['optimal_ba']['ba'].append(mean_ba)
            strategies['optimal_ba']['ba_ci_lower'].append(mean_ba - ci_ba)
            strategies['optimal_ba']['ba_ci_upper'].append(mean_ba + ci_ba)
            strategies['optimal_ba']['best_gal'].append(best_gal_thresh_ba)
            strategies['optimal_ba']['best_q'].append(float(best_q_thresh_ba.replace('q>=', '')))

    # =========================================================================
    # Panel A: R@1 Strategies
    # =========================================================================
    ax1 = fig.add_subplot(131)

    for key in ['baseline', 'optimal']:
        s = strategies[key]
        if s['x'] and s['r1']:
            ax1.plot(s['x'], s['r1'], color=colors[key], linewidth=2.5,
                     marker='o', markersize=8, label=s['label'])
            ax1.fill_between(s['x'], s['r1_ci_lower'], s['r1_ci_upper'],
                             color=colors[key], alpha=0.2)

    ax1.set_xlabel('Examples per Individual', fontsize=12)
    ax1.set_ylabel('Wolverine re-identification performance (Recall at rank 1)', fontsize=12)
    ax1.set_xticks(gallery_sizes)

    # Set y-axis range
    all_ci_lower = strategies['baseline']['r1_ci_lower'] + strategies['optimal']['r1_ci_lower']
    all_ci_upper = strategies['baseline']['r1_ci_upper'] + strategies['optimal']['r1_ci_upper']
    if all_ci_lower and all_ci_upper:
        y_min = max(0, np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05)
        y_max = min(1, np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05)
        ax1.set_ylim(y_min, y_max)
        ax1.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))

    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    # =========================================================================
    # Panel B: Optimal Thresholds (Grouped Bars)
    # =========================================================================
    ax2 = fig.add_subplot(132)

    x = np.arange(len(strategies['optimal']['x']))
    width = 0.35

    if strategies['optimal']['best_gal'] and strategies['optimal']['best_q']:
        ax2.bar(x - width/2, strategies['optimal']['best_gal'], width,
                color='#1f77b4', label='Gallery')
        ax2.bar(x + width/2, strategies['optimal']['best_q'], width,
                color='#ff7f0e', label='Query')

        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies['optimal']['x'])

    ax2.set_xlabel('Examples per Individual', fontsize=12)
    ax2.set_ylabel(r'Image Quality Threshold ($p$ visible pelage)', fontsize=12)
    ax2.legend(title='Threshold Type', loc='lower right', fontsize=10, title_fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    # =========================================================================
    # Panel C: Balanced Accuracy (Open Set)
    # =========================================================================
    ax3 = fig.add_subplot(133)

    for key in ['baseline', 'optimal_ba']:
        s = strategies[key]
        if s['x'] and s['ba']:
            ax3.plot(s['x'], s['ba'], color=colors[key], linewidth=2.5,
                     marker='o', markersize=8, label=s['label'])
            ax3.fill_between(s['x'], s['ba_ci_lower'], s['ba_ci_upper'],
                             color=colors[key], alpha=0.2)

    ax3.set_xlabel('Examples per Individual', fontsize=12)
    ax3.set_ylabel('Novel wolverine detection performance (Balanced accuracy)', fontsize=12)
    ax3.set_xticks(gallery_sizes)

    # Set y-axis range
    all_ci_lower = strategies['baseline']['ba_ci_lower'] + strategies['optimal_ba']['ba_ci_lower']
    all_ci_upper = strategies['baseline']['ba_ci_upper'] + strategies['optimal_ba']['ba_ci_upper']
    if all_ci_lower and all_ci_upper:
        y_min = max(0, np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05)
        y_max = min(1, np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05)
        ax3.set_ylim(y_min, y_max)
        ax3.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))

    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    # =========================================================================
    # Save figure
    # =========================================================================
    plt.suptitle('Raw Cosine Hygiene Sweep: Model Selection by R@1', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_r1_selection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure: {output_path}")


def plot_recall_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing recall@1 curves over epochs.

    Grid: 6 rows (gallery_size) × 6 columns (hygiene threshold)
    Each facet shows 8 lines (one per seed) for query threshold >= 0.3.
    """
    from matplotlib.lines import Line2D

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    thresholds = sorted(results.get_unique('threshold'))
    q_key = "q>=0.3"

    n_rows, n_cols = len(gallery_sizes), len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15), sharex=True, sharey=True)

    # 8 distinct colors for seeds
    seed_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))

    for i, gallery_size in enumerate(gallery_sizes):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)

            for r in filtered:
                seed = r['seed']
                history = r.get('epoch_history', [])
                epochs = [h['epoch'] for h in history]

                recall_values = [
                    h['query_quality_metrics'][q_key]['recall_at_1']
                    for h in history
                ]
                ax.plot(epochs, recall_values, color=seed_colors[seed],
                        alpha=0.8, linewidth=1)

            # Facet labels
            if i == 0:
                ax.set_title(f'thresh={threshold:.1f}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'gallery={gallery_size}', fontsize=9)

            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    # Legend for seeds
    legend_elements = [
        Line2D([0], [0], color=seed_colors[s], label=f'seed {s}', linewidth=2)
        for s in range(8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
               title='Seed')

    fig.supxlabel('Epoch', fontsize=12)
    fig.supylabel('Recall@1 (val)', fontsize=12)
    fig.suptitle('Validation Recall@1 (q>=0.3) by Configuration (Raw Cosine)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_validation_loss_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing validation loss curves over epochs.

    Grid: 6 rows (gallery_size) × 6 columns (hygiene threshold)
    Each facet shows 8 lines (one per seed) for validation loss.
    """
    from matplotlib.lines import Line2D

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    thresholds = sorted(results.get_unique('threshold'))

    n_rows, n_cols = len(gallery_sizes), len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15), sharex=True, sharey=False)

    # 8 distinct colors for seeds
    seed_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))

    for i, gallery_size in enumerate(gallery_sizes):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)

            for r in filtered:
                seed = r['seed']
                history = r.get('epoch_history', [])

                # Check if val_loss exists (backward compatibility)
                if not history or 'val_loss' not in history[0]:
                    continue

                epochs = [h['epoch'] for h in history]
                val_losses = [h['val_loss'] for h in history]

                ax.plot(epochs, val_losses, color=seed_colors[seed],
                        alpha=0.8, linewidth=1)

            # Facet labels
            if i == 0:
                ax.set_title(f'thresh={threshold:.1f}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'gallery={gallery_size}', fontsize=9)

            ax.grid(True, alpha=0.3)

    # Legend for seeds
    legend_elements = [
        Line2D([0], [0], color=seed_colors[s], label=f'seed {s}', linewidth=2)
        for s in range(8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, title='Seed')

    fig.supxlabel('Epoch', fontsize=12)
    fig.supylabel('Validation Loss (ArcFace)', fontsize=12)
    fig.suptitle('Validation Loss by Configuration (Raw Cosine)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_cosine_threshold(results: ResultsCollection, output_path: str):
    """
    Plot cosine similarity decision threshold by gallery size and quality threshold.

    Shows mean threshold with 95% CI across 8 seeds for each configuration.
    X-axis: Gallery size (examples per individual)
    Lines: Different gallery quality thresholds (0.0, 0.1, ..., 0.5)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    # Use a colormap for different gallery thresholds
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gallery_thresholds)))

    for idx, gal_thresh in enumerate(gallery_thresholds):
        x_vals = []
        y_means = []
        ci_lower = []
        ci_upper = []

        for gsize in gallery_sizes:
            filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
            if len(filtered) == 0:
                continue

            # Get threshold at best epoch for each seed
            all_histories = [r.get('epoch_history', []) for r in filtered]
            best_epoch, _, _ = find_optimal_epoch(all_histories, 'q>=0.0')

            thresh_values = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch and 'open_set' in h:
                        thresh_cal = h['open_set'].get('threshold_calibration', {})
                        if 'threshold' in thresh_cal:
                            thresh_values.append(thresh_cal['threshold'])
                        break

            if thresh_values:
                mean = np.mean(thresh_values)
                ci = stats.t.ppf(0.975, len(thresh_values) - 1) * stats.sem(thresh_values) if len(thresh_values) > 1 else 0
                x_vals.append(gsize)
                y_means.append(mean)
                ci_lower.append(mean - ci)
                ci_upper.append(mean + ci)

        if x_vals:
            ax.plot(x_vals, y_means, color=colors[idx], linewidth=2,
                    marker='o', markersize=6, label=f'q>={gal_thresh}')
            ax.fill_between(x_vals, ci_lower, ci_upper, color=colors[idx], alpha=0.2)

    ax.set_xlabel('Examples per Individual', fontsize=14)
    ax.set_ylabel('Cosine Similarity Threshold', fontsize=14)
    ax.set_title('Cosine Similarity Decision Threshold by Configuration', fontsize=16)
    ax.set_xticks(gallery_sizes)
    ax.set_ylim(0, 1)  # Cosine similarity range
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Gallery quality filter', loc='best', fontsize=10, title_fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def print_summary_table(results: ResultsCollection):
    """Print summary statistics for open-set evaluation using configurable best epoch criterion."""
    print("\n" + "=" * 100)
    print(f"SUMMARY TABLE: Best Epoch by {DEFAULT_BEST_EPOCH_CRITERION.upper()} (R@1 & BA)")
    print(f"Quality threshold: {DEFAULT_QUALITY_THRESHOLD}")
    print("=" * 100)

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    print(f"{'Thresh':<8} {'Gallery':<8} {'Epoch':<6} {'R@1':<8} {'BA':<8} {'H-Mean':<8} {'Seeds':<6}")
    print("-" * 100)

    metrics = {}
    for threshold in gallery_thresholds:
        for gallery_size in gallery_sizes:
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
            if len(filtered) == 0:
                continue

            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue

            # Use configurable criterion for best epoch selection
            best_epoch, criterion_value, details = find_optimal_epoch(all_histories)

            # Collect per-seed metrics at the best epoch
            recall_values = []
            ba_values = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch:
                        r = compute_epoch_metric(h, 'recall', DEFAULT_QUALITY_THRESHOLD)
                        b = compute_epoch_metric(h, 'ba', DEFAULT_QUALITY_THRESHOLD)
                        if r is not None:
                            recall_values.append(r)
                        if b is not None:
                            ba_values.append(b)
                        break

            if recall_values and ba_values:
                mean_recall = np.mean(recall_values)
                mean_ba = np.mean(ba_values)
                # Compute harmonic mean of means
                if (mean_recall + mean_ba) > 0:
                    h_mean = 2 * mean_recall * mean_ba / (mean_recall + mean_ba)
                else:
                    h_mean = 0.0

                print(f"{threshold:<8.2f} {gallery_size:<8} {best_epoch:<6} "
                      f"{mean_recall:<8.4f} {mean_ba:<8.4f} {h_mean:<8.4f} {len(recall_values):<6}")
                metrics[(threshold, gallery_size)] = {
                    'recall': mean_recall,
                    'ba': mean_ba,
                    'h_mean': h_mean,
                    'epoch': best_epoch,
                    'n_seeds': len(recall_values)
                }

    # Find best configuration by harmonic mean
    if metrics:
        best_key = max(metrics.keys(), key=lambda k: metrics[k]['h_mean'])
        best_data = metrics[best_key]
        print("-" * 100)
        print(f"Best config: threshold={best_key[0]}, gallery={best_key[1]}, epoch={best_data['epoch']}")
        print(f"  R@1={best_data['recall']:.4f}, BA={best_data['ba']:.4f}, H-Mean={best_data['h_mean']:.4f}")

        # Also show what other criteria would select
        print("\nAlternative criteria comparison:")
        best_by_recall = max(metrics.keys(), key=lambda k: metrics[k]['recall'])
        best_by_ba = max(metrics.keys(), key=lambda k: metrics[k]['ba'])

        if best_by_recall != best_key:
            d = metrics[best_by_recall]
            print(f"  Best by R@1: thresh={best_by_recall[0]}, gallery={best_by_recall[1]} "
                  f"-> R@1={d['recall']:.4f}, BA={d['ba']:.4f}")
        if best_by_ba != best_key:
            d = metrics[best_by_ba]
            print(f"  Best by BA:  thresh={best_by_ba[0]}, gallery={best_by_ba[1]} "
                  f"-> R@1={d['recall']:.4f}, BA={d['ba']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot Open-Set hygiene sweep results (Raw Cosine)')
    parser.add_argument('--results_dir', type=str,
                        default='reid_openset_BA/results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_BA/figures',
                        help='Directory to save figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Open-Set Evaluation: Gallery Hygiene Sweep Results Analysis (Raw Cosine)")
    print("=" * 60)

    # Load results
    results = load_hygiene_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Print summary
    print_summary_table(results)

    # Generate figures
    print("\nGenerating figures...")

    # Main combined figure (3 panels: R@1, optimal thresholds, open-set performance)
    create_combined_figure(results, args.output_dir)

    # Faceted recall@1 curves by query quality threshold
    plot_recall_curves(results, os.path.join(args.output_dir, 'recall_curves.png'))

    # Faceted validation loss curves
    plot_validation_loss_curves(
        results,
        os.path.join(args.output_dir, 'validation_loss_curves.png')
    )

    # Cosine similarity threshold plot
    plot_cosine_threshold(
        results,
        os.path.join(args.output_dir, 'cosine_threshold.png')
    )

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

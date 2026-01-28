#!/usr/bin/env python3
"""
Script 01: Plot Open-Set Hygiene Sweep Results (T-Norm + NDR)

Generates publication-quality figures from open-set evaluation experiment
using DINOv3 backbone with ArcFace loss, T-Norm score normalization,
and Novelty Detection Rate (NDR) metrics.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) 3x2 faceted figure: NDR at FN=1%, 5%, 10%
   - Row 1: NDR by gallery size (strategy lines)
   - Row 2: NDR by quality threshold (gallery size lines)
B) Closed-set Recall@1 (unchanged from tnorm)
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


def find_best_epoch_for_ndr(all_histories: List[List[dict]],
                            query_thresh: str = "q>=0.0",
                            fnr_level: int = 5) -> Tuple[int, float]:
    """
    Find epoch with best mean NDR at specified FNR level across seeds.

    Args:
        all_histories: List of epoch histories (one per seed)
        query_thresh: Quality threshold key for evaluation (e.g., "q>=0.0")
        fnr_level: FNR level to optimize (1, 5, or 10)

    Returns:
        Tuple of (best_epoch, best_mean_ndr)
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0

    metric_key = f'NoveltyDetectionRate@FalseNovelty={fnr_level}%'

    # Get epochs that have open_set metrics with by_quality
    eval_epochs = []
    for h in all_histories[0]:
        if 'open_set' in h and 'by_quality' in h['open_set']:
            eval_epochs.append(h['epoch'])

    if not eval_epochs:
        return 50, 0.0

    best_epoch = None
    best_mean = -1

    for epoch in eval_epochs:
        ndr_values = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch and 'open_set' in h:
                    by_quality = h['open_set'].get('by_quality', {})
                    if query_thresh in by_quality:
                        ndr = by_quality[query_thresh].get(metric_key, 0.0)
                        ndr_values.append(ndr)
                    break

        if ndr_values:
            mean_ndr = np.mean(ndr_values)
            if mean_ndr > best_mean:
                best_mean = mean_ndr
                best_epoch = epoch

    return best_epoch or 50, best_mean


def find_best_epoch_for_recall(all_histories: List[List[dict]]) -> Tuple[int, float]:
    """Find epoch with best mean overall recall across seeds."""
    if not all_histories or not all_histories[0]:
        return 50, 0.0

    eval_epochs = [h['epoch'] for h in all_histories[0] if 'query_quality_metrics' in h]

    if not eval_epochs:
        return 50, 0.0

    best_epoch = None
    best_mean = -1

    for epoch in eval_epochs:
        recalls = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch and 'query_quality_metrics' in h:
                    recalls.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                    break

        if recalls:
            mean_recall = np.mean(recalls)
            if mean_recall > best_mean:
                best_mean = mean_recall
                best_epoch = epoch

    return best_epoch or 50, best_mean


def create_open_set_ndr_figure(results: ResultsCollection, output_dir: str):
    """
    Create 3x2 faceted figure for open-set NDR metrics.

    Columns: FN rate (1%, 5%, 10%)
    Row 1: NDR vs gallery_size (with strategy lines)
    Row 2: NDR vs quality_threshold (with gallery_size lines)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    fn_rates = [1, 5, 10]  # Column facets
    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    # Colors for strategies (row 1)
    strategy_colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    # Colors for gallery sizes (row 2)
    gsize_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gallery_sizes)))

    # Row 1: NDR by gallery size (strategies)
    for col, fn in enumerate(fn_rates):
        ax = axes[0, col]
        metric_key = f'NoveltyDetectionRate@FalseNovelty={fn}%'

        strategies = {
            'baseline': {'label': 'None', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
            'minimal': {'label': 'Minimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
            'optimal': {'label': 'Optimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []}
        }

        for gsize in gallery_sizes:
            # Baseline: gallery_threshold=0.0, eval q>=0.0
            baseline_filtered = results.filter(threshold=0.0, gallery_size=gsize)
            if len(baseline_filtered) > 0:
                all_histories = [r.get('epoch_history', []) for r in baseline_filtered]
                if all_histories and all_histories[0]:
                    best_epoch, _ = find_best_epoch_for_ndr(all_histories, 'q>=0.0', fn)
                    values = []
                    for history in all_histories:
                        for h in history:
                            if h['epoch'] == best_epoch and 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if 'q>=0.0' in by_quality:
                                    values.append(by_quality['q>=0.0'].get(metric_key, 0.0))
                                break
                    if values:
                        mean = np.mean(values)
                        ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                        strategies['baseline']['x'].append(gsize)
                        strategies['baseline']['y'].append(mean)
                        strategies['baseline']['ci_lower'].append(mean - ci)
                        strategies['baseline']['ci_upper'].append(mean + ci)

            # Minimal: gallery_threshold=0.1, eval q>=0.1
            minimal_filtered = results.filter(threshold=0.1, gallery_size=gsize)
            if len(minimal_filtered) > 0:
                all_histories = [r.get('epoch_history', []) for r in minimal_filtered]
                if all_histories and all_histories[0]:
                    best_epoch, _ = find_best_epoch_for_ndr(all_histories, 'q>=0.1', fn)
                    values = []
                    for history in all_histories:
                        for h in history:
                            if h['epoch'] == best_epoch and 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if 'q>=0.1' in by_quality:
                                    values.append(by_quality['q>=0.1'].get(metric_key, 0.0))
                                break
                    if values:
                        mean = np.mean(values)
                        ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                        strategies['minimal']['x'].append(gsize)
                        strategies['minimal']['y'].append(mean)
                        strategies['minimal']['ci_lower'].append(mean - ci)
                        strategies['minimal']['ci_upper'].append(mean + ci)

            # Optimal: best gallery x eval quality combo
            best_mean = -1
            best_values = None
            for gal_thresh in gallery_thresholds:
                filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
                if len(filtered) == 0:
                    continue
                all_histories = [r.get('epoch_history', []) for r in filtered]
                if not all_histories or not all_histories[0]:
                    continue

                for q_thresh in query_thresholds:
                    best_epoch, _ = find_best_epoch_for_ndr(all_histories, q_thresh, fn)
                    values = []
                    for history in all_histories:
                        for h in history:
                            if h['epoch'] == best_epoch and 'open_set' in h:
                                by_quality = h['open_set'].get('by_quality', {})
                                if q_thresh in by_quality:
                                    values.append(by_quality[q_thresh].get(metric_key, 0.0))
                                break
                    if values:
                        mean = np.mean(values)
                        if mean > best_mean:
                            best_mean = mean
                            best_values = values

            if best_values:
                mean = np.mean(best_values)
                ci = stats.t.ppf(0.975, len(best_values) - 1) * stats.sem(best_values) if len(best_values) > 1 else 0
                strategies['optimal']['x'].append(gsize)
                strategies['optimal']['y'].append(mean)
                strategies['optimal']['ci_lower'].append(mean - ci)
                strategies['optimal']['ci_upper'].append(mean + ci)

        # Plot strategies
        for key in ['baseline', 'minimal', 'optimal']:
            s = strategies[key]
            if s['x']:
                ax.plot(s['x'], s['y'], color=strategy_colors[key], linewidth=2,
                        marker='o', markersize=6, label=s['label'])
                ax.fill_between(s['x'], s['ci_lower'], s['ci_upper'],
                                color=strategy_colors[key], alpha=0.2)

        ax.set_title(f'FN={fn}%', fontsize=12)
        ax.set_xlabel('Examples per Individual')
        if col == 0:
            ax.set_ylabel('Novelty Detection Rate')
            ax.legend(title='Strategy', loc='lower right', fontsize=8)
        ax.set_xticks(gallery_sizes)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Row 2: NDR by quality threshold (gallery size lines)
    for col, fn in enumerate(fn_rates):
        ax = axes[1, col]
        metric_key = f'NoveltyDetectionRate@FalseNovelty={fn}%'

        for gsize_idx, gsize in enumerate(gallery_sizes):
            x_vals = []
            y_means = []
            ci_lower = []
            ci_upper = []

            for gal_thresh in gallery_thresholds:
                filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
                if len(filtered) == 0:
                    continue

                all_histories = [r.get('epoch_history', []) for r in filtered]
                if not all_histories or not all_histories[0]:
                    continue

                # Use same quality threshold for eval as for gallery
                q_thresh = f'q>={gal_thresh}'
                best_epoch, _ = find_best_epoch_for_ndr(all_histories, q_thresh, fn)

                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if q_thresh in by_quality:
                                values.append(by_quality[q_thresh].get(metric_key, 0.0))
                            break

                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    x_vals.append(gal_thresh)
                    y_means.append(mean)
                    ci_lower.append(mean - ci)
                    ci_upper.append(mean + ci)

            if x_vals:
                ax.plot(x_vals, y_means, color=gsize_colors[gsize_idx], linewidth=2,
                        marker='s', markersize=5, label=f'n={gsize}')
                ax.fill_between(x_vals, ci_lower, ci_upper,
                                color=gsize_colors[gsize_idx], alpha=0.15)

        ax.set_xlabel('Quality Threshold')
        if col == 0:
            ax.set_ylabel('Novelty Detection Rate')
            ax.legend(title='Gallery Size', loc='lower right', fontsize=7, ncol=2)
        ax.set_xticks(gallery_thresholds)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Open-Set NDR at Different False Novelty Rate Tolerances', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'open_set_ndr_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_closed_set_figure(results: ResultsCollection, output_dir: str):
    """Create combined 2-panel figure for closed-set evaluation (Recall@1)."""
    fig = plt.figure(figsize=(14, 6))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    # Panel A: Recall@1 filtration strategies
    ax1 = fig.add_subplot(121)

    strategies = {
        'baseline': {'label': 'None', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'minimal': {'label': 'Minimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'optimal': {'label': 'Optimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': [], 'best_gal': [], 'best_q': []}
    }

    for gsize in gallery_sizes:
        # Baseline
        baseline_filtered = results.filter(threshold=0.0, gallery_size=gsize)
        if len(baseline_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in baseline_filtered]
            if all_histories and all_histories[0] and 'query_quality_metrics' in all_histories[0][0]:
                best_epoch, _ = find_best_epoch_for_recall(all_histories)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'query_quality_metrics' in h:
                            values.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                            break
                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    strategies['baseline']['x'].append(gsize)
                    strategies['baseline']['y'].append(mean)
                    strategies['baseline']['ci_lower'].append(mean - ci)
                    strategies['baseline']['ci_upper'].append(mean + ci)

        # Minimal
        minimal_filtered = results.filter(threshold=0.1, gallery_size=gsize)
        if len(minimal_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in minimal_filtered]
            if all_histories and all_histories[0] and 'query_quality_metrics' in all_histories[0][0]:
                best_epoch, _ = find_best_epoch_for_recall(all_histories)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'query_quality_metrics' in h:
                            if 'q>=0.1' in h['query_quality_metrics']:
                                values.append(h['query_quality_metrics']['q>=0.1']['recall_at_1'])
                            break
                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    strategies['minimal']['x'].append(gsize)
                    strategies['minimal']['y'].append(mean)
                    strategies['minimal']['ci_lower'].append(mean - ci)
                    strategies['minimal']['ci_upper'].append(mean + ci)

        # Optimal
        best_mean = -1
        best_values = None
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
            best_epoch, _ = find_best_epoch_for_recall(all_histories)
            for q_thresh in query_thresholds:
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'query_quality_metrics' in h:
                            if q_thresh in h['query_quality_metrics']:
                                values.append(h['query_quality_metrics'][q_thresh]['recall_at_1'])
                            break
                if values:
                    mean = np.mean(values)
                    if mean > best_mean:
                        best_mean = mean
                        best_values = values
                        best_gal_thresh = gal_thresh
                        best_q_thresh = q_thresh
        if best_values:
            mean = np.mean(best_values)
            ci = stats.t.ppf(0.975, len(best_values) - 1) * stats.sem(best_values) if len(best_values) > 1 else 0
            strategies['optimal']['x'].append(gsize)
            strategies['optimal']['y'].append(mean)
            strategies['optimal']['ci_lower'].append(mean - ci)
            strategies['optimal']['ci_upper'].append(mean + ci)
            strategies['optimal']['best_gal'].append(best_gal_thresh)
            strategies['optimal']['best_q'].append(best_q_thresh.replace('q>=', ''))

    for key in ['baseline', 'minimal', 'optimal']:
        s = strategies[key]
        if s['x']:
            ax1.plot(s['x'], s['y'], color=colors[key], linewidth=2, marker='o', label=s['label'])
            ax1.fill_between(s['x'], s['ci_lower'], s['ci_upper'], color=colors[key], alpha=0.2)

    ax1.set_xlabel('Examples per Individual')
    ax1.set_ylabel('Recall@1 (Known Wolverines)')
    ax1.set_xticks(gallery_sizes)

    all_ci_lower = [v for s in strategies.values() for v in s['ci_lower']]
    all_ci_upper = [v for s in strategies.values() for v in s['ci_upper']]
    if all_ci_lower and all_ci_upper:
        y_min = max(0, np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05)
        y_max = min(1, np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05)
        ax1.set_ylim(y_min, y_max)
        ax1.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))

    ax1.legend(title='Image quality filter', fontsize=9, title_fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    # Panel B: Optimal thresholds for Recall@1
    ax2 = fig.add_subplot(122)

    best_gal_thresholds = []
    best_query_thresholds = []

    for gsize in gallery_sizes:
        best_mean = -1
        best_gal = None
        best_q = None

        for gal_thresh in gallery_thresholds:
            filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
            if len(filtered) == 0:
                continue
            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue
            if 'query_quality_metrics' not in all_histories[0][0]:
                continue

            best_epoch, _ = find_best_epoch_for_recall(all_histories)

            for q_thresh in query_thresholds:
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'query_quality_metrics' in h:
                            if q_thresh in h['query_quality_metrics']:
                                values.append(h['query_quality_metrics'][q_thresh]['recall_at_1'])
                            break
                if values:
                    mean = np.mean(values)
                    if mean > best_mean:
                        best_mean = mean
                        best_gal = gal_thresh
                        best_q = float(q_thresh.replace('q>=', ''))

        best_gal_thresholds.append(best_gal if best_gal is not None else 0)
        best_query_thresholds.append(best_q if best_q is not None else 0)

    x = np.arange(len(gallery_sizes))
    width = 0.35

    ax2.bar(x - width/2, best_gal_thresholds, width, color='#1f77b4', label='Training Gallery')
    ax2.bar(x + width/2, best_query_thresholds, width, color='#ff7f0e', label='Validation Query')

    ax2.set_xticks(x)
    ax2.set_xticklabels(gallery_sizes)
    ax2.set_xlabel('Examples per Individual')
    ax2.set_ylabel(r'Optimal quality threshold ($p$ visible pelage)')
    ax2.legend(title='Threshold Application', fontsize=9, title_fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    plt.suptitle('Closed-Set Evaluation: Gallery Hygiene Sweep Results (T-Norm)', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'closed_set_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved closed-set figure: {output_path}")


def plot_recall_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing recall@1 curves over epochs.

    Grid: 6 rows (gallery_size) x 6 columns (hygiene threshold)
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
    fig.suptitle('Validation Recall@1 (q>=0.3) by Configuration (T-Norm + NDR)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def print_summary_table(results: ResultsCollection):
    """Print summary statistics for open-set NDR evaluation."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Open-Set NDR by Configuration (Best Epoch Selection)")
    print("=" * 100)

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    print(f"{'Thresh':<8} {'Gallery':<10} {'Best Ep':<10} "
          f"{'NDR@1%':<12} {'NDR@5%':<12} {'NDR@10%':<12} "
          f"{'T@1%':<10} {'T@5%':<10} {'T@10%':<10} {'N Seeds':<8}")
    print("-" * 100)

    for threshold in gallery_thresholds:
        for gallery_size in gallery_sizes:
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
            if len(filtered) == 0:
                continue

            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue

            best_epoch, _ = find_best_epoch_for_ndr(all_histories, 'q>=0.0', fnr_level=5)

            ndr1_values, ndr5_values, ndr10_values = [], [], []
            thresh1_values, thresh5_values, thresh10_values = [], [], []

            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch and 'open_set' in h:
                        by_quality = h['open_set'].get('by_quality', {})
                        if 'q>=0.0' in by_quality:
                            q_data = by_quality['q>=0.0']
                            ndr1_values.append(q_data.get('NoveltyDetectionRate@FalseNovelty=1%', 0.0))
                            ndr5_values.append(q_data.get('NoveltyDetectionRate@FalseNovelty=5%', 0.0))
                            ndr10_values.append(q_data.get('NoveltyDetectionRate@FalseNovelty=10%', 0.0))
                            thresh1_values.append(q_data.get('Threshold_FN=1%', 0.0))
                            thresh5_values.append(q_data.get('Threshold_FN=5%', 0.0))
                            thresh10_values.append(q_data.get('Threshold_FN=10%', 0.0))
                        break

            if ndr5_values:
                print(f"{threshold:<8.2f} {gallery_size:<10} {best_epoch:<10} "
                      f"{np.mean(ndr1_values):.4f}{'':>6} {np.mean(ndr5_values):.4f}{'':>6} {np.mean(ndr10_values):.4f}{'':>6} "
                      f"{np.mean(thresh1_values):.2f}{'':>6} {np.mean(thresh5_values):.2f}{'':>6} {np.mean(thresh10_values):.2f}{'':>6} "
                      f"{len(ndr5_values):<8}")

    # Find best configuration by NDR@5%
    best_ndr5 = -1
    best_config = None
    for threshold in gallery_thresholds:
        for gallery_size in gallery_sizes:
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
            if len(filtered) == 0:
                continue
            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue
            best_epoch, mean_ndr = find_best_epoch_for_ndr(all_histories, 'q>=0.0', fnr_level=5)
            if mean_ndr > best_ndr5:
                best_ndr5 = mean_ndr
                best_config = (threshold, gallery_size, best_epoch)

    if best_config:
        print("-" * 100)
        print(f"Best by NDR@5%: threshold={best_config[0]}, gallery_size={best_config[1]}, "
              f"epoch={best_config[2]} -> NDR@5%={best_ndr5:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot Open-Set hygiene sweep results (T-Norm + NDR)')
    parser.add_argument('--results_dir', type=str,
                        default='reid_openset_NDR/results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_NDR/figures',
                        help='Directory to save figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Open-Set Evaluation: Gallery Hygiene Sweep Results Analysis (T-Norm + NDR)")
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

    # Main NDR faceted figure
    create_open_set_ndr_figure(results, args.output_dir)

    # Closed-set figures (unchanged)
    create_closed_set_figure(results, args.output_dir)

    # Faceted recall@1 curves
    plot_recall_curves(results, os.path.join(args.output_dir, 'recall_curves.png'))

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

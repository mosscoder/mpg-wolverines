#!/usr/bin/env python3
"""
Script 01: Plot Gallery Hygiene Sweep Results

Generates publication-quality figures from hygiene filter experiment.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) Line plot: Filtration strategies - baseline (no filter), minimal (G≥0.1, Q≥0.1),
   and optimal (best G×Q combo) vs examples per individual
B) Bar chart: Best gallery × query filter combination per examples per individual
C) Gallery × Query quality bar chart: Shows how gallery filtering affects
   retrieval performance across different query quality levels
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


def plot_filtration_strategies(results: ResultsCollection, output_path: str):
    """
    Panel A: Line plot comparing filtration strategies.

    Three lines:
    - Baseline (no filtration): gallery_threshold=0.0, query q>=0.0
    - Minimal filtration: gallery_threshold=0.1, query q>=0.1
    - Optimal filtration: best combo of gallery and query filter per gallery_size
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    # Colors for the 3 strategies
    colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    strategies = {
        'baseline': {'label': 'None', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'minimal': {'label': 'Minimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'optimal': {'label': 'Optimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': [], 'best_gal': [], 'best_q': []}
    }

    for gsize in gallery_sizes:
        # --- Baseline: gallery_threshold=0.0, query q>=0.0 ---
        baseline_filtered = results.filter(threshold=0.0, gallery_size=gsize)
        if len(baseline_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in baseline_filtered]
            if all_histories and all_histories[0] and 'query_quality_metrics' in all_histories[0][0]:
                best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)
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

        # --- Minimal: gallery_threshold=0.1, query q>=0.1 ---
        minimal_filtered = results.filter(threshold=0.1, gallery_size=gsize)
        if len(minimal_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in minimal_filtered]
            if all_histories and all_histories[0] and 'query_quality_metrics' in all_histories[0][0]:
                best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)
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

        # --- Optimal: find best gallery × query combo for this gallery_size ---
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

            best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

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

    # Plot each strategy
    for key in ['baseline', 'minimal', 'optimal']:
        s = strategies[key]
        if s['x']:
            line, = ax.plot(s['x'], s['y'], color=colors[key], linewidth=2.5,
                           marker='o', markersize=8, label=s['label'])
            ax.fill_between(s['x'], s['ci_lower'], s['ci_upper'], color=colors[key], alpha=0.2)

    ax.set_xlabel('Examples per Individual', fontsize=14)
    ax.set_ylabel('Wolverine re-identification performance (Recall@1)', fontsize=14)
    ax.set_xticks(gallery_sizes)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Image quality filter', loc='lower right', fontsize=11, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_optimal_thresholds(results: ResultsCollection, output_path: str):
    """
    Panel B: Line plot showing optimal thresholds for training gallery and validation queries.

    For each examples-per-individual value, finds the optimal G×Q combo and plots
    the threshold values as two lines.
    """
    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    fig, ax = plt.subplots(figsize=(10, 7))

    best_gal_thresholds = []
    best_query_thresholds = []

    for gsize in gallery_sizes:
        # Find optimal combo for this gallery_size
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

            best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

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

    # Plot grouped bars
    x = np.arange(len(gallery_sizes))
    width = 0.35

    ax.bar(x - width/2, best_gal_thresholds, width, color='#1f77b4', label='Training Gallery')
    ax.bar(x + width/2, best_query_thresholds, width, color='#ff7f0e', label='Validation Query')

    ax.set_xticks(x)
    ax.set_xticklabels(gallery_sizes)
    ax.set_xlabel('Examples per Individual', fontsize=12)
    ax.set_ylabel(r'Optimal image quality threshold ($p$ visible pelage)', fontsize=12)
    ax.legend(title='Threshold Application', fontsize=11, title_fontsize=11, loc='lower right')
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


def plot_gallery_vs_query_quality(results: ResultsCollection, output_path: str, gallery_size: int = 64):
    """
    Bar chart: Gallery threshold (hue) × Query quality threshold (x-axis) → Recall@1.

    For each gallery threshold, finds the best epoch based on overall recall (q>=0.0),
    then extracts query_quality_metrics at that epoch for all query thresholds.

    Args:
        results: ResultsCollection with experiment results
        output_path: Path to save the figure
        gallery_size: Gallery size to filter for
    """
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(query_thresholds))
    width = 0.8 / len(gallery_thresholds)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gallery_thresholds)))

    for i, gal_thresh in enumerate(gallery_thresholds):
        # Get results for this gallery_size and gallery_threshold
        filtered = results.filter(threshold=gal_thresh, gallery_size=gallery_size)
        if len(filtered) == 0:
            continue

        all_histories = [r.get('epoch_history', []) for r in filtered]

        if not all_histories or not all_histories[0]:
            continue

        # Check if new format with query_quality_metrics exists
        sample_entry = all_histories[0][0]
        if 'query_quality_metrics' not in sample_entry:
            print(f"  Skipping gal_thresh={gal_thresh}: no query_quality_metrics in data")
            continue

        # Find best epoch (based on q>=0.0 overall recall)
        best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

        # Extract query_quality_metrics at best epoch for each seed
        means, errors = [], []
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
                if len(values) > 1:
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
                else:
                    ci = 0
            else:
                mean = 0
                ci = 0
            means.append(mean)
            errors.append(ci)

        offset = (i - len(gallery_thresholds) / 2 + 0.5) * width
        label = f'G≥{gal_thresh}' + (' (baseline)' if gal_thresh == 0.0 else '')
        ax.bar(x + offset, means, width, yerr=errors, label=label, color=colors[i], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['≥0.0\n(all)', '≥0.1', '≥0.2', '≥0.3', '≥0.4', '≥0.5'])
    ax.set_xlabel('Query Quality Threshold', fontsize=12)
    ax.set_ylabel('Recall@1 (Best Epoch)', fontsize=12)
    ax.set_title(f'Gallery × Query Quality Interaction (gallery_size={gallery_size})', fontsize=14)
    ax.legend(title='Gallery Filter', loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_main_figure(results: ResultsCollection, output_dir: str):
    """Create combined 2-panel figure."""
    fig = plt.figure(figsize=(14, 6))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    # Colors for the 3 strategies
    colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    # Panel A: Filtration strategies line plot
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
                best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)
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
                best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)
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
            best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)
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
    ax1.set_ylabel('Wolverine re-identification performance (Recall@1)')
    ax1.set_xticks(gallery_sizes)
    ax1.legend(title='Image quality filter', fontsize=9, title_fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    # Panel B: Optimal thresholds for training gallery and validation queries
    ax2 = fig.add_subplot(122)

    best_gal_thresholds = []
    best_query_thresholds = []

    for gsize in gallery_sizes:
        # Find optimal combo for this gallery_size
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

            best_epoch, _ = find_best_epoch_for_overall_recall(all_histories)

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

    # Plot grouped bars
    x = np.arange(len(gallery_sizes))
    width = 0.35

    ax2.bar(x - width/2, best_gal_thresholds, width, color='#1f77b4', label='Training Gallery')
    ax2.bar(x + width/2, best_query_thresholds, width, color='#ff7f0e', label='Validation Query')

    ax2.set_xticks(x)
    ax2.set_xticklabels(gallery_sizes)
    ax2.set_xlabel('Examples per Individual')
    ax2.set_ylabel(r'Optimal image quality threshold ($p$ visible pelage)')
    ax2.legend(title='Threshold Application', fontsize=9, title_fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

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

    plot_filtration_strategies(
        results,
        os.path.join(args.output_dir, 'panel_a_filtration_strategies.png')
    )

    plot_optimal_thresholds(
        results,
        os.path.join(args.output_dir, 'panel_b_optimal_thresholds.png')
    )

    # Generate combined main figure
    create_main_figure(results, args.output_dir)

    # Gallery × Query quality bar charts for each gallery size
    for gsize in sorted(results.get_unique('gallery_size')):
        try:
            plot_gallery_vs_query_quality(
                results,
                os.path.join(args.output_dir, f'gallery_query_quality_N={gsize}.png'),
                gallery_size=gsize
            )
        except Exception as e:
            print(f"Could not generate gallery vs query plot for gallery_size={gsize}: {e}")

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

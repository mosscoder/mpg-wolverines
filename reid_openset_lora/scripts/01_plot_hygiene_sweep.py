#!/usr/bin/env python3
"""
Script 01: Plot Open-Set Hygiene Sweep Results (LoRA)

Generates publication-quality figures from open-set evaluation experiment
using DINOv3 backbone with LoRA adapters and ArcFace loss.

All metrics are computed and stored during training (in epoch_history),
so no GPU inference is needed for plotting.

Figures:
A) Line plot: Correct Flag Rate for novel wolverines - baseline (no filter),
   minimal (G>=0.1, Q>=0.1), and optimal filtration vs examples per individual
B) Bar chart: Optimal quality thresholds for maximizing correct flag rate
C) Closed-set Recall@1 (same as reid_openset_arcface for comparison)
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


def find_best_epoch_for_balanced_accuracy(all_histories: List[List[dict]],
                                           query_thresh: str = "q>=0.0") -> Tuple[int, float]:
    """
    Find epoch with best mean balanced accuracy across seeds.

    Args:
        all_histories: List of epoch histories (one per seed)
        query_thresh: Quality threshold key for evaluation (e.g., "q>=0.0")

    Returns:
        Tuple of (best_epoch, best_mean_ba)
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0

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
        ba_values = []
        for history in all_histories:
            for h in history:
                if h['epoch'] == epoch and 'open_set' in h:
                    by_quality = h['open_set'].get('by_quality', {})
                    if query_thresh in by_quality:
                        ba = by_quality[query_thresh].get('balanced_accuracy', 0.0)
                        ba_values.append(ba)
                    break

        if ba_values:
            mean_ba = np.mean(ba_values)
            if mean_ba > best_mean:
                best_mean = mean_ba
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


def plot_balanced_accuracy_strategies(results: ResultsCollection, output_path: str):
    """
    Panel A: Line plot comparing filtration strategies for balanced accuracy.

    Three lines:
    - Baseline (no filtration): gallery_threshold=0.0, eval q>=0.0
    - Minimal filtration: gallery_threshold=0.1, eval q>=0.1
    - Optimal filtration: best gallery threshold + eval quality per gallery_size
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
        # --- Baseline: gallery_threshold=0.0, eval q>=0.0 ---
        baseline_filtered = results.filter(threshold=0.0, gallery_size=gsize)
        if len(baseline_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in baseline_filtered]
            if all_histories and all_histories[0]:
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, 'q>=0.0')
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if 'q>=0.0' in by_quality:
                                values.append(by_quality['q>=0.0']['balanced_accuracy'])
                            break
                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    strategies['baseline']['x'].append(gsize)
                    strategies['baseline']['y'].append(mean)
                    strategies['baseline']['ci_lower'].append(mean - ci)
                    strategies['baseline']['ci_upper'].append(mean + ci)

        # --- Minimal: gallery_threshold=0.1, eval q>=0.1 ---
        minimal_filtered = results.filter(threshold=0.1, gallery_size=gsize)
        if len(minimal_filtered) > 0:
            all_histories = [r.get('epoch_history', []) for r in minimal_filtered]
            if all_histories and all_histories[0]:
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, 'q>=0.1')
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if 'q>=0.1' in by_quality:
                                values.append(by_quality['q>=0.1']['balanced_accuracy'])
                            break
                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    strategies['minimal']['x'].append(gsize)
                    strategies['minimal']['y'].append(mean)
                    strategies['minimal']['ci_lower'].append(mean - ci)
                    strategies['minimal']['ci_upper'].append(mean + ci)

        # --- Optimal: find best gallery x eval quality combo for this gallery_size ---
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

            for q_thresh in query_thresholds:
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, q_thresh)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if q_thresh in by_quality:
                                values.append(by_quality[q_thresh]['balanced_accuracy'])
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
    ax.set_ylabel('Balanced Accuracy (Known + Unknown)', fontsize=14)
    ax.set_title('Open-Set Evaluation: Balanced Accuracy (LoRA)', fontsize=16)
    ax.set_xticks(gallery_sizes)

    # Set y-axis range based on CI bands with 0.05 buffer
    all_ci_lower = [v for s in strategies.values() for v in s['ci_lower']]
    all_ci_upper = [v for s in strategies.values() for v in s['ci_upper']]
    if all_ci_lower and all_ci_upper:
        y_min = max(0, np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05)
        y_max = min(1, np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))

    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Image quality filter', loc='lower right', fontsize=11, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_optimal_thresholds_balanced(results: ResultsCollection, output_path: str):
    """
    Panel B: Bar plot showing optimal thresholds for maximizing balanced accuracy.

    For each examples-per-individual value, finds the optimal gallery threshold
    and evaluation quality threshold combination.
    """
    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    fig, ax = plt.subplots(figsize=(10, 7))

    best_gal_thresholds = []
    best_eval_thresholds = []

    for gsize in gallery_sizes:
        # Find optimal gallery + eval quality combo for this gallery_size
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

            for q_thresh in query_thresholds:
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, q_thresh)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if q_thresh in by_quality:
                                values.append(by_quality[q_thresh]['balanced_accuracy'])
                            break
                if values:
                    mean = np.mean(values)
                    if mean > best_mean:
                        best_mean = mean
                        best_gal = gal_thresh
                        best_q = float(q_thresh.replace('q>=', ''))

        best_gal_thresholds.append(best_gal if best_gal is not None else 0)
        best_eval_thresholds.append(best_q if best_q is not None else 0)

    # Plot grouped bars
    x = np.arange(len(gallery_sizes))
    width = 0.35

    ax.bar(x - width/2, best_gal_thresholds, width, color='#1f77b4', label='Training Gallery')
    ax.bar(x + width/2, best_eval_thresholds, width, color='#ff7f0e', label='Evaluation Query')

    ax.set_xticks(x)
    ax.set_xticklabels(gallery_sizes)
    ax.set_xlabel('Examples per Individual', fontsize=12)
    ax.set_ylabel(r'Optimal quality threshold ($p$ visible pelage)', fontsize=12)
    ax.set_title('Optimal Thresholds for Balanced Accuracy (LoRA)', fontsize=14)
    ax.legend(title='Threshold Application', fontsize=11, title_fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_recall_strategies(results: ResultsCollection, output_path: str):
    """
    Panel C: Line plot comparing filtration strategies for closed-set Recall@1.
    (Same as reid_openset_arcface for comparison)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    strategies = {
        'baseline': {'label': 'None', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'minimal': {'label': 'Minimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []},
        'optimal': {'label': 'Optimal', 'x': [], 'y': [], 'ci_lower': [], 'ci_upper': []}
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
        if best_values:
            mean = np.mean(best_values)
            ci = stats.t.ppf(0.975, len(best_values) - 1) * stats.sem(best_values) if len(best_values) > 1 else 0
            strategies['optimal']['x'].append(gsize)
            strategies['optimal']['y'].append(mean)
            strategies['optimal']['ci_lower'].append(mean - ci)
            strategies['optimal']['ci_upper'].append(mean + ci)

    for key in ['baseline', 'minimal', 'optimal']:
        s = strategies[key]
        if s['x']:
            ax.plot(s['x'], s['y'], color=colors[key], linewidth=2.5, marker='o', markersize=8, label=s['label'])
            ax.fill_between(s['x'], s['ci_lower'], s['ci_upper'], color=colors[key], alpha=0.2)

    ax.set_xlabel('Examples per Individual', fontsize=14)
    ax.set_ylabel('Recall@1 (Known Wolverines)', fontsize=14)
    ax.set_title('Closed-Set Evaluation: Known Individual Re-ID (LoRA)', fontsize=16)
    ax.set_xticks(gallery_sizes)

    all_ci_lower = [v for s in strategies.values() for v in s['ci_lower']]
    all_ci_upper = [v for s in strategies.values() for v in s['ci_upper']]
    if all_ci_lower and all_ci_upper:
        y_min = np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05
        y_max = np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))

    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Image quality filter', loc='lower right', fontsize=11, title_fontsize=11)

    plt.tight_layout()
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
        y_min = np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05
        y_max = np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05
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

    plt.suptitle('Closed-Set Evaluation: Gallery Hygiene Sweep Results (LoRA)', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'closed_set_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved closed-set figure: {output_path}")


def create_open_set_figure(results: ResultsCollection, output_dir: str):
    """Create combined 2-panel figure for open-set evaluation (Balanced Accuracy)."""
    fig = plt.figure(figsize=(14, 6))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    colors = {'baseline': '#1f77b4', 'minimal': '#ff7f0e', 'optimal': '#2ca02c'}

    # Panel A: Balanced Accuracy strategies
    ax1 = fig.add_subplot(121)

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
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, 'q>=0.0')
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if 'q>=0.0' in by_quality:
                                values.append(by_quality['q>=0.0']['balanced_accuracy'])
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
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, 'q>=0.1')
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if 'q>=0.1' in by_quality:
                                values.append(by_quality['q>=0.1']['balanced_accuracy'])
                            break
                if values:
                    mean = np.mean(values)
                    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
                    strategies['minimal']['x'].append(gsize)
                    strategies['minimal']['y'].append(mean)
                    strategies['minimal']['ci_lower'].append(mean - ci)
                    strategies['minimal']['ci_upper'].append(mean + ci)

        # Optimal: find best gallery x eval quality combo
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
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, q_thresh)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if q_thresh in by_quality:
                                values.append(by_quality[q_thresh]['balanced_accuracy'])
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

    for key in ['baseline', 'minimal', 'optimal']:
        s = strategies[key]
        if s['x']:
            ax1.plot(s['x'], s['y'], color=colors[key], linewidth=2, marker='o', label=s['label'])
            ax1.fill_between(s['x'], s['ci_lower'], s['ci_upper'], color=colors[key], alpha=0.2)

    ax1.set_xlabel('Examples per Individual')
    ax1.set_ylabel('Balanced Accuracy (Known + Unknown)')
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

    # Panel B: Optimal thresholds for balanced accuracy
    ax2 = fig.add_subplot(122)

    best_gal_thresholds = []
    best_eval_thresholds = []

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

            for q_thresh in query_thresholds:
                best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, q_thresh)
                values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch and 'open_set' in h:
                            by_quality = h['open_set'].get('by_quality', {})
                            if q_thresh in by_quality:
                                values.append(by_quality[q_thresh]['balanced_accuracy'])
                            break
                if values:
                    mean = np.mean(values)
                    if mean > best_mean:
                        best_mean = mean
                        best_gal = gal_thresh
                        best_q = float(q_thresh.replace('q>=', ''))

        best_gal_thresholds.append(best_gal if best_gal is not None else 0)
        best_eval_thresholds.append(best_q if best_q is not None else 0)

    x = np.arange(len(gallery_sizes))
    width = 0.35

    ax2.bar(x - width/2, best_gal_thresholds, width, color='#1f77b4', label='Training Gallery')
    ax2.bar(x + width/2, best_eval_thresholds, width, color='#ff7f0e', label='Evaluation Query')

    ax2.set_xticks(x)
    ax2.set_xticklabels(gallery_sizes)
    ax2.set_xlabel('Examples per Individual')
    ax2.set_ylabel(r'Optimal quality threshold ($p$ visible pelage)')
    ax2.legend(title='Threshold Application', fontsize=9, title_fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='left')

    plt.suptitle('Open-Set Evaluation: Gallery Hygiene Sweep Results (LoRA)', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'open_set_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved open-set figure: {output_path}")


def print_summary_table(results: ResultsCollection):
    """Print summary statistics for open-set evaluation."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Open-Set Balanced Accuracy by Configuration (Best Epoch Selection)")
    print("=" * 80)

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    print(f"{'Thresh':<8} {'Gallery':<10} {'Best Ep':<10} {'BA Mean':<12} {'BA Std':<12} {'N Seeds':<8}")
    print("-" * 80)

    metrics = {}
    for threshold in gallery_thresholds:
        for gallery_size in gallery_sizes:
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
            if len(filtered) == 0:
                continue

            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue

            best_epoch, _ = find_best_epoch_for_balanced_accuracy(all_histories, 'q>=0.0')

            values = []
            for history in all_histories:
                for h in history:
                    if h['epoch'] == best_epoch and 'open_set' in h:
                        by_quality = h['open_set'].get('by_quality', {})
                        if 'q>=0.0' in by_quality:
                            values.append(by_quality['q>=0.0']['balanced_accuracy'])
                        break

            if values:
                mean_ba = np.mean(values)
                std_ba = np.std(values, ddof=1) if len(values) > 1 else 0
                print(f"{threshold:<8.2f} {gallery_size:<10} {best_epoch:<10} {mean_ba:.4f}{'':>6} "
                      f"{std_ba:.4f}{'':>6} {len(values):<8}")
                metrics[(threshold, gallery_size)] = {'mean': mean_ba, 'std': std_ba, 'epoch': best_epoch}

    # Find best configuration
    if metrics:
        best_key = max(metrics.keys(), key=lambda k: metrics[k]['mean'])
        best_data = metrics[best_key]
        print("-" * 80)
        print(f"Best: threshold={best_key[0]}, gallery_size={best_key[1]}, epoch={best_data['epoch']} -> "
              f"BA={best_data['mean']:.4f}")


def print_trainable_params_summary(results: ResultsCollection):
    """Print summary of trainable parameters from LoRA configuration."""
    print("\n" + "=" * 60)
    print("TRAINABLE PARAMETERS (LoRA Configuration)")
    print("=" * 60)

    # Get first result to extract param counts
    if len(results) > 0:
        first_result = results[0]
        trainable_params = first_result.get('trainable_params', {})
        config = first_result.get('config', {})

        print(f"LoRA rank (r): {config.get('lora_r', 'N/A')}")
        print(f"LoRA alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"LoRA dropout: {config.get('lora_dropout', 'N/A')}")
        print()
        print(f"Parameter counts:")
        print(f"  LoRA adapters: {trainable_params.get('lora', 'N/A'):,}")
        print(f"  Embedding head: {trainable_params.get('head', 'N/A'):,}")
        print(f"  ArcFace weights: {trainable_params.get('arcface', 'N/A'):,}")
        print(f"  Total trainable: {trainable_params.get('total', 'N/A'):,}")


def main():
    parser = argparse.ArgumentParser(description='Plot Open-Set hygiene sweep results (LoRA)')
    parser.add_argument('--results_dir', type=str,
                        default='reid_openset_lora/results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_lora/figures',
                        help='Directory to save figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Open-Set Evaluation: Gallery Hygiene Sweep Results (LoRA)")
    print("=" * 60)

    # Load results
    results = load_hygiene_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Print LoRA parameters summary
    print_trainable_params_summary(results)

    # Print summary
    print_summary_table(results)

    # Generate individual plots
    print("\nGenerating figures...")

    plot_balanced_accuracy_strategies(
        results,
        os.path.join(args.output_dir, 'panel_a_balanced_accuracy.png')
    )

    plot_optimal_thresholds_balanced(
        results,
        os.path.join(args.output_dir, 'panel_b_optimal_thresholds.png')
    )

    plot_recall_strategies(
        results,
        os.path.join(args.output_dir, 'panel_c_recall_strategies.png')
    )

    # Generate combined figures
    create_closed_set_figure(results, args.output_dir)
    create_open_set_figure(results, args.output_dir)

    print(f"\nAnalysis complete! Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

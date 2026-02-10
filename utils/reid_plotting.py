"""
Shared plotting utilities for reid open-set hygiene sweep results.

Supports two modes:
  --model <name>    Per-backbone diagnostics (tables + diagnostic figures)
  --aggregate       Cross-backbone comparison (1x3 faceted figures)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple

from utils.results import ResultsCollection


# Settings for diagnostic functions (print_summary_table, plot_cosine_threshold).
# Note: The main figure uses hardcoded 'recall' criterion.
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

    Returns:
        Tuple of (best_epoch, best_mean_metric, details_dict)
    """
    if not all_histories or not all_histories[0]:
        return 50, 0.0, {}

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
    """Find optimal epoch using the configured default criterion."""
    if query_thresh is None:
        query_thresh = DEFAULT_QUALITY_THRESHOLD
    return find_best_epoch(all_histories, criterion=DEFAULT_BEST_EPOCH_CRITERION, query_thresh=query_thresh)


def collect_strategies_data(results: ResultsCollection) -> dict:
    """
    Collect metrics for all strategies (baseline, optimal R@1, optimal BA).

    Returns a dict with strategy data that can be used by multiple plotting functions.
    """
    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = ['q>=0.0', 'q>=0.1', 'q>=0.2', 'q>=0.3', 'q>=0.4', 'q>=0.5']

    strategies = {
        'baseline': {
            'label': 'None',
            'x': [], 'r1': [], 'r1_ci_lower': [], 'r1_ci_upper': [],
            'ba': [], 'ba_ci_lower': [], 'ba_ci_upper': []
        },
        'optimal': {
            'label': 'Optimal',
            'x': [], 'r1': [], 'r1_ci_lower': [], 'r1_ci_upper': [],
            'ba': [], 'ba_ci_lower': [], 'ba_ci_upper': [],
            'best_gal': [], 'best_q': []
        },
        'optimal_ba': {
            'label': 'Optimal',
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
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='recall', query_thresh='q>=0.0')

                r1_values = []
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            if 'query_quality_metrics' in h and 'q>=0.0' in h['query_quality_metrics']:
                                r1_values.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
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

        # --- Optimal: find best gallery x eval quality combo for R@1 ---
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
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='recall', query_thresh=q_thresh)

                r1_values = []
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            if 'query_quality_metrics' in h and q_thresh in h['query_quality_metrics']:
                                r1_values.append(h['query_quality_metrics'][q_thresh]['recall_at_1'])
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

        # --- Optimal BA: find best gallery x eval quality combo for BA ---
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
                best_epoch, _, _ = find_best_epoch(all_histories, criterion='ba', query_thresh=q_thresh)

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

    return strategies


def save_scores_table(strategies: dict, output_dir: str):
    """Save scores data as CSV tables."""
    # R@1 comparison table
    r1_data = []
    for i, x in enumerate(strategies['baseline']['x']):
        row = {'gallery_size': x}
        if i < len(strategies['baseline']['r1']):
            row['baseline_r1'] = strategies['baseline']['r1'][i]
            row['baseline_r1_ci_lower'] = strategies['baseline']['r1_ci_lower'][i]
            row['baseline_r1_ci_upper'] = strategies['baseline']['r1_ci_upper'][i]
        r1_data.append(row)

    for i, x in enumerate(strategies['optimal']['x']):
        match = next((r for r in r1_data if r['gallery_size'] == x), None)
        if match is None:
            match = {'gallery_size': x}
            r1_data.append(match)
        if i < len(strategies['optimal']['r1']):
            match['optimal_r1'] = strategies['optimal']['r1'][i]
            match['optimal_r1_ci_lower'] = strategies['optimal']['r1_ci_lower'][i]
            match['optimal_r1_ci_upper'] = strategies['optimal']['r1_ci_upper'][i]
            if i < len(strategies['optimal']['best_gal']):
                match['optimal_gal_thresh'] = strategies['optimal']['best_gal'][i]
            if i < len(strategies['optimal']['best_q']):
                match['optimal_query_thresh'] = strategies['optimal']['best_q'][i]

    r1_df = pd.DataFrame(r1_data).sort_values('gallery_size')
    if 'baseline_r1' in r1_df.columns and 'optimal_r1' in r1_df.columns:
        r1_df['delta_r1'] = r1_df['optimal_r1'] - r1_df['baseline_r1']

    r1_path = os.path.join(output_dir, 'scores_r1.csv')
    r1_df.to_csv(r1_path, index=False, float_format='%.4f')
    print(f"Saved R@1 scores table: {r1_path}")

    # Balanced Accuracy comparison table
    ba_data = []
    for i, x in enumerate(strategies['baseline']['x']):
        row = {'gallery_size': x}
        if i < len(strategies['baseline']['ba']):
            row['baseline_ba'] = strategies['baseline']['ba'][i]
            row['baseline_ba_ci_lower'] = strategies['baseline']['ba_ci_lower'][i]
            row['baseline_ba_ci_upper'] = strategies['baseline']['ba_ci_upper'][i]
        ba_data.append(row)

    for i, x in enumerate(strategies['optimal_ba']['x']):
        match = next((r for r in ba_data if r['gallery_size'] == x), None)
        if match is None:
            match = {'gallery_size': x}
            ba_data.append(match)
        if i < len(strategies['optimal_ba']['ba']):
            match['optimal_ba'] = strategies['optimal_ba']['ba'][i]
            match['optimal_ba_ci_lower'] = strategies['optimal_ba']['ba_ci_lower'][i]
            match['optimal_ba_ci_upper'] = strategies['optimal_ba']['ba_ci_upper'][i]
            if i < len(strategies['optimal_ba']['best_gal']):
                match['optimal_gal_thresh'] = strategies['optimal_ba']['best_gal'][i]
            if i < len(strategies['optimal_ba']['best_q']):
                match['optimal_query_thresh'] = strategies['optimal_ba']['best_q'][i]

    ba_df = pd.DataFrame(ba_data).sort_values('gallery_size')
    if 'baseline_ba' in ba_df.columns and 'optimal_ba' in ba_df.columns:
        ba_df['delta_ba'] = ba_df['optimal_ba'] - ba_df['baseline_ba']

    ba_path = os.path.join(output_dir, 'scores_ba.csv')
    ba_df.to_csv(ba_path, index=False, float_format='%.4f')
    print(f"Saved BA scores table: {ba_path}")


def save_thresholds_table(strategies: dict, output_dir: str):
    """Save optimal thresholds data as CSV tables."""
    r1_thresh_data = []
    for i, x in enumerate(strategies['optimal']['x']):
        row = {'gallery_size': x}
        if i < len(strategies['optimal']['best_gal']):
            row['gallery_threshold'] = strategies['optimal']['best_gal'][i]
        if i < len(strategies['optimal']['best_q']):
            row['query_threshold'] = strategies['optimal']['best_q'][i]
        r1_thresh_data.append(row)

    r1_df = pd.DataFrame(r1_thresh_data).sort_values('gallery_size')
    r1_path = os.path.join(output_dir, 'thresholds_r1_optimized.csv')
    r1_df.to_csv(r1_path, index=False, float_format='%.2f')
    print(f"Saved R@1-optimized thresholds table: {r1_path}")

    ba_thresh_data = []
    for i, x in enumerate(strategies['optimal_ba']['x']):
        row = {'gallery_size': x}
        if i < len(strategies['optimal_ba']['best_gal']):
            row['gallery_threshold'] = strategies['optimal_ba']['best_gal'][i]
        if i < len(strategies['optimal_ba']['best_q']):
            row['query_threshold'] = strategies['optimal_ba']['best_q'][i]
        ba_thresh_data.append(row)

    ba_df = pd.DataFrame(ba_thresh_data).sort_values('gallery_size')
    ba_path = os.path.join(output_dir, 'thresholds_ba_optimized.csv')
    ba_df.to_csv(ba_path, index=False, float_format='%.2f')
    print(f"Saved BA-optimized thresholds table: {ba_path}")


def save_summary_table(results: ResultsCollection, output_dir: str):
    """Save full summary statistics as CSV."""
    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    rows = []
    for threshold in gallery_thresholds:
        for gallery_size in gallery_sizes:
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)
            if len(filtered) == 0:
                continue

            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue

            best_epoch, criterion_value, details = find_optimal_epoch(all_histories)

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
                h_mean = 2 * mean_recall * mean_ba / (mean_recall + mean_ba) if (mean_recall + mean_ba) > 0 else 0.0
                recall_ci = stats.t.ppf(0.975, len(recall_values) - 1) * stats.sem(recall_values) if len(recall_values) > 1 else 0
                ba_ci = stats.t.ppf(0.975, len(ba_values) - 1) * stats.sem(ba_values) if len(ba_values) > 1 else 0

                rows.append({
                    'gallery_threshold': threshold,
                    'gallery_size': gallery_size,
                    'best_epoch': best_epoch,
                    'recall_at_1': mean_recall,
                    'recall_at_1_ci': recall_ci,
                    'balanced_accuracy': mean_ba,
                    'balanced_accuracy_ci': ba_ci,
                    'harmonic_mean': h_mean,
                    'n_seeds': len(recall_values)
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'summary_all_configs.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved full summary table: {csv_path}")


def _plot_scores_facet(ax, strategies, baseline_key, optimal_key, metric, colors, ylabel):
    """Draw one baseline-vs-optimal panel with lines and CI ribbons.

    Does NOT set y-limits — returns CI bounds so the caller can unify axes
    across multiple facets.

    Args:
        ax: Matplotlib axes
        strategies: Dict from collect_strategies_data()
        baseline_key: Strategy key for baseline (e.g. 'baseline')
        optimal_key: Strategy key for optimal (e.g. 'optimal' or 'optimal_ba')
        metric: Metric prefix in strategies dict ('r1' or 'ba')
        colors: Dict mapping strategy keys to colors
        ylabel: Y-axis label string

    Returns:
        (ci_lower_list, ci_upper_list) collected from both strategies
    """
    gallery_sizes = sorted(set(strategies['baseline']['x'] + strategies[optimal_key]['x']))

    for key in [baseline_key, optimal_key]:
        s = strategies[key]
        if s['x'] and s[metric]:
            ax.plot(s['x'], s[metric], color=colors[key], linewidth=2.5,
                    marker='o', markersize=8, label=s['label'])
            ax.fill_between(s['x'], s[f'{metric}_ci_lower'], s[f'{metric}_ci_upper'],
                            color=colors[key], alpha=0.2)

    ax.set_xlabel('Training examples per individual', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if gallery_sizes:
        ax.set_xticks(gallery_sizes)

    ax.grid(True, alpha=0.3, axis='y')

    all_ci_lower = strategies[baseline_key].get(f'{metric}_ci_lower', []) + strategies[optimal_key].get(f'{metric}_ci_lower', [])
    all_ci_upper = strategies[baseline_key].get(f'{metric}_ci_upper', []) + strategies[optimal_key].get(f'{metric}_ci_upper', [])
    return all_ci_lower, all_ci_upper


def _apply_shared_ylim(axes, all_ci_lower, all_ci_upper):
    """Apply a unified y-axis scale across all axes based on global CI bounds.

    Args:
        axes: List of Matplotlib axes to unify
        all_ci_lower: Flat list of all CI lower bounds across every facet
        all_ci_upper: Flat list of all CI upper bounds across every facet
    """
    if not all_ci_lower or not all_ci_upper:
        return
    y_min = max(0, np.floor((min(all_ci_lower) - 0.05) / 0.05) * 0.05)
    y_max = min(1, np.ceil((max(all_ci_upper) + 0.05) / 0.05) * 0.05)
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))


def _plot_thresholds_facet(ax, strategies, strategy_key):
    """Draw one bar-chart panel showing gallery vs query thresholds.

    Does NOT set y-limits — returns the max bar value so the caller can
    unify axes across multiple facets.

    Args:
        ax: Matplotlib axes
        strategies: Dict from collect_strategies_data()
        strategy_key: 'optimal' (R@1-optimized) or 'optimal_ba' (BA-optimized)

    Returns:
        Maximum bar value across both series, or 0.0 if no data
    """
    s = strategies[strategy_key]
    x = np.arange(len(s['x']))
    width = 0.35
    max_val = 0.0

    if s['best_gal'] and s['best_q']:
        ax.bar(x - width/2, s['best_gal'], width,
               color='#1f77b4', label='Training Gallery')
        ax.bar(x + width/2, s['best_q'], width,
               color='#ff7f0e', label='Validation Queries')
        ax.set_xticks(x)
        ax.set_xticklabels(s['x'])
        max_val = max(max(s['best_gal']), max(s['best_q']))

    ax.set_xlabel('Training examples per individual', fontsize=12)
    ax.set_ylabel(r'Best image quality threshold ($p$ visible pelage)', fontsize=12)
    ax.legend(title='Threshold applied to:', loc='lower right', fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    return max_val


def create_rank1_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for Recall@1.

    Args:
        model_data: Dict mapping model name -> {"strategies", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    colors = {'baseline': '#1f77b4', 'optimal': '#2ca02c'}
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_ci_lower = []
    global_ci_upper = []

    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies']

        ci_lo, ci_hi = _plot_scores_facet(ax, strategies, 'baseline', 'optimal', 'r1', colors,
                                          'Wolverine re-identification score (Recall at rank 1)')
        global_ci_lower.extend(ci_lo)
        global_ci_upper.extend(ci_hi)
        label = md['label'].replace('Frozen ', '')
        ax.legend(title="$\\bf{Model:}$" + f"\n{label}\n\n" + "$\\bf{Image\\ quality\\ filters:}$",
                  loc='lower right', fontsize=10, title_fontsize=10)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    _apply_shared_ylim([axes[0, i] for i in range(len(models))], global_ci_lower, global_ci_upper)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rank@1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone R@1 figure: {output_path}")


def create_novelty_detection_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for Balanced Accuracy (novelty detection).

    Args:
        model_data: Dict mapping model name -> {"strategies", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    colors = {'baseline': '#1f77b4', 'optimal_ba': '#2ca02c'}
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_ci_lower = []
    global_ci_upper = []

    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies']

        ci_lo, ci_hi = _plot_scores_facet(ax, strategies, 'baseline', 'optimal_ba', 'ba', colors,
                                          'Novel wolverine detection score (Balanced accuracy)')
        global_ci_lower.extend(ci_lo)
        global_ci_upper.extend(ci_hi)
        label = md['label'].replace('Frozen ', '')
        ax.legend(title="$\\bf{Model:}$" + f"\n{label}\n\n" + "$\\bf{Image\\ quality\\ filters:}$",
                  loc='lower right', fontsize=10, title_fontsize=10)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    _apply_shared_ylim([axes[0, i] for i in range(len(models))], global_ci_lower, global_ci_upper)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'novelty_detection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone novelty detection figure: {output_path}")


def create_rank1_thresholds_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for R@1-optimized thresholds.

    Args:
        model_data: Dict mapping model name -> {"strategies", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_max = 0.0
    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies']

        facet_max = _plot_thresholds_facet(ax, strategies, 'optimal')
        global_max = max(global_max, facet_max)
        ax.set_title(md['label'].replace('Frozen ', ''), fontsize=12)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    # Shared y-axis: 0 to next 0.1 step above the tallest bar
    if global_max > 0:
        shared_ymax = np.ceil(global_max * 10) / 10 + 0.05
        for i in range(len(models)):
            axes[0, i].set_ylim(0, shared_ymax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'best_threshold_rank.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone R@1 thresholds figure: {output_path}")


def create_novelty_thresholds_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for BA-optimized thresholds.

    Args:
        model_data: Dict mapping model name -> {"strategies", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_max = 0.0
    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies']

        facet_max = _plot_thresholds_facet(ax, strategies, 'optimal_ba')
        global_max = max(global_max, facet_max)
        ax.set_title(md['label'].replace('Frozen ', ''), fontsize=12)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    # Shared y-axis: 0 to next 0.1 step above the tallest bar
    if global_max > 0:
        shared_ymax = np.ceil(global_max * 10) / 10 + 0.05
        for i in range(len(models)):
            axes[0, i].set_ylim(0, shared_ymax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'best_threshold_novelty.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone BA thresholds figure: {output_path}")


def plot_recall_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing recall@1 curves over epochs."""
    from matplotlib.lines import Line2D

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    thresholds = sorted(results.get_unique('threshold'))
    q_key = "q>=0.3"

    n_rows, n_cols = len(gallery_sizes), len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15), sharex=True, sharey=True)

    seed_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))

    for i, gallery_size in enumerate(gallery_sizes):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)

            for r in filtered:
                seed = r['seed']
                history = r.get('epoch_history', [])
                epochs = [h['epoch'] for h in history]
                recall_values = [h['query_quality_metrics'][q_key]['recall_at_1'] for h in history]
                ax.plot(epochs, recall_values, color=seed_colors[seed], alpha=0.8, linewidth=1)

            if i == 0:
                ax.set_title(f'thresh={threshold:.1f}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'gallery={gallery_size}', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    legend_elements = [
        Line2D([0], [0], color=seed_colors[s], label=f'seed {s}', linewidth=2)
        for s in range(8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, title='Seed')
    fig.supxlabel('Epoch', fontsize=12)
    fig.supylabel('Recall@1 (val)', fontsize=12)
    fig.suptitle('Validation Recall@1 (q>=0.3) by Configuration (Raw Cosine)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_validation_loss_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing validation loss curves over epochs."""
    from matplotlib.lines import Line2D

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    thresholds = sorted(results.get_unique('threshold'))

    n_rows, n_cols = len(gallery_sizes), len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15), sharex=True, sharey=False)

    seed_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))

    for i, gallery_size in enumerate(gallery_sizes):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)

            for r in filtered:
                seed = r['seed']
                history = r.get('epoch_history', [])
                if not history or 'val_loss' not in history[0]:
                    continue
                epochs = [h['epoch'] for h in history]
                val_losses = [h['val_loss'] for h in history]
                ax.plot(epochs, val_losses, color=seed_colors[seed], alpha=0.8, linewidth=1)

            if i == 0:
                ax.set_title(f'thresh={threshold:.1f}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'gallery={gallery_size}', fontsize=9)
            ax.grid(True, alpha=0.3)

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
    """Plot cosine similarity decision threshold by gallery size and quality threshold."""
    fig, ax = plt.subplots(figsize=(10, 8))

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
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
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Gallery quality filter', loc='best', fontsize=10, title_fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def print_summary_table(results: ResultsCollection):
    """Print summary statistics for open-set evaluation."""
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

            best_epoch, criterion_value, details = find_optimal_epoch(all_histories)

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
                if (mean_recall + mean_ba) > 0:
                    h_mean = 2 * mean_recall * mean_ba / (mean_recall + mean_ba)
                else:
                    h_mean = 0.0

                print(f"{threshold:<8.2f} {gallery_size:<8} {best_epoch:<6} "
                      f"{mean_recall:<8.4f} {mean_ba:<8.4f} {h_mean:<8.4f} {len(recall_values):<6}")
                metrics[(threshold, gallery_size)] = {
                    'recall': mean_recall, 'ba': mean_ba, 'h_mean': h_mean,
                    'epoch': best_epoch, 'n_seeds': len(recall_values)
                }

    if metrics:
        best_key = max(metrics.keys(), key=lambda k: metrics[k]['h_mean'])
        best_data = metrics[best_key]
        print("-" * 100)
        print(f"Best config: threshold={best_key[0]}, gallery={best_key[1]}, epoch={best_data['epoch']}")
        print(f"  R@1={best_data['recall']:.4f}, BA={best_data['ba']:.4f}, H-Mean={best_data['h_mean']:.4f}")

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


def run_single_model(model_name: str, results_dir: str, output_dir: str):
    """Per-backbone diagnostics: tables and diagnostic figures.

    Args:
        model_name: Key in MODEL_CONFIGS (e.g. 'dinov3')
        results_dir: Path to results JSON files
        output_dir: Base output dir (figures go to output_dir, tables to sibling tables dir)
    """
    print("=" * 60)
    print(f"Open-Set Evaluation: {model_name} Results Analysis (Raw Cosine)")
    print("=" * 60)

    results = load_hygiene_results(results_dir)

    if len(results) == 0:
        print(f"No results found for {model_name}. Skipping.")
        return None

    print_summary_table(results)

    # Tables -> reid_openset/tables/{model_name}/
    tables_dir = os.path.join('reid_openset', 'tables', model_name)
    os.makedirs(tables_dir, exist_ok=True)

    strategies = collect_strategies_data(results)

    save_scores_table(strategies, tables_dir)
    save_thresholds_table(strategies, tables_dir)
    save_summary_table(results, tables_dir)

    # Diagnostic figures -> reid_openset/figures/{model_name}/
    figures_dir = os.path.join('reid_openset', 'figures', model_name)
    os.makedirs(figures_dir, exist_ok=True)

    plot_recall_curves(results, os.path.join(figures_dir, 'recall_curves.png'))
    plot_validation_loss_curves(results, os.path.join(figures_dir, 'validation_loss_curves.png'))
    plot_cosine_threshold(results, os.path.join(figures_dir, 'cosine_threshold.png'))

    print(f"\nDiagnostics complete for {model_name}!")
    print(f"  Tables: {tables_dir}")
    print(f"  Figures: {figures_dir}")

    return {'strategies': strategies, 'results': results}


def run_aggregate(base_dir: str = "reid_openset"):
    """Cross-backbone aggregate analysis.

    Iterates MODEL_CONFIGS, loads each backbone's results, produces per-backbone
    diagnostics and cross-backbone comparison figures.
    """
    from utils.reid import MODEL_CONFIGS

    print("=" * 60)
    print("Cross-Backbone Aggregate Analysis")
    print("=" * 60)

    model_data = {}

    for model_name, config in MODEL_CONFIGS.items():
        results_dir = f"{config['experiment_dir']}/results"
        print(f"\n--- Loading {model_name} from {results_dir} ---")

        result = run_single_model(model_name, results_dir, base_dir)
        if result is None:
            print(f"Skipping {model_name} (no results)")
            continue

        model_data[model_name] = {
            'strategies': result['strategies'],
            'results': result['results'],
            'label': config['backbone_label'],
        }

    if not model_data:
        print("No backbone results found. Exiting.")
        return

    # Cross-backbone figures -> reid_openset/figures/
    cross_figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(cross_figures_dir, exist_ok=True)

    print(f"\nGenerating cross-backbone figures ({len(model_data)} backbones)...")
    create_rank1_figure(model_data, cross_figures_dir)
    create_novelty_detection_figure(model_data, cross_figures_dir)
    create_rank1_thresholds_figure(model_data, cross_figures_dir)
    create_novelty_thresholds_figure(model_data, cross_figures_dir)

    print(f"\nAggregate analysis complete!")
    print(f"  Cross-backbone figures: {cross_figures_dir}")


if __name__ == "__main__":
    import argparse
    from utils.reid import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description="Plot reid open-set hygiene sweep results")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()),
                        help="Single backbone mode")
    parser.add_argument("--aggregate", action="store_true",
                        help="Run cross-backbone aggregate analysis")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.aggregate:
        run_aggregate()
    elif args.model:
        config = MODEL_CONFIGS[args.model]
        results_dir = args.results_dir or f"{config['experiment_dir']}/results"
        output_dir = args.output_dir or "reid_openset"
        run_single_model(args.model, results_dir, output_dir)
    else:
        parser.error("Either --model or --aggregate is required")

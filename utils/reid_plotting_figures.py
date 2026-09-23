"""
Figure generation functions for reid plotting: scores facets, recall curves, loss curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils.results import ResultsCollection


def _plot_scores_facet(ax, strategies, strategy_keys, metric, colors, ylabel):
    """Draw multi-strategy panel with lines and CI ribbons.

    Does NOT set y-limits — returns CI bounds so the caller can unify axes
    across multiple facets.

    Args:
        ax: Matplotlib axes
        strategies: Dict from collect_strategies_data()
        strategy_keys: Ordered list of strategy keys to plot
        metric: Metric prefix in strategies dict ('r1' or 'ba')
        colors: Dict mapping strategy keys to colors
        ylabel: Y-axis label string

    Returns:
        (ci_lower_list, ci_upper_list) collected from all strategies
    """
    all_x = set()
    for key in strategy_keys:
        all_x.update(strategies[key]['x'])
    gallery_sizes = sorted(all_x)

    for key in strategy_keys:
        s = strategies[key]
        if s['x'] and s[metric]:
            ax.plot(s['x'], s[metric], color=colors[key], linewidth=2.5,
                    marker='o', markersize=8, label=s['label'])
            ax.fill_between(s['x'], s[f'{metric}_ci_lower'], s[f'{metric}_ci_upper'],
                            color=colors[key], alpha=0.2)

    ax.set_xlabel('Gallery images per individual', fontsize=17.3)
    ax.set_ylabel(ylabel, fontsize=17.3)
    if gallery_sizes:
        ax.set_xticks(gallery_sizes)

    ax.grid(True, alpha=0.3, axis='y')

    all_ci_lower = []
    all_ci_upper = []
    for key in strategy_keys:
        all_ci_lower.extend(strategies[key].get(f'{metric}_ci_lower', []))
        all_ci_upper.extend(strategies[key].get(f'{metric}_ci_upper', []))
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


def create_rank1_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for Recall@1.

    Args:
        model_data: Dict mapping model name -> {"strategies_r1", "strategies_ba", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    strategy_keys = ['none', 'gallery', 'query', 'gallery_query']
    colors = {'none': '#888888', 'gallery': '#1f77b4', 'query': '#ff7f0e', 'gallery_query': '#2ca02c'}
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_ci_lower = []
    global_ci_upper = []

    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies_r1']

        ylabel = 'Recall at rank one' if i == 0 else ''
        ci_lo, ci_hi = _plot_scores_facet(ax, strategies, strategy_keys, 'r1', colors, ylabel)
        global_ci_lower.extend(ci_lo)
        global_ci_upper.extend(ci_hi)
        label = md['label'].replace('Frozen ', '')
        ax.set_title(label, fontsize=17.3, fontweight='bold', pad=10)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=23.0, fontweight='bold', va='top', ha='left')
        ax.tick_params(labelsize=13.0)
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    _apply_shared_ylim([axes[0, i] for i in range(len(models))], global_ci_lower, global_ci_upper)

    # Single shared legend from the first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=13.0, ncol=len(strategy_keys),
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
               title="$\\bf{Image\\ quality\\ filters}$", title_fontsize=13.0)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fewshot_rank@1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone R@1 figure: {output_path}")


def create_novelty_detection_figure(model_data: dict, output_dir: str):
    """Create 1x3 cross-backbone figure for Balanced Accuracy (novelty detection).

    Args:
        model_data: Dict mapping model name -> {"strategies_r1", "strategies_ba", "results", "label"}
        output_dir: Directory to save figure
    """
    panel_labels = ['A', 'B', 'C']
    strategy_keys = ['none', 'gallery', 'query', 'gallery_query']
    colors = {'none': '#888888', 'gallery': '#1f77b4', 'query': '#ff7f0e', 'gallery_query': '#2ca02c'}
    models = list(model_data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)

    global_ci_lower = []
    global_ci_upper = []

    for i, model_name in enumerate(models):
        ax = axes[0, i]
        md = model_data[model_name]
        strategies = md['strategies_ba']

        ylabel = 'Balanced accuracy' if i == 0 else ''
        ci_lo, ci_hi = _plot_scores_facet(ax, strategies, strategy_keys, 'ba', colors, ylabel)
        global_ci_lower.extend(ci_lo)
        global_ci_upper.extend(ci_hi)
        label = md['label'].replace('Frozen ', '')
        ax.set_title(label, fontsize=17.3, fontweight='bold', pad=10)
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=23.0, fontweight='bold', va='top', ha='left')
        ax.tick_params(labelsize=13.0)
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    _apply_shared_ylim([axes[0, i] for i in range(len(models))], global_ci_lower, global_ci_upper)

    # Single shared legend from the first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=13.0, ncol=len(strategy_keys),
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
               title="$\\bf{Image\\ quality\\ filters}$", title_fontsize=13.0)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fewshot_novelty_detection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-backbone novelty detection figure: {output_path}")


def plot_recall_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing recall@1 curves over steps."""
    from matplotlib.lines import Line2D

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    thresholds = sorted(results.get_unique('threshold'))
    # Pick middle query threshold from result data
    q_key = "q>=0.0"
    for r in results:
        history = r.get('step_history', [])
        if history and 'query_quality_metrics' in history[0]:
            available = sorted(history[0]['query_quality_metrics'].keys())
            q_key = available[len(available) // 2]
            break

    n_rows, n_cols = len(gallery_sizes), len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15), sharex=True, sharey=True)

    seed_colors = plt.cm.tab10(np.linspace(0, 0.8, 8))

    for i, gallery_size in enumerate(gallery_sizes):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            filtered = results.filter(threshold=threshold, gallery_size=gallery_size)

            for r in filtered:
                seed = r['seed']
                history = r.get('step_history', [])
                steps = [h['step'] for h in history]
                recall_values = [h['query_quality_metrics'][q_key]['recall_at_1'] for h in history]
                ax.plot(steps, recall_values, color=seed_colors[seed], alpha=0.8, linewidth=1)

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
    fig.supxlabel('Step', fontsize=12)
    fig.supylabel('Recall@1 (val)', fontsize=12)
    fig.suptitle(f'Validation Recall@1 ({q_key}) by Configuration (Raw Cosine)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_validation_loss_curves(results: ResultsCollection, output_path: str):
    """Create faceted figure showing validation loss curves over steps."""
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
                history = r.get('step_history', [])
                if not history or 'val_loss' not in history[0]:
                    continue
                steps = [h['step'] for h in history]
                val_losses = [h['val_loss'] for h in history]
                ax.plot(steps, val_losses, color=seed_colors[seed], alpha=0.8, linewidth=1)

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
    fig.supxlabel('Step', fontsize=12)
    fig.supylabel('Validation Loss (ArcFace)', fontsize=12)
    fig.suptitle('Validation Loss by Configuration (Raw Cosine)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_cosine_threshold(results: ResultsCollection, output_path: str):
    """Plot cosine similarity decision threshold by gallery size and quality threshold."""
    from utils.reid_plotting import find_optimal_step

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

            all_histories = [r.get('step_history', []) for r in filtered]
            best_step, _, _ = find_optimal_step(all_histories, 'q>=0.0')

            thresh_values = []
            for history in all_histories:
                for h in history:
                    if h['step'] == best_step and 'test_open_set' in h:
                        open_set = h['test_open_set']
                        ct = open_set['threshold_calibration']['global_threshold']
                        thresh_values.append(ct)
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

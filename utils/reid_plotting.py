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
from scipy import stats
from typing import List, Tuple

from utils.results import ResultsCollection

# Re-export from submodules for backward compatibility
from utils.reid_plotting_tables import (  # noqa: F401
    save_scores_table, save_thresholds_table, load_test_results,
    create_thresholds_table, create_test_performance_table,
    create_rank1_thresholds_table, create_novelty_thresholds_table,
)
from utils.reid_plotting_figures import (  # noqa: F401
    _plot_scores_facet, _apply_shared_ylim,
    create_rank1_figure, create_novelty_detection_figure,
    plot_recall_curves, plot_validation_loss_curves, plot_cosine_threshold,
)


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
    # Discover query thresholds from result data
    query_thresholds = ['q>=0.0']
    for r in results:
        history = r.get('epoch_history', [])
        if history and 'query_quality_metrics' in history[0]:
            query_thresholds = sorted(history[0]['query_quality_metrics'].keys())
            break

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
                # R@1: select epoch that maximizes recall
                best_r1_epoch, _, _ = find_best_epoch(all_histories, criterion='recall', query_thresh='q>=0.0')
                r1_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_r1_epoch:
                            if 'query_quality_metrics' in h and 'q>=0.0' in h['query_quality_metrics']:
                                r1_values.append(h['query_quality_metrics']['q>=0.0']['recall_at_1'])
                            break

                # BA: select epoch that maximizes balanced accuracy
                best_ba_epoch, _, _ = find_best_epoch(all_histories, criterion='ba', query_thresh='q>=0.0')
                ba_values = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_ba_epoch:
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

        # --- Optimal BA: find best gallery threshold for BA ---
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


def print_summary_table(results: ResultsCollection):
    """Print summary statistics for open-set evaluation."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Best Epoch by R@1 and BA (independent)")
    print("=" * 100)

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))

    # Discover available query thresholds from data
    query_thresholds = [DEFAULT_QUALITY_THRESHOLD]
    for r in results:
        history = r.get('epoch_history', [])
        if history and 'query_quality_metrics' in history[0]:
            query_thresholds = sorted(history[0]['query_quality_metrics'].keys())
            break

    print(f"{'Thresh':<8} {'Gallery':<8} "
          f"{'R@1_epoch':<10} {'R@1_mean':<10} {'R@1_query_t':<12} "
          f"{'BA_epoch':<9} {'BA_mean':<9} {'BA_query_t':<11} {'Seeds':<6}")
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

            # Find best R@1 across query thresholds
            best_r1_epoch, best_r1_mean, best_r1_qt = None, -1, query_thresholds[0]
            for qt in query_thresholds:
                epoch, mean_val, _ = find_best_epoch(all_histories, criterion='recall', query_thresh=qt)
                if mean_val is not None and mean_val > best_r1_mean:
                    best_r1_epoch, best_r1_mean, best_r1_qt = epoch, mean_val, qt

            # Find best BA across query thresholds
            best_ba_epoch, best_ba_mean, best_ba_qt = None, -1, query_thresholds[0]
            for qt in query_thresholds:
                epoch, mean_val, _ = find_best_epoch(all_histories, criterion='ba', query_thresh=qt)
                if mean_val is not None and mean_val > best_ba_mean:
                    best_ba_epoch, best_ba_mean, best_ba_qt = epoch, mean_val, qt

            n_seeds = len(all_histories)

            print(f"{threshold:<8.2f} {gallery_size:<8} "
                  f"{best_r1_epoch:<10} {best_r1_mean:<10.4f} {best_r1_qt:<12} "
                  f"{best_ba_epoch:<9} {best_ba_mean:<9.4f} {best_ba_qt:<11} {n_seeds:<6}")
            metrics[(threshold, gallery_size)] = {
                'r1_epoch': best_r1_epoch, 'r1_mean': best_r1_mean, 'r1_query_t': best_r1_qt,
                'ba_epoch': best_ba_epoch, 'ba_mean': best_ba_mean, 'ba_query_t': best_ba_qt,
                'n_seeds': n_seeds
            }

    if metrics:
        best_r1_key = max(metrics.keys(), key=lambda k: metrics[k]['r1_mean'])
        best_ba_key = max(metrics.keys(), key=lambda k: metrics[k]['ba_mean'])
        print("-" * 100)
        d = metrics[best_r1_key]
        print(f"Best R@1: thresh={best_r1_key[0]}, gallery={best_r1_key[1]}, "
              f"epoch={d['r1_epoch']}, R@1={d['r1_mean']:.4f}, query_t={d['r1_query_t']}")
        d = metrics[best_ba_key]
        print(f"Best BA:  thresh={best_ba_key[0]}, gallery={best_ba_key[1]}, "
              f"epoch={d['ba_epoch']}, BA={d['ba_mean']:.4f}, query_t={d['ba_query_t']}")


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
    from utils.reid_config import MODEL_CONFIGS

    print("=" * 60)
    print("Cross-Backbone Aggregate Analysis")
    print("=" * 60)

    model_data = {}

    for model_name, config in MODEL_CONFIGS.items():
        results_dir = f"{config['experiment_dir']}/results/hygiene"
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

    # Cross-backbone tables -> reid_openset/tables/
    cross_tables_dir = os.path.join(base_dir, 'tables')
    os.makedirs(cross_tables_dir, exist_ok=True)

    print(f"\nGenerating cross-backbone figures ({len(model_data)} backbones)...")
    create_rank1_figure(model_data, cross_figures_dir)
    create_novelty_detection_figure(model_data, cross_figures_dir)
    create_rank1_thresholds_table(model_data, cross_tables_dir)
    create_novelty_thresholds_table(model_data, cross_tables_dir)

    # Test performance tables (skip gracefully if no test results exist)
    print(f"\nGenerating test performance tables...")
    create_test_performance_table(model_data, cross_tables_dir, 'closedset')
    create_test_performance_table(model_data, cross_tables_dir, 'openset')

    print(f"\nAggregate analysis complete!")
    print(f"  Cross-backbone figures: {cross_figures_dir}")
    print(f"  Cross-backbone tables: {cross_tables_dir}")


if __name__ == "__main__":
    import argparse
    from utils.reid_config import MODEL_CONFIGS

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
        results_dir = args.results_dir or f"{config['experiment_dir']}/results/hygiene"
        output_dir = args.output_dir or "reid_openset"
        run_single_model(args.model, results_dir, output_dir)
    else:
        parser.error("Either --model or --aggregate is required")

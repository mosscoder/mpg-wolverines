#!/usr/bin/env python
"""Aggregate reid results into tidy tables.

Produces:
  reid_openset/summary/fewshot/recall_at_1.csv + .md         Full factorial few-shot results
  reid_openset/summary/fewshot/balanced_accuracy.csv + .md
  reid_openset/summary/fewshot/hyperparameters.csv + .md      Per-backbone hyperparameters
  reid_openset/summary/test_eval/recall_at_1.csv + .md        Test eval results
  reid_openset/summary/test_eval/balanced_accuracy.csv + .md
  reid_openset/summary/test_eval/*.png                        Test eval figures
"""

import json
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
import numpy as np
from scipy import stats


class _TitleHandler(HandlerBase):
    """Legend handler that draws an invisible handle, so text aligns with patch icons."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        patch = mpatches.FancyBboxPatch((0, 0), 0, 0, visible=False,
                                        transform=trans)
        return [patch]

def write_table(rows, cols, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{name}.csv')
    md_path = os.path.join(out_dir, f'{name}.md')
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    with open(md_path, 'w') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('| ' + ' | '.join('---' for _ in cols) + ' |\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(f'  {name}: {len(rows)} rows -> {csv_path}, {md_path}')
    with open(md_path) as f:
        print(f.read())


def _strip_frozen(name):
    """Remove 'Frozen ' prefix from backbone name for display."""
    return name.replace('Frozen ', '')


def classify_backbone(name):
    """Map full backbone string to short label via substring matching."""
    if 'DINOv3' in name:
        return 'DINOv3'
    elif 'MegaDescriptor' in name:
        return 'MegaDescriptor'
    elif 'BioCLIP' in name:
        return 'BioCLIP-2'
    return name


def _fmt_ci(values):
    """Format mean ± 95% CI as string."""
    mean = np.mean(values)
    if len(values) > 1:
        ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    else:
        ci = 0.0
    return f"{mean:.4f} (\u00b1{ci:.4f})"


def create_fewshot_tables(model_data, out_dir):
    """Generate full-factorial few-shot tables from hygiene sweep results."""
    from utils.reid_plotting import find_best_step, get_test_recall, get_test_ba

    r1_rows = []
    ba_rows = []

    for model_name, mdata in model_data.items():
        results = mdata['results']
        label = mdata['label']

        gallery_sizes = sorted(results.get_unique('gallery_size'))
        gallery_thresholds = sorted(results.get_unique('threshold'))

        # Discover query thresholds from data
        query_thresholds = ['q>=0.0']
        for r in results:
            history = r.get('step_history', [])
            if history and 'query_quality_metrics' in history[0]:
                query_thresholds = sorted(history[0]['query_quality_metrics'].keys())
                break

        for gsize in gallery_sizes:
            for gal_thresh in gallery_thresholds:
                filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
                if len(filtered) == 0:
                    continue
                all_histories = [r.get('step_history', []) for r in filtered]
                if not all_histories or not all_histories[0]:
                    continue
                if 'query_quality_metrics' not in all_histories[0][0]:
                    continue

                for q_key in query_thresholds:
                    best_step, _, _ = find_best_step(all_histories, query_thresh=q_key)

                    r1_values = []
                    ba_values = []
                    for history in all_histories:
                        for h in history:
                            if h['step'] == best_step:
                                tr = get_test_recall(h, q_key)
                                if tr is not None:
                                    r1_values.append(tr)
                                tb = get_test_ba(h, q_key)
                                if tb is not None:
                                    ba_values.append(tb)
                                break

                    n_seeds = len(all_histories)

                    if r1_values:
                        r1_rows.append({
                            'backbone': label,
                            'gallery_size': gsize,
                            'gallery_q': f"{gal_thresh:.2f}",
                            'query_q': q_key.replace('q>=', ''),
                            'R@1 (95% CI)': _fmt_ci(r1_values),
                            'best_step': best_step,
                            'seeds': n_seeds,
                        })

                    if ba_values:
                        ba_rows.append({
                            'backbone': label,
                            'gallery_size': gsize,
                            'gallery_q': f"{gal_thresh:.2f}",
                            'query_q': q_key.replace('q>=', ''),
                            'BA (95% CI)': _fmt_ci(ba_values),
                            'best_step': best_step,
                            'seeds': n_seeds,
                        })

    r1_cols = ['backbone', 'gallery_size', 'gallery_q', 'query_q',
               'R@1 (95% CI)', 'best_step', 'seeds']
    ba_cols = ['backbone', 'gallery_size', 'gallery_q', 'query_q',
               'BA (95% CI)', 'best_step', 'seeds']

    print('=== Few-shot Recall@1 ===')
    write_table(r1_rows, r1_cols, 'recall_at_1', out_dir)
    print('=== Few-shot Balanced Accuracy ===')
    write_table(ba_rows, ba_cols, 'balanced_accuracy', out_dir)


def create_hyperparameters_table(out_dir):
    """Generate per-backbone hyperparameters table."""
    from utils.reid_config import MODEL_CONFIGS, ARCFACE_MARGIN, ARCFACE_SCALE, TRAIN_BATCH_SIZE
    from utils.reid import load_best_hyperparams

    rows = []
    for model_name, config in MODEL_CONFIGS.items():
        best_lr, best_size, best_emb_dim = load_best_hyperparams(model_name)
        rows.append({
            'backbone': config['backbone_label'],
            'lr': best_lr,
            'emb_dim': best_emb_dim,
            'image_size': best_size,
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'batch_size': TRAIN_BATCH_SIZE,
        })

    cols = ['backbone', 'lr', 'emb_dim', 'image_size',
            'arcface_margin', 'arcface_scale', 'batch_size']
    print('=== Hyperparameters ===')
    write_table(rows, cols, 'hyperparameters', out_dir)


def create_test_eval_tables(all_data, out_dir):
    """Aggregate test_eval JSONs into recall_at_1 and balanced_accuracy tables."""
    r1_cols = ['backbone', 'gallery_q', 'query_q', 'R@1', 'cv_R@1',
               'lr', 'emb_dim', 'steps']
    r1_rows = []
    for data in all_data:
        cfg = data['config']
        res = data['results']
        r1_rows.append({
            'backbone':  cfg['backbone'],
            'gallery_q': f"{cfg['gallery_threshold']:.2f}",
            'query_q':   f"{cfg['query_threshold']:.2f}",
            'R@1':       f"{res['recall_at_1']:.4f}",
            'cv_R@1':    f"{cfg.get('cv_recall_at_1', 0):.4f}",
            'lr':        cfg['learning_rate'],
            'emb_dim':   cfg['embedding_dim'],
            'steps':     cfg['best_step'],
        })

    r1_rows.sort(key=lambda r: (r['backbone'], r['gallery_q'], r['query_q']))
    print('=== Test Eval Recall@1 ===')
    write_table(r1_rows, r1_cols, 'recall_at_1', out_dir)

    ba_cols = ['backbone', 'gallery_q', 'query_q', 'BA', 'KAR', 'URR',
               'lr', 'emb_dim', 'steps']
    ba_rows = []
    for data in all_data:
        cfg = data['config']
        res = data['results']
        ba_rows.append({
            'backbone':  cfg['backbone'],
            'gallery_q': f"{cfg['gallery_threshold']:.2f}",
            'query_q':   f"{cfg['query_threshold']:.2f}",
            'BA':        f"{res['balanced_accuracy']:.4f}",
            'KAR':       f"{res['known_accept_rate']:.4f}",
            'URR':       f"{res['unknown_reject_rate']:.4f}",
            'lr':        cfg['learning_rate'],
            'emb_dim':   cfg['embedding_dim'],
            'steps':     cfg['best_step'],
        })

    ba_rows.sort(key=lambda r: (r['backbone'], r['gallery_q'], r['query_q']))
    print('=== Test Eval Balanced Accuracy ===')
    write_table(ba_rows, ba_cols, 'balanced_accuracy', out_dir)

    print(f'\nTotal: {len(r1_rows)} R@1 rows + {len(ba_rows)} BA rows '
          f'from {len(all_data)} JSONs')
    return all_data


def create_test_eval_figures(all_data, out_dir):
    """Create grouped bar charts for Recall@1 and Balanced Accuracy.

    X-axis: query threshold, bars grouped by backbone with gallery threshold
    controlling saturation (higher gallery threshold = darker fill).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build records, keeping full backbone name for display
    records = []
    backbone_full_names = {}
    for data in all_data:
        cfg = data['config']
        res = data['results']
        short = classify_backbone(cfg['backbone'])
        backbone_full_names[short] = _strip_frozen(cfg['backbone'])
        records.append({
            'backbone': short,
            'gallery_q': cfg['gallery_threshold'],
            'query_q': cfg['query_threshold'],
            'recall_at_1': res['recall_at_1'],
            'balanced_accuracy': res['balanced_accuracy'],
        })

    if not records:
        print('No test_eval data for figures.')
        return

    # Ordered categories
    backbones = ['DINOv3', 'MegaDescriptor', 'BioCLIP-2']
    gallery_thresholds = sorted(set(r['gallery_q'] for r in records))
    query_thresholds = sorted(set(r['query_q'] for r in records))

    # Colors: backbone -> base hue, gallery threshold -> alpha (saturation)
    backbone_hues = {
        'DINOv3': '#1f77b4',
        'MegaDescriptor': '#ff7f0e',
        'BioCLIP-2': '#2ca02c',
    }
    gallery_alphas = {gt: alpha for gt, alpha in
                      zip(gallery_thresholds, [0.3, 0.6, 1.0])}

    # Build lookup: (backbone, gallery_q, query_q) -> record
    lookup = {}
    for r in records:
        lookup[(r['backbone'], r['gallery_q'], r['query_q'])] = r

    # Bar layout parameters
    bar_width = 0.08
    inter_gap = 0.04      # gap between backbone groups
    n_gallery = len(gallery_thresholds)
    n_backbones = len(backbones)
    group_w = n_gallery * bar_width
    tick_w = n_backbones * group_w + (n_backbones - 1) * inter_gap

    metrics = [
        ('recall_at_1',
         'Wolverine re-identification score\n(Recall@1 - Test)',
         'test_rank@1.png', (0.5, 0.95)),
        ('balanced_accuracy',
         'Novel wolverine detection score\n(Balanced Accuracy - Test)',
         'test_novelty_detection.png', (0.4, 0.85)),
    ]

    for metric_key, metric_label, filename, ylim in metrics:
        fig, ax = plt.subplots(figsize=(7, 3))

        for qi, qt in enumerate(query_thresholds):
            tick_center = qi
            tick_left = tick_center - tick_w / 2

            for bi, backbone in enumerate(backbones):
                group_left = tick_left + bi * (group_w + inter_gap)
                base_color = backbone_hues[backbone]

                for gi, gal in enumerate(gallery_thresholds):
                    x = group_left + gi * bar_width + bar_width / 2
                    key = (backbone, gal, qt)
                    val = lookup.get(key, {}).get(metric_key, 0)
                    alpha = gallery_alphas[gal]

                    ax.bar(x, val, width=bar_width,
                           color=base_color, alpha=alpha,
                           edgecolor=base_color, linewidth=1.2)
                    ax.text(x, val + 0.01, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=6, rotation=90)

        # X-axis: query thresholds
        ax.set_xticks(range(len(query_thresholds)))
        ax.set_xticklabels([f'{q:.2f}' for q in query_thresholds])
        ax.set_xlabel('Query Filter Threshold')
        ax.set_ylabel(metric_label)
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.05))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Legend with bolded section titles aligned over patches
        legend_handles = []
        handler_map = {}
        # Model section header
        model_title = mpatches.Patch(
            facecolor='none', edgecolor='none', label=r'$\bf{Model}$')
        legend_handles.append(model_title)
        handler_map[model_title] = _TitleHandler()
        for backbone in backbones:
            patch = mpatches.Patch(facecolor=backbone_hues[backbone],
                                  edgecolor=backbone_hues[backbone],
                                  label=backbone_full_names[backbone])
            legend_handles.append(patch)
        # Gallery filter section header
        gallery_title = mpatches.Patch(
            facecolor='none', edgecolor='none',
            label=r'$\bf{Gallery\ filter\ thresh.}$')
        legend_handles.append(gallery_title)
        handler_map[gallery_title] = _TitleHandler()
        for gt in gallery_thresholds:
            patch = mpatches.Patch(facecolor='gray', alpha=gallery_alphas[gt],
                                  edgecolor='gray',
                                  label=f'{gt:.2f}')
            legend_handles.append(patch)

        leg = ax.legend(handles=legend_handles, handler_map=handler_map,
                        loc='lower right',
                        fontsize=4.8,
                        ncol=2, framealpha=0.9, handletextpad=0.5,
                        columnspacing=0.8, labelspacing=0.25,
                        handleheight=0.5)
        leg._legend_box.align = 'left'

        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved figure: {path}')


def main():
    # Run from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    # ---- Hygiene sweep figures/tables (direct import) ----
    print("Running aggregate analysis ...")
    from utils.reid_plotting import run_aggregate
    model_data = run_aggregate()

    # ---- Few-shot tables -> reid_openset/summary/fewshot/ ----
    fewshot_dir = 'reid_openset/summary/fewshot'
    if model_data:
        create_fewshot_tables(model_data, fewshot_dir)
        create_hyperparameters_table(fewshot_dir)

    # ---- Test eval -> reid_openset/summary/test_eval/ ----
    test_eval_dir = 'reid_openset/summary/test_eval'

    all_data = []
    for path in sorted(glob.glob('reid_openset/*/results/test_eval/*.json')):
        with open(path) as f:
            all_data.append(json.load(f))

    if not all_data:
        print('No test_eval JSONs found.')
    else:
        create_test_eval_tables(all_data, test_eval_dir)
        print(f'\nGenerating test eval figures -> {test_eval_dir}/')
        create_test_eval_figures(all_data, test_eval_dir)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Aggregate test_eval JSONs into recall_at_1 and balanced_accuracy tables."""

import json
import glob
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def write_table(rows, cols, name, out_dir):
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
        ('recall_at_1', 'Recall@1', 'test_rank@1.png'),
        ('balanced_accuracy', 'Balanced Accuracy', 'test_novelty_detection.png'),
    ]

    for metric_key, metric_label, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

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
                    ax.text(x, val + 0.01, f'{val:.2f}',
                            ha='center', va='bottom', fontsize=6, rotation=90)

        # X-axis: query thresholds
        ax.set_xticks(range(len(query_thresholds)))
        ax.set_xticklabels([f'{q:.2f}' for q in query_thresholds])
        ax.set_xlabel('Query Filter Threshold')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Query Threshold, Backbone, and Gallery Threshold')
        ax.set_ylim(0, 1.05)

        # Legend with bolded section titles
        legend_handles = []
        # Model section header
        legend_handles.append(mpatches.Patch(
            facecolor='none', edgecolor='none', label=r'$\bf{Model}$'))
        for backbone in backbones:
            patch = mpatches.Patch(facecolor=backbone_hues[backbone],
                                  edgecolor=backbone_hues[backbone],
                                  label=backbone_full_names[backbone])
            legend_handles.append(patch)
        # Gallery filter section header
        legend_handles.append(mpatches.Patch(
            facecolor='none', edgecolor='none',
            label=r'$\bf{Gallery\ filter\ thresh.}$'))
        for gt in gallery_thresholds:
            patch = mpatches.Patch(facecolor='gray', alpha=gallery_alphas[gt],
                                  edgecolor='gray',
                                  label=f'{gt:.2f}')
            legend_handles.append(patch)

        ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
                  ncol=2, framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved figure: {path}')


def main():
    # Run from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # ---- Hygiene sweep figures/tables (existing) ----
    print("Running reid_plotting --aggregate ...")
    subprocess.run([sys.executable, '-u', '-m', 'utils.reid_plotting', '--aggregate'],
                   check=False)

    # ---- Test eval: aggregate per-job JSONs into tables ----
    out_dir = 'reid_openset/tables'
    os.makedirs(out_dir, exist_ok=True)

    all_data = []
    for path in sorted(glob.glob('reid_openset/*/results/test_eval/*.json')):
        with open(path) as f:
            all_data.append(json.load(f))

    if not all_data:
        print('No test_eval JSONs found.')
        return

    # ---- Recall@1 table ----
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
    print('=== Recall@1 ===')
    write_table(r1_rows, r1_cols, 'recall_at_1', out_dir)

    # ---- Balanced Accuracy table ----
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
    print('=== Balanced Accuracy ===')
    write_table(ba_rows, ba_cols, 'balanced_accuracy', out_dir)

    print(f'\nTotal: {len(r1_rows)} R@1 rows + {len(ba_rows)} BA rows '
          f'from {len(all_data)} JSONs')

    # ---- Test eval figures ----
    fig_dir = 'reid_openset/figures'
    print(f'\nGenerating test eval figures -> {fig_dir}/')
    create_test_eval_figures(all_data, fig_dir)


if __name__ == '__main__':
    main()

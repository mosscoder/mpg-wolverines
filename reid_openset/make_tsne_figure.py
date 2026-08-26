"""Build the best-condition t-SNE figure (manuscript Figure 7).

Columns are the three backbones. Row 1 shows each model's best test
Recall@1 filter combination: the training gallery as washed identity-tinted
circles with test knowns overlaid as black-edged identity-colored triangles.
Row 2 shows each model's best balanced accuracy combination: the
quality-filtered training gallery by identity with unknowns overlaid as red
triangles. Each combination's query filter is applied to the displayed
test-side points. Panels are annotated with their filter combination and
test metric.

Each panel is an independent t-SNE fit over ALL sets for that model state
(gallery + query + rare embeddings), perplexity 50, 2000 iterations,
seed 0, PCA initialization. Panels are aligned to the unfiltered DINOv3
layout by orthogonal Procrustes (rotation plus reflection) over the shared
test rows, so orientation is consistent across panels without changing any
layout's internal geometry.

Inputs are the embedding exports in {model}/results/embeddings/, generated
by each model's scripts/05_generate_embeddings.sbatch. Coordinates are
cached in summary/tsne/tsne_coords.npz and recomputed from the embeddings
if the cache is absent (about 30 to 40 minutes at N_JOBS=9).

Palette: Turk orange and HFW12-F7 brown (swapped from tab10 order for
contrast against HLC20-H3 blue), Tex cyan (black is reserved for edges and
test knowns), unknowns red (tab10 red is reserved for them).

    python reid_openset/make_tsne_figure.py
"""
import json
import os
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from scipy.linalg import orthogonal_procrustes

HERE = os.path.dirname(os.path.abspath(__file__))          # reid_openset/
MODELS = ['dinov3', 'megadescriptor', 'bioclip2']
LABELS = {'dinov3': 'DINOv3-ViT-B/16', 'megadescriptor': 'MegaDescriptor-L-384',
          'bioclip2': 'BioCLIP-2 ViT-L/14'}
# Best (gallery_q, query_q) combination per model and metric, from
# results/test_eval/g{g}_q{q}.json (looked up 2026-08-26). The embedding
# file carries that configuration's best training step. Verified against
# the JSONs at runtime below.
# (cache key, npz file, gallery_q, query_q, metric value)
BEST = {
    'r1': {
        'dinov3': ('dinov3', 'emb_g0.00_step2600.npz', 0.00, 0.25, 0.870),
        'megadescriptor': ('megadescriptor', 'emb_g0.00_step2600.npz', 0.00, 0.50, 0.786),
        'bioclip2': ('bioclip2', 'emb_g0.00_step1200.npz', 0.00, 0.50, 0.827),
    },
    'ba': {
        'dinov3': ('dinov3_g050', 'emb_g0.50_step700.npz', 0.50, 0.50, 0.801),
        'megadescriptor': ('megadescriptor_g050_s1200', 'emb_g0.50_step1200.npz', 0.50, 0.25, 0.614),
        'bioclip2': ('bioclip2_g050_s1200', 'emb_g0.50_step1200.npz', 0.50, 0.00, 0.502),
    },
}
METRIC_KEYS = {'r1': 'recall_at_1', 'ba': 'balanced_accuracy'}
ROW_META = [('r1', 'Best re-identification', 'Recall@1'),
            ('ba', 'Best novelty detection', 'Balanced acc.')]
KNOWN_ORDER = ['HLC20-H3', 'HFW12-F7', 'BDF10-M6', 'Turk', 'LH23-M1', 'Tex']
PALETTE = plt.get_cmap('tab10')
ID_COLORS = {name: PALETTE(i) for i, name in enumerate(KNOWN_ORDER)}
ID_COLORS['Turk'] = '#ff7f0e'      # orange (swapped with HFW12-F7 for contrast)
ID_COLORS['HFW12-F7'] = '#8c564b'  # brown
ID_COLORS['Tex'] = '#17becf'       # cyan
UNKNOWN_COLOR = '#d62728'
SEED = 0
PERPLEXITY = 50
ITERS = 2000
N_JOBS = 9
ITER_KW = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
OUT_DIR = os.path.join(HERE, 'summary', 'tsne')
CACHE = os.path.join(OUT_DIR, 'tsne_coords.npz')


def verify_best(model, rk, fname, g_q, q_q, val):
    """The BEST table must agree with the committed test_eval results."""
    path = os.path.join(HERE, model, 'results', 'test_eval',
                        f'g{g_q:.2f}_q{q_q:.2f}.json')
    with open(path) as f:
        r = json.load(f)
    step = int(fname.split('_step')[1].split('.npz')[0])
    assert r['config']['best_step'] == step, \
        f'{model}/{rk}: step {step} != {r["config"]["best_step"]} in {path}'
    metric = r['results'][METRIC_KEYS[rk]]
    assert abs(metric - val) < 5e-4, \
        f'{model}/{rk}: metric {val} != {metric:.4f} in {path}'


os.makedirs(OUT_DIR, exist_ok=True)
cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}


def load_coords(model, key, fname, d, n_g):
    if key in cache:
        return cache[key]
    X = np.concatenate([d['gallery_emb'], d['query_emb'],
                        d['rare_emb']]).astype(np.float32)
    xy = TSNE(n_components=2, perplexity=PERPLEXITY, init='pca',
              random_state=SEED, learning_rate='auto', n_jobs=N_JOBS,
              **{ITER_KW: ITERS}).fit_transform(X)
    cache[key] = xy
    np.savez(CACHE, **cache)
    print(f'{model} {fname}: t-SNE done ({n_g} gallery)', flush=True)
    return xy


# orientation reference: test rows of the unfiltered DINOv3 layout
ref_key, ref_fname = BEST['r1']['dinov3'][0], BEST['r1']['dinov3'][1]
ref_d = np.load(os.path.join(HERE, 'dinov3', 'results', 'embeddings', ref_fname),
                allow_pickle=True)
n_ref_g = len(ref_d['gallery_emb'])
ref_xy_all = load_coords('dinov3', ref_key, ref_fname, ref_d, n_ref_g)
ref_test_xy = ref_xy_all[n_ref_g:]
ref_test_xy = ref_test_xy - ref_test_xy.mean(axis=0)

fig = plt.figure(figsize=(16.5, 9.6))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.30],
              hspace=0.08, wspace=0.04)

for row, (rk, row_label, metric_name) in enumerate(ROW_META):
    for col, model in enumerate(MODELS):
        key, fname, g_q, q_q, val = BEST[rk][model]
        verify_best(model, rk, fname, g_q, q_q, val)
        d = np.load(os.path.join(HERE, model, 'results', 'embeddings', fname),
                    allow_pickle=True)
        n_g = len(d['gallery_emb'])
        n_q = len(d['query_ids'])
        gallery_ids = d['gallery_ids']
        test_ids = np.concatenate([d['query_ids'], d['rare_ids']])
        known_mask = np.zeros(len(test_ids), bool)
        known_mask[:n_q] = True
        quality = np.concatenate([d['query_quality'], d['rare_quality']])
        xy_all = load_coords(model, key, fname, d, n_g)
        c = xy_all - xy_all[n_g:].mean(axis=0)
        R, _ = orthogonal_procrustes(c[n_g:], ref_test_xy)
        aligned = c @ R
        g_xy, t_xy = aligned[:n_g], aligned[n_g:]
        keep = quality >= q_q if q_q > 0 else np.ones(len(quality), bool)

        ax = fig.add_subplot(gs[row, col])
        if col == 2:
            if row == 0:
                row0_end_ax = ax
            else:
                row1_end_ax = ax
        for name in KNOWN_ORDER:
            m = gallery_ids == name
            base = np.array(matplotlib.colors.to_rgb(ID_COLORS[name]))
            tint = 1 - (1 - base) * 0.55  # wash toward white
            g_alpha = 0.25 if rk == 'r1' else 0.6
            ax.scatter(g_xy[m, 0], g_xy[m, 1], s=5,
                       color=tint, linewidths=0, alpha=g_alpha, zorder=2)
        if rk == 'r1':
            for name in KNOWN_ORDER:
                m = (test_ids == name) & known_mask & keep
                ax.scatter(t_xy[m, 0], t_xy[m, 1], s=16, marker='^',
                           color=ID_COLORS[name], edgecolors='black',
                           linewidths=0.4, zorder=4)
        else:
            m = ~known_mask & keep
            ax.scatter(t_xy[m, 0], t_xy[m, 1], s=16, marker='^',
                       color=UNKNOWN_COLOR, edgecolors='black',
                       linewidths=0.4, alpha=0.6, zorder=4)
        ax.set_xlim(aligned[:, 0].min() - 2, aligned[:, 0].max() + 2)
        ax.set_ylim(aligned[:, 1].min() - 2, aligned[:, 1].max() + 2)
        if row == 0:
            ax.set_title(LABELS[model], fontsize=13, fontweight='bold', pad=10)
        ax.text(0.02, 0.98, 'ABCDEF'[row * 3 + col], transform=ax.transAxes,
                fontsize=17, fontweight='bold', va='top', ha='left')
        ax.text(0.98, 0.02,
                f'gallery >= {g_q:.2f}\nquery >= {q_q:.2f}\n{metric_name} = {val:.3f}',
                transform=ax.transAxes, fontsize=7.2, ha='right', va='bottom',
                color='#444444', zorder=5,
                bbox=dict(facecolor='white', edgecolor='#cccccc',
                          boxstyle='round,pad=0.35'))
        if row == 1:
            ax.set_xlabel('t-SNE dimension 1', fontsize=11)
        if col == 0:
            ax.set_ylabel('t-SNE dimension 2', fontsize=11)
            ax.text(-0.13, 0.5, row_label, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', rotation=90,
                    va='center', ha='center')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#cccccc')

handles1 = ([Line2D([], [], ls='', marker='o', ms=6, color=ID_COLORS[n], label=n)
             for n in KNOWN_ORDER] +
            [Line2D([], [], ls='', marker='', label='$\\bf{Sample\\ set}$'),
             Line2D([], [], ls='', marker='o', ms=6, color='#666666',
                    label='Train known'),
             Line2D([], [], ls='', marker='^', ms=8, color='#666666',
                    markerfacecolor='#666666', label='Test known')])
l1 = row0_end_ax.legend(handles=handles1, loc='upper left',
                        bbox_to_anchor=(1.012, 1.0), fontsize=9, frameon=True,
                        title='Individual',
                        title_fontproperties={'weight': 'bold', 'size': 10})
row0_end_ax.add_artist(l1)

handles2 = [
    Line2D([], [], ls='', marker='o', ms=6, color='#666666',
           label='Train known'),
    Line2D([], [], ls='', marker='^', ms=8, color=UNKNOWN_COLOR,
           markerfacecolor=UNKNOWN_COLOR, label='Unknown'),
]
row1_end_ax.legend(handles=handles2, loc='upper left',
                   bbox_to_anchor=(1.012, 1.0), fontsize=9, frameon=True,
                   title='Sample set',
                   title_fontproperties={'weight': 'bold', 'size': 10})

out = os.path.join(OUT_DIR, 'tsne_best_conditions.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'saved {out}')

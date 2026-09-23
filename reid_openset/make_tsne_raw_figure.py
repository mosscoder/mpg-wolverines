"""Build the unfiltered-versus-filtered t-SNE figure (manuscript Figure 7).

Columns are the three backbones. Row 1 is the trained ArcFace embedding with
no quality filter (gallery and query thresholds of 0): a seeded subsample of
the known individuals' gallery images as identity-tinted circles and every
image of the four unknowns as black-edged red triangles. Row 2 is the trained
ArcFace embedding at gallery and query thresholds of 0.50, showing that
filtered gallery and the unknown images scoring >= 0.50. Each row draws the
unknown images its query threshold admits.

Each panel is one t-SNE fit over all of that panel's points (gallery plus
test knowns plus unknowns; the test knowns are fitted but not drawn),
perplexity 50, 1000 iterations, seed 0, PCA initialization. Panels are
aligned to the DINOv3 panel of their row by orthogonal Procrustes over the
shared test rows, so orientation is consistent without changing any
layout's internal geometry.

Inputs. Both rows read the tracked embedding exports in
{model}/results/embeddings/ (see each model's
scripts/05_generate_embeddings.sbatch), each at the step selected on the
validation queries for its filter combination (Supplementary Table S5). The
row 1 subsample is recorded, with image filenames, in
summary/tsne/tsne_raw_meta.npz, which this script writes from the Hugging
Face dataset if absent. t-SNE coordinates are cached in
summary/tsne/tsne_raw_coords.npz. Both files are tracked so the figure
regenerates without recompute.

Palette: Turk orange, HFW12-F7 brown, Tex cyan,
unknowns red.

    python reid_openset/make_tsne_raw_figure.py
"""
import inspect
import json
import os
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

HERE = os.path.dirname(os.path.abspath(__file__))          # reid_openset/
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

MODELS = ['dinov3', 'megadescriptor', 'bioclip2']
LABELS = {'dinov3': 'DINOv3-ViT-B/16', 'megadescriptor': 'MegaDescriptor-L-384',
          'bioclip2': 'BioCLIP-2 ViT-L/14'}
# Embedding exports at the step selected on the validation queries for each
# filter combination (Supplementary Table S5). Row 1: gallery and query
# thresholds of 0. Row 2: gallery and query thresholds of 0.50.
UNFILTERED = {'dinov3': 'emb_g0.00_step2500.npz',
              'megadescriptor': 'emb_g0.00_step700.npz',
              'bioclip2': 'emb_g0.00_step1400.npz'}
FILTERED = {'dinov3': 'emb_g0.50_step700.npz',
            'megadescriptor': 'emb_g0.50_step1000.npz',
            'bioclip2': 'emb_g0.50_step900.npz'}
KNOWN_ORDER = ['HLC20-H3', 'HFW12-F7', 'BDF10-M6', 'Turk', 'LH23-M1', 'Tex']
PALETTE = plt.get_cmap('tab10')
ID_COLORS = {n: PALETTE(i) for i, n in enumerate(KNOWN_ORDER)}
ID_COLORS.update({'Turk': '#ff7f0e', 'HFW12-F7': '#8c564b', 'Tex': '#17becf'})
UNKNOWN_COLOR = '#d62728'
Q = 0.5                      # row 2 gallery filter and unknown query filter
N_PER_KNOWN, SEED = 400, 0   # row 1 gallery subsample per known individual
PERPLEXITY, ITERS, N_JOBS = 50, 1000, 9
ITER_KW = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'

OUT_DIR = os.path.join(HERE, 'summary', 'tsne')
META = os.path.join(OUT_DIR, 'tsne_raw_meta.npz')
CACHE = os.path.join(OUT_DIR, 'tsne_raw_coords.npz')
OUT = os.path.join(OUT_DIR, 'tsne_raw_vs_trained.png')


# ---------------------------------------------------------------- row 1 inputs
def select_images():
    """Seeded subsample of known gallery images, all test images of the
    knowns, and every image of the unknowns (both splits)."""
    from utils.reid_data import (load_reidentification_dataset,
                                 load_reidentification_test_dataset)
    cfg = json.load(open(os.path.join(REPO, 'preprocessing', 'results',
                                      'feasible_individuals.json')))
    unk = json.load(open(os.path.join(REPO, 'preprocessing', 'results',
                                      'unknown_assignment.json')))
    train = load_reidentification_dataset()
    test = load_reidentification_test_dataset()
    tr_ids = np.array(train['id'], dtype=object)
    te_ids = np.array(test['id'], dtype=object)
    tr_q = np.array(train['pelage_score'], np.float32)
    te_q = np.array(test['pelage_score'], np.float32)
    tr_fn = np.array(train['filename'], dtype=object)
    te_fn = np.array(test['filename'], dtype=object)
    rng = np.random.RandomState(SEED)
    rows = []   # (split, index, role, individual, quality, filename)
    for ind in cfg['qualified_individuals']:
        idx = np.where(tr_ids == ind)[0]
        for i in rng.choice(idx, min(N_PER_KNOWN, len(idx)), replace=False):
            rows.append(('train', int(i), 'gallery', ind, tr_q[i], tr_fn[i]))
    for ind in cfg['qualified_individuals']:
        for i in np.where(te_ids == ind)[0]:
            rows.append(('test', int(i), 'test_known', ind, te_q[i], te_fn[i]))
    for ind, d in unk['per_individual'].items():
        for i in d['train_indices']:
            rows.append(('train', int(i), 'unknown', ind, tr_q[i], tr_fn[i]))
        for i in d['test_indices']:
            rows.append(('test', int(i), 'unknown', ind, te_q[i], te_fn[i]))
    np.savez(META, split=np.array([r[0] for r in rows]),
             index=np.array([r[1] for r in rows]),
             role=np.array([r[2] for r in rows]),
             ident=np.array([r[3] for r in rows], dtype=object),
             quality=np.array([r[4] for r in rows], np.float32),
             filename=np.array([r[5] for r in rows]))


# ---------------------------------------------------------------------- t-SNE
cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}


def tsne(key, X):
    if key not in cache:
        cache[key] = TSNE(n_components=2, perplexity=PERPLEXITY, init='pca',
                          random_state=SEED, learning_rate='auto', n_jobs=N_JOBS,
                          **{ITER_KW: ITERS}).fit_transform(X.astype(np.float32))
        np.savez(CACHE, **cache)
        print(f'{key}: t-SNE done on {len(X)} points', flush=True)
    return cache[key]


def load_export(model, files):
    return np.load(os.path.join(HERE, model, 'results', 'embeddings', files[model]),
                   allow_pickle=True)


def test_rows(d):
    """Known test queries then unknowns, the rows shared across panels."""
    known = np.zeros(len(d['query_ids']) + len(d['rare_ids']), bool)
    known[:len(d['query_ids'])] = True
    quality = np.concatenate([d['query_quality'], d['rare_quality']])
    order = np.concatenate([d['query_filenames'], d['rare_filenames']])
    return known, quality, order


def panel_unfiltered(model):
    d = load_export(model, UNFILTERED)
    meta = np.load(META, allow_pickle=True)
    subsample = meta['filename'][meta['role'] == 'gallery']
    pos = {n: i for i, n in enumerate(d['gallery_filenames'])}
    gi = np.array([pos[n] for n in subsample])
    key = f'unfiltered_{model}'
    X = None if key in cache else np.concatenate(
        [d['gallery_emb'][gi], d['query_emb'], d['rare_emb']]).astype(np.float32)
    known, _, order = test_rows(d)
    return dict(xy=tsne(key, X), n_g=len(gi),
                gallery_ids=d['gallery_ids'][gi].astype(str),
                known=known, keep=np.ones(len(known), bool), order=order)


def panel_filtered(model):
    d = load_export(model, FILTERED)
    key = f'trained_{model}'
    X = None if key in cache else np.concatenate(
        [d['gallery_emb'], d['query_emb'], d['rare_emb']]).astype(np.float32)
    known, quality, order = test_rows(d)
    return dict(xy=tsne(key, X), n_g=len(d['gallery_emb']),
                gallery_ids=d['gallery_ids'].astype(str),
                known=known, keep=quality >= Q, order=order)


# ----------------------------------------------------------------------- figure
def main():
    if not os.path.exists(META):
        select_images()

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2))
    rows = [(panel_unfiltered, 'After ArcFace training\nNo quality filtering'),
            (panel_filtered, f'After ArcFace training on gallery >= {Q:.1f}\n'
                             f'Unknown queries >= {Q:.1f}')]
    for row, (maker, row_label) in enumerate(rows):
        panels = {m: maker(m) for m in MODELS}
        ref = panels['dinov3']
        assert all((panels[m]['order'] == ref['order']).all() for m in MODELS)
        ref_test = ref['xy'][ref['n_g']:] - ref['xy'][ref['n_g']:].mean(axis=0)
        for col, model in enumerate(MODELS):
            p = panels[model]
            c = p['xy'] - p['xy'][p['n_g']:].mean(axis=0)
            R, _ = orthogonal_procrustes(c[p['n_g']:], ref_test)
            a = c @ R
            g_xy, t_xy = a[:p['n_g']], a[p['n_g']:]
            ax = axes[row, col]
            g_cols = np.array([1 - (1 - np.array(matplotlib.colors.to_rgb(ID_COLORS[n]))) * 0.55
                               for n in p['gallery_ids']])
            m = ~p['known'] & p['keep']
            u_xy = t_xy[m]
            # Unknowns are black-edged red triangles. Row 1 draws them under the
            # known dots (2,542 unknowns would otherwise bury the knowns); row 2
            # draws its 78 unknowns on top.
            z_unknown, z_known = (2, 5) if row == 0 else (5, 2)
            ax.scatter(u_xy[:, 0], u_xy[:, 1], s=16, marker='^',
                       color=UNKNOWN_COLOR, edgecolors='black', linewidths=0.4,
                       alpha=0.8, zorder=z_unknown)
            ax.scatter(g_xy[:, 0], g_xy[:, 1], s=9, c=g_cols,
                       edgecolors='white', linewidths=0.35, alpha=0.70, zorder=z_known)
            ax.set_xlim(a[:, 0].min() - 2, a[:, 0].max() + 2)
            ax.set_ylim(a[:, 1].min() - 2, a[:, 1].max() + 2)
            if row == 0:
                ax.set_title(LABELS[model], fontsize=13, fontweight='bold', pad=10)
            ax.text(0.02, 0.98, 'ABCDEF'[row * 3 + col], transform=ax.transAxes,
                    fontsize=17, fontweight='bold', va='top', ha='left')
            if col == 0:
                ax.set_ylabel('t-SNE dimension 2', fontsize=11)
                ax.text(-0.13, 0.5, row_label, transform=ax.transAxes, fontsize=13,
                        fontweight='bold', rotation=90, va='center', ha='center')
            if row == 1:
                ax.set_xlabel('t-SNE dimension 1', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#cccccc')

    handles = ([Line2D([], [], ls='', marker='o', ms=6, color=ID_COLORS[n], label=n)
                for n in KNOWN_ORDER]
               + [Line2D([], [], ls='', marker='^', ms=7, mew=0.6,
                         color=UNKNOWN_COLOR, markeredgecolor='black', label='Unknown')])
    axes[0, 2].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=9, frameon=True, title='Individual',
                      title_fontproperties={'weight': 'bold', 'size': 10})
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches='tight')
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()

"""Build the raw-versus-trained t-SNE figure (manuscript Figure 7).

Columns are the three backbones. Row 1 is each backbone's frozen feature
space with no training and no quality filter: a seeded subsample of the
known individuals' training images as identity-tinted circles and every
image of the four simulated unknowns as black-edged red triangles. Row 2 is the
trained ArcFace embedding at the gallery >= 0.50 configuration, showing
that filtered training gallery and the unknown images scoring >= 0.50.

Each panel is one t-SNE fit over all of that panel's points (gallery plus
test knowns plus unknowns; the test knowns are fitted but not drawn),
perplexity 50, 1000 iterations, seed 0, PCA initialization. Panels are
aligned to the DINOv3 panel of their row by orthogonal Procrustes over the
shared test rows, so orientation is consistent without changing any
layout's internal geometry.

Inputs. Row 2 reads the tracked embedding exports in
{model}/results/embeddings/ (see each model's
scripts/05_generate_embeddings.sbatch). Row 1 reads raw backbone features
cached in summary/tsne/raw_features/{model}_raw.npy, which this script
computes from the Hugging Face dataset if absent (about 20 minutes on a
Mac, MegaDescriptor being the slow one). Image selection is recorded in
summary/tsne/tsne_raw_meta.npz and t-SNE coordinates in
summary/tsne/tsne_raw_coords.npz; both are tracked so the figure
regenerates without recompute.

Palette: Turk orange, HFW12-F7 brown, Tex cyan,
unknowns red.

    python reid_openset/make_tsne_raw_figure.py
"""
import inspect
import json
import os
import sys
import time

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
# Row 2: the gallery >= 0.50 embedding export at the step that maximized
# test balanced accuracy for query >= 0.50 (results/test_eval/g0.50_q0.50.json).
TRAINED = {'dinov3': 'emb_g0.50_step700.npz',
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
RAW_DIR = os.path.join(OUT_DIR, 'raw_features')
META = os.path.join(OUT_DIR, 'tsne_raw_meta.npz')
CACHE = os.path.join(OUT_DIR, 'tsne_raw_coords.npz')
OUT = os.path.join(OUT_DIR, 'tsne_raw_vs_trained.png')


# ---------------------------------------------------------------- row 1 inputs
def select_images():
    """Seeded subsample of known training images, all test images of the
    knowns, and every image of the simulated unknowns (both splits)."""
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
    rng = np.random.RandomState(SEED)
    rows = []   # (split, index, role, individual, quality)
    for ind in cfg['qualified_individuals']:
        idx = np.where(tr_ids == ind)[0]
        for i in rng.choice(idx, min(N_PER_KNOWN, len(idx)), replace=False):
            rows.append(('train', int(i), 'gallery', ind, tr_q[i]))
    for ind in cfg['qualified_individuals']:
        for i in np.where(te_ids == ind)[0]:
            rows.append(('test', int(i), 'test_known', ind, te_q[i]))
    for ind, d in unk['per_individual'].items():
        for i in d['train_indices']:
            rows.append(('train', int(i), 'unknown', ind, tr_q[i]))
        for i in d['test_indices']:
            rows.append(('test', int(i), 'unknown', ind, te_q[i]))
    np.savez(META, split=np.array([r[0] for r in rows]),
             index=np.array([r[1] for r in rows]),
             role=np.array([r[2] for r in rows]),
             ident=np.array([r[3] for r in rows], dtype=object),
             quality=np.array([r[4] for r in rows], np.float32))
    return train, test


def backbone_fn(model, device):
    """The pipeline's own backbone factories, features only."""
    from utils.arcface import (create_bioclip2_arcface_model,
                               create_dinov3_arcface_model,
                               create_megadescriptor_arcface_model)
    if model == 'dinov3':
        m, _ = create_dinov3_arcface_model(embedding_dim=256, device=device)
        return lambda x: m.backbone(x).last_hidden_state[:, 0, :]
    if model == 'megadescriptor':
        m, _ = create_megadescriptor_arcface_model(embedding_dim=256, device=device)
        return lambda x: m.backbone(x)
    m, _ = create_bioclip2_arcface_model(embedding_dim=256, image_size=224,
                                         device=device)
    return lambda x: m.backbone.encode_image(x)


def extract_raw_features(missing):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from utils.reid_config import create_transform_for_model

    class ImageSet(Dataset):
        def __init__(self, train, test, meta, tf):
            self.train, self.test, self.meta, self.tf = train, test, meta, tf

        def __len__(self):
            return len(self.meta['index'])

        def __getitem__(self, i):
            ds = self.train if self.meta['split'][i] == 'train' else self.test
            return self.tf(ds[int(self.meta['index'][i])]['image'].convert('RGB'))

    train, test = select_images()
    meta = np.load(META, allow_pickle=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    os.makedirs(RAW_DIR, exist_ok=True)
    for model in missing:
        fn = backbone_fn(model, device)
        loader = DataLoader(ImageSet(train, test, meta,
                                     create_transform_for_model(model)),
                            batch_size=32, num_workers=6, shuffle=False)
        feats, t0 = [], time.time()
        with torch.no_grad():
            for x in loader:
                feats.append(fn(x.to(device)).float().cpu().numpy())
        np.save(os.path.join(RAW_DIR, f'{model}_raw.npy'), np.concatenate(feats))
        print(f'{model}: raw features cached in {(time.time() - t0) / 60:.1f} min',
              flush=True)


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


def panel_raw(model):
    key = f'raw_{model}'
    if key in cache:
        X = None
    else:
        X = np.load(os.path.join(RAW_DIR, f'{model}_raw.npy'))
    meta = np.load(META, allow_pickle=True)
    role, ident, quality = meta['role'], meta['ident'].astype(str), meta['quality']
    g = role == 'gallery'
    order = np.concatenate([np.where(g)[0], np.where(~g)[0]])   # gallery first
    xy = tsne(key, X[order] if X is not None else None)
    n_g = int(g.sum())
    return dict(xy=xy, n_g=n_g, gallery_ids=ident[order][:n_g],
                known=(role[order][n_g:] == 'test_known'),
                keep=np.ones(len(order) - n_g, bool))


def panel_trained(model):
    d = np.load(os.path.join(HERE, model, 'results', 'embeddings', TRAINED[model]),
                allow_pickle=True)
    key = f'trained_{model}'
    X = None if key in cache else np.concatenate(
        [d['gallery_emb'], d['query_emb'], d['rare_emb']]).astype(np.float32)
    xy = tsne(key, X)
    n_g = len(d['gallery_emb'])
    known = np.zeros(len(d['query_ids']) + len(d['rare_ids']), bool)
    known[:len(d['query_ids'])] = True
    quality = np.concatenate([d['query_quality'], d['rare_quality']])
    return dict(xy=xy, n_g=n_g, gallery_ids=d['gallery_ids'].astype(str),
                known=known, keep=quality >= Q)


# ----------------------------------------------------------------------- figure
def main():
    if not all(f'raw_{m}' in cache for m in MODELS):
        missing = [m for m in MODELS
                   if not os.path.exists(os.path.join(RAW_DIR, f'{m}_raw.npy'))]
        if missing or not os.path.exists(META):
            extract_raw_features(missing or MODELS)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2))
    rows = [(panel_raw, 'Before ArcFace training\nNo quality filtering'),
            (panel_trained, f'After ArcFace training on gallery >= {Q:.1f}\n'
                            f'Unknown queries >= {Q:.1f}')]
    for row, (maker, row_label) in enumerate(rows):
        panels = {m: maker(m) for m in MODELS}
        ref = panels['dinov3']
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

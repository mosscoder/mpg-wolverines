"""Figure 2 (R1 revision): the quality score as a gradient.

Rows are the three known individuals with the most training images (Turk,
HLC20-H3, HFW12-F7). Columns are the ten tenths of the quality score range
([0.0, 0.1), ..., [0.9, 1.0]). Each cell shows one training image of that
individual whose score falls in that tenth, with the exact score printed
lower left, and a white-to-green gradient arrow labeled "Quality score"
runs beneath the grid. The word "bin" appears nowhere on the figure.

Selection is deterministic. Within each cell the candidates are that
individual's training crops in the score range whose bounding-box aspect
(height over width) is within ATOL of the 1:2 thumbnail box, or the KA
nearest in aspect when fewer than KA qualify, so that every thumbnail keeps
at least 96% of its crop after the center crop to the box. One candidate is
drawn with a seeded generator (SEED, default 1, chosen 2026-09-14). A
different seed rerolls every cell: python preprocessing/make_quality_gradient_figure.py 3

Inputs: the local re-identification dataset train split
(hugging_face_dataset/v2/data/reidentification_dataset/train, columns id,
pelage_score, bbox_width, bbox_height, image). Output:
preprocessing/results/quality_gradient.png, copied to
manuscript/flat_submission/ by manuscript/update.py.

Run in the wolverines env (datasets, pyarrow).
"""
import os
import sys

import numpy as np
from datasets import load_from_disk
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))              # preprocessing/
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'hugging_face_dataset', 'v2', 'data',
                    'reidentification_dataset', 'train')
OUT = os.path.join(HERE, 'results', 'quality_gradient.png')

ROWS = ['Turk', 'HLC20-H3', 'HFW12-F7']   # most training images, Table 2
NB = 10                                    # tenths of the score range
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
W, H = 240, 480                            # thumbnail box, 1:2 like the typical crop
BOX = H / W
ATOL = 0.08                                # accept crops within this much of BOX
KA = 8                                     # else fall back to the KA nearest in aspect
ARROW_GREEN = '#2ca02c'


def fit_box(img):
    """Center-crop to the W:H box, then resize, so the thumb fills its cell."""
    w, h = img.size
    if w / h > W / H:
        nw = int(h * W / H)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    else:
        nh = int(w * H / W)
        top = (h - nh) // 2
        img = img.crop((0, top, w, top + nh))
    return img.resize((W, H), Image.Resampling.LANCZOS)


def main():
    d = load_from_disk(DATA)
    ids = np.array(d['id'])
    scores = np.array(d['pelage_score'], dtype=float)
    aspect = (np.array(d['bbox_height'], dtype=float)
              / np.array(d['bbox_width'], dtype=float))
    upright = (aspect >= 1.0) & (aspect <= 2.5)
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(len(ROWS), NB,
                             figsize=(NB * 1.12 + 0.45, len(ROWS) * 2.55))
    fig.subplots_adjust(wspace=0.01, hspace=0.02, left=0.04, right=0.995,
                        top=0.95, bottom=0.10)
    for r, ind in enumerate(ROWS):
        for c in range(NB):
            lo, hi = c / NB, (c + 1) / NB
            in_range = (scores >= lo) & ((scores < hi) if c < NB - 1 else (scores <= hi))
            cand = np.flatnonzero((ids == ind) & upright & in_range)
            da = np.abs(aspect[cand] - BOX)
            fit = cand[da <= ATOL]
            cand = fit if len(fit) >= KA else cand[np.argsort(da, kind='stable')[:KA]]
            i = int(rng.choice(cand))
            ax = axes[r, c]
            ax.imshow(fit_box(d[i]['image'].convert('RGB')))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(ind, fontsize=9, fontweight='bold')
            ax.text(0.03, 0.03, f'{scores[i]:.3f}', transform=ax.transAxes,
                    fontsize=7, ha='left', va='bottom', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
            kept = min(BOX / aspect[i], aspect[i] / BOX)
            print(f'{ind} [{lo:.1f}, {hi:.1f}): {len(cand):3d} candidates, drew '
                  f'{d[i]["filename"]} score {scores[i]:.3f} kept {kept:.0%}')

    # gradient arrow along the x axis, spanning the image columns
    fig.canvas.draw()
    x0 = axes[-1, 0].get_position().x0
    x1 = axes[-1, -1].get_position().x1
    y_top = axes[-1, 0].get_position().y0
    ah = 0.10
    arr = fig.add_axes([x0, y_top - 0.015 - ah, x1 - x0, ah])
    arr.set_xlim(0, 1)
    arr.set_ylim(0, 1)
    arr.axis('off')
    cmap = LinearSegmentedColormap.from_list('wg', ['#ffffff', ARROW_GREEN])
    im = arr.imshow(np.linspace(0, 1, 512)[None, :], cmap=cmap, aspect='auto',
                    extent=(0, 1, 0, 1), zorder=1)
    head = 0.035
    shape = Polygon([(0, 0.12), (1 - head, 0.12), (1 - head, 0.0), (1, 0.5),
                     (1 - head, 1.0), (1 - head, 0.88), (0, 0.88)], closed=True,
                    facecolor='none', edgecolor='#555555', linewidth=0.8, zorder=2)
    arr.add_patch(shape)
    im.set_clip_path(shape)
    arr.text(0.5, 0.5, 'Quality score', ha='center', va='center', fontsize=12,
             fontweight='bold', color='#222222', zorder=3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()

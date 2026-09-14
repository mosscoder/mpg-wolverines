"""Manuscript Figures 3 and 4 with every test metric read at the fixed
final checkpoint (step 300) instead of the best cross-validated step.
Built for the R1.2 response letter, which shows the step-300 rendering of
Figure 4 to separate checkpoint selection from the non-monotonic curves.

Layout follows utils/reid_plotting_figures.py (linear x at the actual
gallery sizes, 2.5-pt lines, size-8 markers, 0.2 ribbons, shared y-limits
rounded to 0.05, legend below). Family envelope as in the manuscript:
within each filter family, the plotted combination at each gallery size is
the one with the highest mean test metric. Ribbons are 95% t-intervals over
the 8 seeds. Inputs: the committed hygiene JSONs
({model}/results/hygiene/threshold={g}_gallery={n}_seed={s}.json), whose
step_history carries test metrics at every 10th step through 300.

Outputs fig3_step300.png and fig4_step300.png to
manuscript/review/external/figs/ (the response-letter figure directory,
untracked like every other PNG in this repo). Regenerates in seconds.

    python reid_openset/fig34_step300.py
"""
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))            # reid_openset/
OUT_DIR = os.path.join(os.path.dirname(HERE), 'manuscript', 'review',
                       'external', 'figs')
MODELS = ['dinov3', 'megadescriptor', 'bioclip2']
TITLES = {'dinov3': 'DINOv3-ViT-B/16', 'megadescriptor': 'MegaDescriptor-L-384',
          'bioclip2': 'BioCLIP-2 ViT-L/14'}
SIZES = [2, 4, 8, 16, 32, 64]
STEP = 300
FAMILIES = [
    ('None', [(0.0, 'q>=0.0')], '#888888'),
    ('Gallery', [(0.25, 'q>=0.0'), (0.5, 'q>=0.0')], '#1f77b4'),
    ('Query', [(0.0, 'q>=0.25'), (0.0, 'q>=0.5')], '#ff7f0e'),
    ('Gallery + Query', [(g, q) for g in (0.25, 0.5)
                         for q in ('q>=0.25', 'q>=0.5')], '#2ca02c'),
]


def values(model, g, size, q, metric):
    out = []
    pattern = os.path.join(HERE, model, 'results', 'hygiene',
                           f'threshold={g:.2f}_gallery={size}_seed=*.json')
    for p in sorted(glob.glob(pattern)):
        hist = json.load(open(p))['step_history']
        h = [e for e in hist if e['step'] == STEP][0]
        if metric == 'r1':
            out.append(h['test_query_quality_metrics'][q]['recall_at_1'])
        else:
            out.append(h['test_open_set']['by_quality'][q]['balanced_accuracy'])
    assert len(out) == 8, f'{pattern}: {len(out)} seeds'
    return out


def figure(metric, ylabel, outname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    lo_all, hi_all = [], []
    for i, (model, letter) in enumerate(zip(MODELS, 'ABC')):
        ax = axes[0, i]
        for label, combos, color in FAMILIES:
            means, los, his = [], [], []
            for size in SIZES:
                best = max((values(model, g, size, q, metric) for g, q in combos),
                           key=np.mean)
                m = np.mean(best)
                ci = stats.t.ppf(0.975, len(best) - 1) * stats.sem(best)
                means.append(m)
                los.append(m - ci)
                his.append(m + ci)
            ax.plot(SIZES, means, color=color, linewidth=2.5, marker='o',
                    markersize=8, label=label)
            ax.fill_between(SIZES, los, his, color=color, alpha=0.2)
            lo_all += los
            hi_all += his
        ax.set_xlabel('Training examples per individual', fontsize=17.3)
        ax.set_ylabel(ylabel if i == 0 else '', fontsize=17.3)
        ax.set_xticks(SIZES)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(TITLES[model], fontsize=17.3, fontweight='bold', pad=10)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=23.0,
                fontweight='bold', va='top', ha='left')
        ax.tick_params(labelsize=13.0)
    y_min = max(0, np.floor((min(lo_all) - 0.05) / 0.05) * 0.05)
    y_max = min(1, np.ceil((max(hi_all) + 0.05) / 0.05) * 0.05)
    for i, ax in enumerate(axes[0]):
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))
        if i > 0:
            ax.set_yticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=13.0, ncol=4,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
               title=r"$\bf{Image\ quality\ filters}$"
                     + f" (test metrics at the fixed final checkpoint, step {STEP})",
               title_fontsize=13.0)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    out = os.path.join(OUT_DIR, outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    figure('r1', 'Re-identification score\n(Test Recall@1)', 'fig3_step300.png')
    figure('ba', 'Novel wolverine detection score\n(Test Balanced Accuracy)',
           'fig4_step300.png')

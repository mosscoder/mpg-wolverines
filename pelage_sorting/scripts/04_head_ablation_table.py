"""Write the R1.4 head-ablation supplementary table fragment.

Reads the per-seed JSONs written by 04_head_ablation.py (all three head
configurations, including the h=256 width rejected by cross-validation)
and writes a LaTeX tabular body to
results/04_head_ablation/table_s7_head_ablation.tex. manuscript/update.py
copies the fragment into flat_submission/, where supplement.tex \\input's
it. Columns: head, hidden width, trainable parameters, cross-validated
epochs, test macro F1 (mean and standard deviation across five seeds),
macro precision, macro recall.

    python pelage_sorting/scripts/04_head_ablation_table.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results', '04_head_ablation')
OUT = os.path.join(RESULTS, 'table_s7_head_ablation.tex')
LABELS = {('linear', None): 'Linear',
          ('mlp', 256): 'Two-layer, ReLU',
          ('mlp', 768): 'Two-layer, ReLU'}
ORDER = [('linear', None), ('mlp', 256), ('mlp', 768)]

runs = defaultdict(list)
for path in sorted(glob.glob(os.path.join(RESULTS, 'head=*_seed=*.json'))):
    with open(path) as f:
        d = json.load(f)
    runs[(d['head'], d.get('hidden'))].append(d)

lines = []
for key in ORDER:
    ds = runs[key]
    assert len(ds) == 5, f'{key}: expected 5 seeds, found {len(ds)}'
    f1 = np.array([d['final_test']['f1_score'] for d in ds])
    pr = np.array([d['final_test']['precision'] for d in ds])
    rc = np.array([d['final_test']['recall'] for d in ds])
    epochs = {d['optimal_epochs'] for d in ds}
    params = {d['trainable_parameters'] for d in ds}
    assert len(epochs) == 1 and len(params) == 1, key
    hidden = '--' if key[1] is None else str(key[1])
    lines.append(f"{LABELS[key]:<18} & {hidden:>3} & {params.pop():,} & {epochs.pop()} & "
                 f"{f1.mean():.3f} $\\pm$ {f1.std(ddof=1):.3f} & "
                 f"{pr.mean():.3f} & {rc.mean():.3f} \\\\")

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'wrote {OUT}')
print('\n'.join(lines))

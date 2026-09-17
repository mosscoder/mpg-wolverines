"""Write the per-individual table fragment (main-text Table 2).

Reads the local re-identification dataset splits and the validation
carve-out in preprocessing/results/feasible_individuals.json, and writes a
LaTeX tabular body to preprocessing/results/table_individuals.tex.
manuscript/update.py copies it into flat_submission/, where supplement.tex
\\input's it (manuscript.tex, promoted from the supplement 2026-09-11).

Columns: individual, role, then for each of train, validation, and test:
images, capture events, and inclusive span in days from the first to the
last event day; the test group adds the in-sample gap, hours from the
last training or validation event to the first test event. Unknown
individuals have no training or validation data: every one of their
events, including those the upstream dataset labels as training, is a
test query, so all of their images are pooled under test. The last row
carries no
trailing row break; the manuscript writes \input{...} \\ like the other
fragments.

Run in the wolverines env:
    python preprocessing/make_table_individuals.py
"""
import datetime as dt
import json
import os

import numpy as np
from datasets import load_from_disk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'hugging_face_dataset', 'v2', 'data', 'reidentification_dataset')
CONFIG = os.path.join(HERE, 'results', 'feasible_individuals.json')
OUT = os.path.join(HERE, 'results', 'table_individuals.tex')

KNOWN = ['Turk', 'HLC20-H3', 'HFW12-F7', 'BDF10-M6', 'LH23-M1', 'Tex']
UNKNOWN = ['PA23-F1', 'Powder Paws', 'PA23-M2', 'PA23-M1']


def day(ymdh):
    s = str(ymdh)
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def stamp(ymdh):
    s = str(ymdh)
    return dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]))


def cells(ymdh):
    if len(ymdh) == 0:
        return '-- & -- & --'
    days = sorted({day(y) for y in ymdh})
    span = (days[-1] - days[0]).days + 1
    return f'{len(ymdh):,} & {len(set(ymdh))} & {span:,}'


with open(CONFIG) as f:
    cfg = json.load(f)
assert set(cfg['qualified_individuals']) == set(KNOWN), cfg['qualified_individuals']
val_idx = {int(i) for v in cfg['validation_indices'].values() for i in v['indices']}

tr = load_from_disk(os.path.join(DATA, 'train'))
te = load_from_disk(os.path.join(DATA, 'test'))
tr_id, tr_ymdh = np.array(tr['id']), np.array(tr['ymdh'])
te_id, te_ymdh = np.array(te['id']), np.array(te['ymdh'])
val_mask = np.zeros(len(tr_id), bool)
val_mask[list(val_idx)] = True

lines = []
for role, names in (('Known', KNOWN), ('Unknown', UNKNOWN)):
    if role == 'Unknown':
        lines.append('\\midrule')
    for ind in names:
        m_tr, m_te = tr_id == ind, te_id == ind
        if role == 'Known':
            train = cells(tr_ymdh[m_tr & ~val_mask])
            val = cells(tr_ymdh[m_tr & val_mask])
            test = cells(te_ymdh[m_te])
        else:
            train = val = '-- & -- & --'
            test = cells(np.concatenate([tr_ymdh[m_tr], te_ymdh[m_te]]))
        if role == 'Known':
            last_in = max(stamp(y) for y in tr_ymdh[m_tr])
            first_test = min(stamp(y) for y in te_ymdh[m_te])
            gap = f'{(first_test - last_in).total_seconds() / 3600:.1f}'
        else:
            gap = '--'
        lines.append(f'{ind} & {role} & {train} & {val} & {test} & {gap} \\\\')

lines[-1] = lines[-1].rstrip(' \\')
with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'wrote {OUT}')
print('\n'.join(lines))

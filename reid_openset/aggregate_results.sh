#!/bin/bash
cd /home/kdoherty/wolverines

# ---- Hygiene sweep figures/tables (existing) ----
python -u -m utils.reid_plotting --aggregate

# ---- Test eval: aggregate per-job JSONs into combined .csv and .md ----
python -u -c "
import json, glob, os

out_dir = 'reid_openset/tables'
os.makedirs(out_dir, exist_ok=True)

cols = ['backbone', 'metric', 'gallery_q', 'query_q', 'lr',
        'R@1', 'BA', 'KAR', 'URR',
        'emb_dim', 'samples_per_id', 'steps',
        'n_query', 'n_known', 'n_unknown']

rows = []
for path in sorted(glob.glob('reid_openset/*/results/test_eval/*.json')):
    with open(path) as f:
        data = json.load(f)
    cfg = data['config']
    for q_key in sorted(data['results']['by_quality']):
        q = data['results']['by_quality'][q_key]
        rows.append({
            'backbone':       cfg['backbone'],
            'metric':         cfg['metric'],
            'gallery_q':      f\"{cfg['gallery_threshold']:.2f}\",
            'query_q':        q_key.replace('q>=', ''),
            'lr':             cfg['learning_rate'],
            'R@1':            f\"{q['recall_at_1']:.4f}\",
            'BA':             f\"{q['balanced_accuracy']:.4f}\",
            'KAR':            f\"{q['known_accept_rate']:.4f}\",
            'URR':            f\"{q['unknown_reject_rate']:.4f}\",
            'emb_dim':        cfg['embedding_dim'],
            'samples_per_id': cfg['sample_size'],
            'steps':          cfg['best_step'],
            'n_query':        q['n_query'],
            'n_known':        q['n_known_individuals'],
            'n_unknown':      q['n_unknown_individuals'],
        })

if not rows:
    print('No test_eval JSONs found.')
    exit(0)

# CSV
csv_path = os.path.join(out_dir, 'test_eval.csv')
with open(csv_path, 'w') as f:
    f.write(','.join(cols) + '\n')
    for r in rows:
        f.write(','.join(str(r[c]) for c in cols) + '\n')

# Markdown
md_path = os.path.join(out_dir, 'test_eval.md')
with open(md_path, 'w') as f:
    f.write('| ' + ' | '.join(cols) + ' |\n')
    f.write('| ' + ' | '.join('---' for _ in cols) + ' |\n')
    for r in rows:
        f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')

print(f'Aggregated {len(rows)} rows from {len(set(r[\"backbone\"] for r in rows))} backbone(s)')
print(f'  CSV: {csv_path}')
print(f'  MD:  {md_path}')
print()
with open(md_path) as f:
    print(f.read())
"

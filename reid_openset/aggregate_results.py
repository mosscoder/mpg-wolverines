#!/usr/bin/env python
"""Aggregate test_eval JSONs into recall_at_1 and balanced_accuracy tables."""

import json
import glob
import os
import subprocess
import sys

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
    r1_cols = ['backbone', 'gallery_q', 'query_q', 'R@1',
               'lr', 'emb_dim', 'steps']
    r1_rows = []
    for data in all_data:
        cfg = data['config']
        if cfg['metric'] != 'recall':
            continue
        res = data['results']
        r1_rows.append({
            'backbone':  cfg['backbone'],
            'gallery_q': f"{cfg['gallery_threshold']:.2f}",
            'query_q':   f"{cfg['query_threshold']:.2f}",
            'R@1':       f"{res['recall_at_1']:.4f}",
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
        if cfg['metric'] != 'ba':
            continue
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


if __name__ == '__main__':
    main()

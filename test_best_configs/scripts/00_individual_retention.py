"""
Pre-run diagnostic: report which individuals are retained vs excluded
by each backbone's quality thresholds, with image counts per set.

Outputs:
  tables/individual_retention_retained.csv       — one row per (individual × backbone), train/test counts
  tables/individual_retention_excluded.csv       — individuals dropped and why, per backbone
  tables/individual_retention_summary.csv        — per-backbone totals + shared intersection
  tables/individual_retention_epoch_search.csv   — epoch search temporal split viability per backbone

Usage:
    python test_best_configs/scripts/00_individual_retention.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append('.')

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datasets import load_dataset
import datasets
datasets.config.NUM_PROC = 1

from utils.reid import (
    MODEL_CONFIGS, BATCH_K, MIN_P,
    load_best_hyperparams, build_metadata_cache,
    filter_training_pool_by_quality, load_feasibility_config,
    QUERY_QUALITY_THRESHOLDS,
)
from utils.reid_plotting import load_hygiene_results, find_best_epoch


MODELS = list(MODEL_CONFIGS.keys())
TABLES_DIR = 'test_best_configs/tables'


def extract_best_hygiene_config(model_name):
    """Extract best hygiene config by searching all combos (from sweep_best.py)."""
    config = MODEL_CONFIGS[model_name]
    results_dir = os.path.join(config['experiment_dir'], 'results')
    results = load_hygiene_results(results_dir)

    if len(results) == 0:
        raise ValueError(f"No hygiene results found for {model_name} in {results_dir}")

    gallery_sizes = sorted(results.get_unique('gallery_size'))
    gallery_thresholds = sorted(results.get_unique('threshold'))
    query_thresholds = [f'q>={q}' for q in QUERY_QUALITY_THRESHOLDS]

    best_global_r1 = -1
    best_config = None

    for gsize in gallery_sizes:
        for gal_thresh in gallery_thresholds:
            filtered = results.filter(threshold=gal_thresh, gallery_size=gsize)
            if len(filtered) == 0:
                continue
            all_histories = [r.get('epoch_history', []) for r in filtered]
            if not all_histories or not all_histories[0]:
                continue
            if 'query_quality_metrics' not in all_histories[0][0]:
                continue

            for q_thresh in query_thresholds:
                best_epoch, _, _ = find_best_epoch(
                    all_histories, criterion='recall', query_thresh=q_thresh
                )
                r1_values = []
                cos_thresholds = []
                for history in all_histories:
                    for h in history:
                        if h['epoch'] == best_epoch:
                            qm = h.get('query_quality_metrics', {})
                            if q_thresh in qm:
                                r1_values.append(qm[q_thresh]['recall_at_1'])
                            os_data = h.get('open_set', {})
                            tc = os_data.get('threshold_calibration', {})
                            if 'threshold' in tc:
                                cos_thresholds.append(tc['threshold'])
                            break

                if r1_values:
                    mean_r1 = np.mean(r1_values)
                    if mean_r1 > best_global_r1:
                        best_global_r1 = mean_r1
                        best_config = {
                            'best_epoch': best_epoch,
                            'gallery_threshold': gal_thresh,
                            'query_threshold': float(q_thresh.replace('q>=', '')),
                            'cosine_similarity_threshold': float(np.mean(cos_thresholds)) if cos_thresholds else 0.0,
                            'gallery_size': gsize,
                            'mean_r1': mean_r1,
                        }

    if best_config is None:
        raise ValueError(f"Could not find any valid hygiene config for {model_name}")
    return best_config


def create_temporal_val_split(train_metadata, val_fraction=0.1):
    """Split re-id pool into train/val by reserving last 10% of events per individual."""
    ymdh_arr = train_metadata['ymdh']
    train_indices_per_id = {}
    val_indices_per_id = {}

    for ind_id, indices in train_metadata['id_to_indices'].items():
        # Get unique ymdh values, sorted chronologically
        ymdh_values = sorted(set(ymdh_arr[i] for i in indices))
        n_val_events = max(1, int(len(ymdh_values) * val_fraction))
        val_ymdh = set(ymdh_values[-n_val_events:])

        train_idx = [i for i in indices if ymdh_arr[i] not in val_ymdh]
        val_idx = [i for i in indices if ymdh_arr[i] in val_ymdh]

        train_indices_per_id[ind_id] = train_idx
        val_indices_per_id[ind_id] = val_idx

    return train_indices_per_id, val_indices_per_id


def main():
    print("=" * 70)
    print("Individual Retention Diagnostic")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    train_hf = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    test_hf = load_dataset("kdoherty/wolverines", "reidentification", split="test")
    print(f"  Train: {len(train_hf)} images")
    print(f"  Test:  {len(test_hf)} images")

    train_meta = build_metadata_cache(train_hf)
    test_meta = build_metadata_cache(test_hf)

    feasibility_config = load_feasibility_config()
    excluded_entirely = feasibility_config.get('excluded_entirely', []) if feasibility_config else []

    # All individual IDs across both splits (excluding excluded_entirely)
    train_ids = set(train_meta['id_to_indices'].keys())
    test_ids = set(test_meta['id_to_indices'].keys())
    all_ids = sorted(train_ids | test_ids)
    candidate_ids = sorted((train_ids & test_ids) - set(excluded_entirely))

    print(f"\n  IDs in train only:      {sorted(train_ids - test_ids)}")
    print(f"  IDs in test only:       {sorted(test_ids - train_ids)}")
    print(f"  IDs in both:            {len(train_ids & test_ids)}")
    print(f"  Excluded entirely:      {excluded_entirely}")
    print(f"  Candidates (both, not excluded): {len(candidate_ids)}")

    # Extract best hygiene configs
    print("\nExtracting best hygiene configs...")
    backbone_configs = {}
    for mn in MODELS:
        backbone_configs[mn] = extract_best_hygiene_config(mn)
        cfg = backbone_configs[mn]
        print(f"  {mn}: gallery>={cfg['gallery_threshold']}, query>={cfg['query_threshold']}, "
              f"epoch={cfg['best_epoch']}, gallery_size={cfg['gallery_size']}")

    # --- Build retained sheet ---
    retained_rows = []
    excluded_rows = []
    viable_per_backbone = {}

    for mn in MODELS:
        cfg = backbone_configs[mn]
        gal_thresh = cfg['gallery_threshold']
        q_thresh = cfg['query_threshold']
        backbone_label = MODEL_CONFIGS[mn]['backbone_label'].replace('Frozen ', '')
        viable = []

        for ind_id in candidate_ids:
            train_indices = train_meta['id_to_indices'].get(ind_id, [])
            test_indices = test_meta['id_to_indices'].get(ind_id, [])

            train_total = len(train_indices)
            test_total = len(test_indices)

            train_filtered = filter_training_pool_by_quality(
                train_meta, train_indices, gal_thresh
            ) if train_indices else []
            test_filtered = filter_training_pool_by_quality(
                test_meta, test_indices, q_thresh
            ) if test_indices else []

            train_pass = len(train_filtered) >= BATCH_K
            test_pass = len(test_filtered) >= 1
            retained = train_pass and test_pass

            if retained:
                viable.append(ind_id)
                retained_rows.append({
                    'model': mn,
                    'backbone': backbone_label,
                    'gallery_threshold': gal_thresh,
                    'query_threshold': q_thresh,
                    'individual': ind_id,
                    'train_total': train_total,
                    'train_after_filter': len(train_filtered),
                    'test_total': test_total,
                    'test_after_filter': len(test_filtered),
                })
            else:
                reasons = []
                if not train_pass:
                    reasons.append(f'train<{BATCH_K} after gallery filter ({len(train_filtered)}/{train_total})')
                if not test_pass:
                    reasons.append(f'test<1 after query filter ({len(test_filtered)}/{test_total})')
                excluded_rows.append({
                    'model': mn,
                    'backbone': backbone_label,
                    'gallery_threshold': gal_thresh,
                    'query_threshold': q_thresh,
                    'individual': ind_id,
                    'train_total': train_total,
                    'train_after_filter': len(train_filtered),
                    'test_total': test_total,
                    'test_after_filter': len(test_filtered),
                    'reason': '; '.join(reasons),
                })

        viable_per_backbone[mn] = set(viable)

    # Add globally-excluded individuals to excluded sheet
    for ind_id in excluded_entirely:
        train_total = len(train_meta['id_to_indices'].get(ind_id, []))
        test_total = len(test_meta['id_to_indices'].get(ind_id, []))
        for mn in MODELS:
            backbone_label = MODEL_CONFIGS[mn]['backbone_label'].replace('Frozen ', '')
            cfg = backbone_configs[mn]
            excluded_rows.append({
                'model': mn,
                'backbone': backbone_label,
                'gallery_threshold': cfg['gallery_threshold'],
                'query_threshold': cfg['query_threshold'],
                'individual': ind_id,
                'train_total': train_total,
                'train_after_filter': '-',
                'test_total': test_total,
                'test_after_filter': '-',
                'reason': 'excluded_entirely (feasibility config)',
            })

    # Add individuals only in one split
    for ind_id in sorted(train_ids ^ test_ids):
        if ind_id in excluded_entirely:
            continue
        in_train = ind_id in train_ids
        in_test = ind_id in test_ids
        train_total = len(train_meta['id_to_indices'].get(ind_id, []))
        test_total = len(test_meta['id_to_indices'].get(ind_id, []))
        for mn in MODELS:
            backbone_label = MODEL_CONFIGS[mn]['backbone_label'].replace('Frozen ', '')
            cfg = backbone_configs[mn]
            reason = 'missing from test split' if not in_test else 'missing from train split'
            excluded_rows.append({
                'model': mn,
                'backbone': backbone_label,
                'gallery_threshold': cfg['gallery_threshold'],
                'query_threshold': cfg['query_threshold'],
                'individual': ind_id,
                'train_total': train_total,
                'train_after_filter': '-',
                'test_total': test_total,
                'test_after_filter': '-',
                'reason': reason,
            })

    # Shared intersection
    shared = sorted(set.intersection(*viable_per_backbone.values()))

    # --- Build summary sheet ---
    summary_rows = []
    for mn in MODELS:
        cfg = backbone_configs[mn]
        backbone_label = MODEL_CONFIGS[mn]['backbone_label'].replace('Frozen ', '')
        viable = viable_per_backbone[mn]

        # Count images for viable individuals
        train_imgs = 0
        test_imgs = 0
        for ind_id in viable:
            train_indices = train_meta['id_to_indices'].get(ind_id, [])
            test_indices = test_meta['id_to_indices'].get(ind_id, [])
            train_imgs += len(filter_training_pool_by_quality(
                train_meta, train_indices, cfg['gallery_threshold']
            ))
            test_imgs += len(filter_training_pool_by_quality(
                test_meta, test_indices, cfg['query_threshold']
            ))

        # Count images for shared individuals
        shared_train = 0
        shared_test = 0
        for ind_id in shared:
            train_indices = train_meta['id_to_indices'].get(ind_id, [])
            test_indices = test_meta['id_to_indices'].get(ind_id, [])
            shared_train += len(filter_training_pool_by_quality(
                train_meta, train_indices, cfg['gallery_threshold']
            ))
            shared_test += len(filter_training_pool_by_quality(
                test_meta, test_indices, cfg['query_threshold']
            ))

        summary_rows.append({
            'model': mn,
            'backbone': backbone_label,
            'gallery_threshold': cfg['gallery_threshold'],
            'query_threshold': cfg['query_threshold'],
            'viable_individuals': len(viable),
            'viable_train_images': train_imgs,
            'viable_test_images': test_imgs,
            'shared_individuals': len(shared),
            'shared_train_images': shared_train,
            'shared_test_images': shared_test,
        })

    df_retained = pd.DataFrame(retained_rows)
    df_excluded = pd.DataFrame(excluded_rows)
    df_summary = pd.DataFrame(summary_rows)

    # Print summary
    print(f"\nShared individuals across all backbones: {len(shared)}")
    print(f"  {shared}")
    print(f"\nSummary:")
    print(df_summary.to_string(index=False))
    print(f"\nRetained (per backbone):")
    if not df_retained.empty:
        for mn in MODELS:
            sub = df_retained[df_retained['model'] == mn]
            print(f"  {mn}: {len(sub)} individuals, "
                  f"{sub['train_after_filter'].sum()} train imgs, "
                  f"{sub['test_after_filter'].sum()} test imgs")
    print(f"\nExcluded (per backbone):")
    if not df_excluded.empty:
        for mn in MODELS:
            sub = df_excluded[df_excluded['model'] == mn]
            print(f"  {mn}: {len(sub)} entries")

    # Write CSVs
    os.makedirs(TABLES_DIR, exist_ok=True)
    for name, df in [('retained', df_retained), ('excluded', df_excluded), ('summary', df_summary)]:
        path = os.path.join(TABLES_DIR, f'individual_retention_{name}.csv')
        df.to_csv(path, index=False)
        print(f"\nSaved: {path}")

    # --- Epoch search retention ---
    print("\n" + "=" * 70)
    print("Epoch Search Retention (temporal val split)")
    print("=" * 70)

    train_indices_per_id, val_indices_per_id = create_temporal_val_split(train_meta)

    # All individuals in re-id pool (excluding excluded_entirely)
    reid_candidate_ids = sorted(set(train_meta['id_to_indices'].keys()) - set(excluded_entirely))

    epoch_search_rows = []
    for mn in MODELS:
        cfg = backbone_configs[mn]
        gal_thresh = cfg['gallery_threshold']
        q_thresh = cfg['query_threshold']
        backbone_label = MODEL_CONFIGS[mn]['backbone_label'].replace('Frozen ', '')

        print(f"\n  {mn} (gallery>={gal_thresh}, query>={q_thresh}):")

        for ind_id in reid_candidate_ids:
            all_indices = train_meta['id_to_indices'].get(ind_id, [])
            total_images = len(all_indices)

            # Count unique events
            ymdh_arr = train_meta['ymdh']
            all_ymdh = sorted(set(ymdh_arr[i] for i in all_indices))
            n_events = len(all_ymdh)

            # Temporal split counts
            train_after_split = len(train_indices_per_id.get(ind_id, []))
            val_after_split = len(val_indices_per_id.get(ind_id, []))

            # Val events count
            n_val_events = max(1, int(n_events * 0.1))

            # Quality-filtered counts
            train_filtered = filter_training_pool_by_quality(
                train_meta, train_indices_per_id.get(ind_id, []), gal_thresh
            ) if train_indices_per_id.get(ind_id, []) else []
            val_filtered = filter_training_pool_by_quality(
                train_meta, val_indices_per_id.get(ind_id, []), q_thresh
            ) if val_indices_per_id.get(ind_id, []) else []

            viable = len(train_filtered) >= BATCH_K and len(val_filtered) >= 1

            epoch_search_rows.append({
                'model': mn,
                'backbone': backbone_label,
                'gallery_threshold': gal_thresh,
                'query_threshold': q_thresh,
                'individual': ind_id,
                'total_images': total_images,
                'n_events': n_events,
                'val_events': n_val_events,
                'train_after_split': train_after_split,
                'val_after_split': val_after_split,
                'train_after_filter': len(train_filtered),
                'val_after_filter': len(val_filtered),
                'viable': viable,
            })

            status = "viable" if viable else "excluded"
            print(f"    {ind_id}: events={n_events}, train_filt={len(train_filtered)}, "
                  f"val_filt={len(val_filtered)} [{status}]")

    df_epoch_search = pd.DataFrame(epoch_search_rows)

    # Print epoch search summary per backbone
    print(f"\nEpoch search viability summary:")
    for mn in MODELS:
        sub = df_epoch_search[df_epoch_search['model'] == mn]
        n_viable = sub['viable'].sum()
        print(f"  {mn}: {n_viable}/{len(sub)} viable")

    path = os.path.join(TABLES_DIR, 'individual_retention_epoch_search.csv')
    df_epoch_search.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()

"""
Epoch selection via temporal validation split of the re-identification pool.

For each backbone, trains on the re-id pool (HF train split) minus a temporal
validation set (last 10% of capture events per individual), evaluates R@1 on
the val set each epoch, and saves the best epoch. This avoids contaminating
the HF test holdout when choosing the training epoch.

All individuals passing quality filtering are included (not just the 5 knowns
from the feasibility config).

Usage:
    python -u test_best_configs/scripts/01_best_epoch_search.py --idx $SLURM_ARRAY_TASK_ID --device gpu
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', '/data/hf_cache')

sys.path.append('.')

from datasets import load_dataset
import datasets
datasets.config.NUM_PROC = 1

from utils.dataset import set_all_seeds
from utils.reid import (
    MODEL_CONFIGS,
    ARCFACE_MARGIN, ARCFACE_SCALE, BATCH_K, MIN_P,
    create_arcface_model, create_transform_for_model,
    load_best_hyperparams, build_metadata_cache,
    filter_training_pool_by_quality,
    ArcFaceDataset, train_epoch_arcface, evaluate_recall_simple,
    count_trainable_parameters,
    QUERY_QUALITY_THRESHOLDS,
)
from utils.reid_plotting import (
    load_hygiene_results, find_best_epoch,
)
from utils.arcface import ArcFaceLoss, PKBatchSampler


MODELS = list(MODEL_CONFIGS.keys())  # ['dinov3', 'megadescriptor', 'bioclip2']


def extract_best_hygiene_config(model_name):
    """
    Extract the best hygiene configuration for a backbone by searching all
    (gallery_threshold x query_threshold x gallery_size) combos.

    Returns dict with best_epoch, gallery_threshold, query_threshold,
    cosine_similarity_threshold, gallery_size, mean_r1.
    """
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

    print(f"\nBest hygiene config for {model_name}:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

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


def build_epoch_search_datasets(train_hf, train_metadata, train_indices_per_id,
                                val_indices_per_id, gallery_threshold, query_threshold,
                                excluded):
    """
    Build train/val datasets for epoch search using temporal split and quality filtering.

    Works with ALL individuals (not just the 5 knowns from feasibility config).
    An individual is viable if:
      - len(train_filtered) >= BATCH_K
      - len(val_filtered) >= 1

    Returns: train_dataset, val_dataset, individual_to_class, viable_individuals, dataset_info
    """
    viable_individuals = []
    all_train_indices = []
    all_val_indices = []
    train_per_individual = {}
    val_per_individual = {}

    candidate_ids = sorted(set(train_indices_per_id.keys()) - set(excluded))

    print(f"\nBuilding epoch search datasets:")
    print(f"  Candidates (excluding {len(excluded)} excluded): {len(candidate_ids)}")
    print(f"  Gallery threshold: {gallery_threshold}, Query threshold: {query_threshold}")

    for ind_id in candidate_ids:
        train_candidates = train_indices_per_id.get(ind_id, [])
        val_candidates = val_indices_per_id.get(ind_id, [])

        train_filtered = filter_training_pool_by_quality(
            train_metadata, train_candidates, gallery_threshold
        )
        val_filtered = filter_training_pool_by_quality(
            train_metadata, val_candidates, query_threshold
        )

        if len(train_filtered) >= BATCH_K and len(val_filtered) >= 1:
            viable_individuals.append(ind_id)
            all_train_indices.extend(train_filtered)
            all_val_indices.extend(val_filtered)
            train_per_individual[ind_id] = len(train_filtered)
            val_per_individual[ind_id] = len(val_filtered)
            print(f"  {ind_id}: train={len(train_filtered)}, val={len(val_filtered)} [viable]")
        else:
            print(f"  {ind_id}: train={len(train_filtered)}, val={len(val_filtered)} [excluded]")

    print(f"\n  Viable individuals: {len(viable_individuals)}")
    print(f"  Total train: {len(all_train_indices)}, Total val: {len(all_val_indices)}")

    if len(viable_individuals) < MIN_P:
        raise ValueError(
            f"Not enough viable individuals ({len(viable_individuals)}) for PK sampling (need {MIN_P})"
        )

    individual_to_class = {ind: i for i, ind in enumerate(sorted(viable_individuals))}

    train_dataset = train_hf.select(all_train_indices)
    val_dataset = train_hf.select(all_val_indices)

    dataset_info = {
        'viable_individuals': viable_individuals,
        'n_viable': len(viable_individuals),
        'train_total': len(all_train_indices),
        'val_total': len(all_val_indices),
        'train_per_individual': train_per_individual,
        'val_per_individual': val_per_individual,
    }

    return train_dataset, val_dataset, individual_to_class, viable_individuals, dataset_info


def main():
    parser = argparse.ArgumentParser(description='Epoch search via temporal validation split')
    parser.add_argument('--idx', type=int, required=True, help='SLURM array task ID (0-2)')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs to search')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    args = parser.parse_args()

    if args.idx >= len(MODELS):
        print(f"Job {args.idx} has no work (only {len(MODELS)} backbones). Exiting.")
        return

    model_name = MODELS[args.idx]
    model_config = MODEL_CONFIGS[model_name]
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Epoch Search: {model_config['backbone_label']} - Job {args.idx}")
    print("=" * 80)

    # Step A: Load best hyperparams from optimization sweeps
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)

    # Step B: Extract best hygiene config for THIS backbone only
    print("\nExtracting best hygiene config...")
    my_hygiene = extract_best_hygiene_config(model_name)
    gallery_threshold = my_hygiene['gallery_threshold']
    query_threshold = my_hygiene['query_threshold']

    print(f"\nConfig for {model_name}:")
    print(f"  LR: {best_lr}, Size: {best_size}, Emb: {best_embedding_dim}")
    print(f"  Gallery thresh: {gallery_threshold}, Query thresh: {query_threshold}")
    print(f"  Max epochs: {args.epochs}")

    # Step C: Load HF train split
    print("\nLoading dataset...")
    train_hf = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    print(f"  Train split: {len(train_hf)} images")

    train_metadata = build_metadata_cache(train_hf)

    # Get excluded individuals from feasibility config
    from utils.reid import load_feasibility_config
    feasibility_config = load_feasibility_config()
    excluded = feasibility_config.get('excluded_entirely', []) if feasibility_config else []

    # Step D: Create temporal val split
    print("\nCreating temporal validation split (last 10% of events per individual)...")
    train_indices_per_id, val_indices_per_id = create_temporal_val_split(train_metadata)

    # Step E: Build epoch search datasets
    train_dataset, val_dataset, individual_to_class, viable_individuals, dataset_info = \
        build_epoch_search_datasets(
            train_hf, train_metadata, train_indices_per_id, val_indices_per_id,
            gallery_threshold, query_threshold, excluded
        )

    # Check output path
    output_dir = 'test_best_configs/results/01_best_epoch'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_epoch_search.json')

    if os.path.exists(output_path) and not args.overwrite:
        print(f"\nResult already exists: {output_path}")
        print("Use --overwrite to re-run.")
        return

    # Step F: Create model and training setup
    print(f"\nTraining {model_config['backbone_label']}...")
    set_all_seeds(0)

    model, emb_dim = create_arcface_model(
        model_name, embedding_dim=best_embedding_dim,
        image_size=best_size, device=device
    )
    transform = create_transform_for_model(model_name, size=best_size)

    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)

    pk_sampler = PKBatchSampler(
        labels=train_torch.get_labels(),
        p=min(MIN_P, len(viable_individuals)),
        k=BATCH_K,
        drop_last=True
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch,
        batch_sampler=pk_sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    criterion = ArcFaceLoss(
        num_classes=len(viable_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    head_params = count_trainable_parameters(model)
    arcface_params = count_trainable_parameters(criterion)
    trainable_params = {
        'head': head_params,
        'arcface': arcface_params,
        'total': head_params + arcface_params
    }

    print(f"  Trainable params - Head: {head_params:,}, ArcFace: {arcface_params:,}, Total: {trainable_params['total']:,}")

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=best_lr
    )

    # Step G: Training loop with val evaluation
    start_time = time.time()
    epoch_history = []
    best_r1 = -1
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        # Evaluate R@1 on temporal val set
        val_r1 = evaluate_recall_simple(
            model, train_dataset, val_dataset, individual_to_class,
            transform, device
        )

        if val_r1 > best_r1:
            best_r1 = val_r1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:3d}/{args.epochs}: Loss={train_loss:.4f}, Val R@1={val_r1:.4f} (best={best_r1:.4f} @ epoch {best_epoch})")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_r1': val_r1,
        })

    training_time = time.time() - start_time

    print(f"\nTraining complete in {training_time:.1f}s")
    print(f"Best Val R@1: {best_r1:.4f} at epoch {best_epoch}")

    # Step H: Save result JSON
    result = {
        'model_name': model_name,
        'backbone': model_config['backbone_label'],
        'config': {
            'learning_rate': best_lr,
            'image_size': best_size,
            'embedding_dim': best_embedding_dim,
            'max_epochs': args.epochs,
            'gallery_threshold': gallery_threshold,
            'query_threshold': query_threshold,
            'loss': 'ArcFace',
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'optimizer': 'AdamW',
            'batch_k': BATCH_K,
            'seed': 0,
            'val_split_method': 'temporal_last_10pct_events',
        },
        'dataset': {
            'train_total': len(train_dataset),
            'val_total': len(val_dataset),
            'viable_individuals': viable_individuals,
            'train_per_individual': dataset_info['train_per_individual'],
            'val_per_individual': dataset_info['val_per_individual'],
        },
        'results': {
            'best_epoch': best_epoch,
            'best_val_r1': best_r1,
        },
        'trainable_params': trainable_params,
        'epoch_history': epoch_history,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == '__main__':
    main()

"""
Final test set evaluation with best configs per backbone.

Trains one model per backbone (dinov3, megadescriptor, bioclip2) on the full
training split using optimal hyperparameters from prior optimization and hygiene
sweeps, then evaluates Recall@1 on the held-out test split.

Usage:
    python -u test_best_configs/scripts/sweep_best.py --idx $SLURM_ARRAY_TASK_ID --device gpu
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
    load_hygiene_results, find_best_epoch, compute_epoch_metric,
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


def select_shared_individuals(train_metadata, test_metadata, backbone_configs, excluded):
    """
    Select individuals viable across ALL 3 backbones.

    An individual is viable for a backbone if:
    - Has >= BATCH_K (8) training samples with pelage_score >= gallery_threshold
    - Has >= 1 test sample with pelage_score >= query_threshold

    Returns the intersection of viable individuals across all backbones.
    """
    # Get all candidate IDs present in both train and test
    train_ids = set(train_metadata['id_to_indices'].keys())
    test_ids = set(test_metadata['id_to_indices'].keys())
    candidate_ids = train_ids & test_ids
    candidate_ids -= set(excluded)

    print(f"\nCross-backbone individual selection:")
    print(f"  Candidates (in both train & test, not excluded): {len(candidate_ids)}")

    viable_per_backbone = {}
    for model_name, cfg in backbone_configs.items():
        gal_thresh = cfg['gallery_threshold']
        q_thresh = cfg['query_threshold']
        viable = []

        for ind_id in candidate_ids:
            # Check training samples above gallery threshold
            train_indices = train_metadata['id_to_indices'].get(ind_id, [])
            if train_indices:
                filtered = filter_training_pool_by_quality(
                    train_metadata, train_indices, gal_thresh
                )
                if len(filtered) < BATCH_K:
                    continue
            else:
                continue

            # Check test samples above query threshold
            test_indices = test_metadata['id_to_indices'].get(ind_id, [])
            if test_indices:
                filtered_test = filter_training_pool_by_quality(
                    test_metadata, test_indices, q_thresh
                )
                if len(filtered_test) < 1:
                    continue
            else:
                continue

            viable.append(ind_id)

        viable_per_backbone[model_name] = set(viable)
        print(f"  {model_name}: {len(viable)} viable (gal>={gal_thresh}, q>={q_thresh})")

    # Intersection across all backbones
    shared = set.intersection(*viable_per_backbone.values())
    shared = sorted(shared)

    print(f"  Shared across all backbones: {len(shared)}")
    print(f"  Individuals: {shared}")

    if len(shared) < MIN_P:
        raise ValueError(
            f"Not enough shared individuals ({len(shared)}) for PK sampling (need {MIN_P})"
        )

    return shared


def main():
    parser = argparse.ArgumentParser(description='Final test evaluation with best configs')
    parser.add_argument('--idx', type=int, required=True, help='SLURM array task ID (0-2)')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    args = parser.parse_args()

    if args.idx >= len(MODELS):
        print(f"Job {args.idx} has no work (only {len(MODELS)} backbones). Exiting.")
        return

    model_name = MODELS[args.idx]
    model_config = MODEL_CONFIGS[model_name]
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Final Test Evaluation: {model_config['backbone_label']} - Job {args.idx}")
    print("=" * 80)

    # Step A: Load best hyperparams from optimization sweeps
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)

    # Step B: Extract best hygiene configs for ALL backbones
    print("\nExtracting best hygiene configs for all backbones...")
    backbone_configs = {}
    for mn in MODELS:
        backbone_configs[mn] = extract_best_hygiene_config(mn)

    my_hygiene = backbone_configs[model_name]
    best_epoch = my_hygiene['best_epoch']
    gallery_threshold = my_hygiene['gallery_threshold']
    query_threshold = my_hygiene['query_threshold']
    cosine_sim_threshold = my_hygiene['cosine_similarity_threshold']
    gallery_size_from_hygiene = my_hygiene['gallery_size']

    print(f"\nOptimal config for {model_name}:")
    print(f"  LR: {best_lr}, Size: {best_size}, Emb: {best_embedding_dim}")
    print(f"  Epochs: {best_epoch}, Gallery thresh: {gallery_threshold}, Query thresh: {query_threshold}")
    print(f"  Cosine sim threshold: {cosine_sim_threshold:.4f}")

    # Step C: Load datasets
    print("\nLoading datasets...")
    train_hf = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    test_hf = load_dataset("kdoherty/wolverines", "reidentification", split="test")
    print(f"  Train: {len(train_hf)} images")
    print(f"  Test: {len(test_hf)} images")

    train_metadata = build_metadata_cache(train_hf)
    test_metadata = build_metadata_cache(test_hf)

    # Get excluded individuals from feasibility config
    from utils.reid import load_feasibility_config
    feasibility_config = load_feasibility_config()
    excluded = feasibility_config.get('excluded_entirely', []) if feasibility_config else []

    # Step C.5: Cross-backbone individual selection
    shared_individuals = select_shared_individuals(
        train_metadata, test_metadata, backbone_configs, excluded
    )

    # Check output path
    output_dir = 'test_best_configs/results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_test.json')

    if os.path.exists(output_path) and not args.overwrite:
        print(f"\nResult already exists: {output_path}")
        print("Use --overwrite to re-run.")
        return

    # Step D: Build training set (full, quality-filtered)
    print(f"\nBuilding training set (gallery_threshold >= {gallery_threshold})...")
    all_train_indices = []
    train_per_individual = {}

    for ind_id in shared_individuals:
        indices = train_metadata['id_to_indices'].get(ind_id, [])
        filtered = filter_training_pool_by_quality(train_metadata, indices, gallery_threshold)
        all_train_indices.extend(filtered)
        train_per_individual[ind_id] = len(filtered)
        print(f"  {ind_id}: {len(filtered)} training samples (from {len(indices)} total)")

    train_dataset = train_hf.select(all_train_indices)
    print(f"Total training samples: {len(train_dataset)}")

    # Step E: Build test set (quality-filtered)
    print(f"\nBuilding test set (query_threshold >= {query_threshold})...")
    all_test_indices = []
    test_per_individual = {}

    for ind_id in shared_individuals:
        indices = test_metadata['id_to_indices'].get(ind_id, [])
        if not indices:
            print(f"  {ind_id}: no test images, skipping")
            test_per_individual[ind_id] = 0
            continue
        filtered = filter_training_pool_by_quality(test_metadata, indices, query_threshold)
        all_test_indices.extend(filtered)
        test_per_individual[ind_id] = len(filtered)
        print(f"  {ind_id}: {len(filtered)} test samples (from {len(indices)} total)")

    test_dataset = test_hf.select(all_test_indices)
    print(f"Total test samples: {len(test_dataset)}")

    # Build class mapping
    individual_to_class = {ind: i for i, ind in enumerate(sorted(shared_individuals))}

    # Step F: Train
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
        p=min(MIN_P, len(shared_individuals)),
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
        num_classes=len(shared_individuals),
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

    start_time = time.time()
    epoch_history = []

    for epoch in range(best_epoch):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        # Evaluate R@1 on test set each epoch
        test_r1 = evaluate_recall_simple(
            model, train_dataset, test_dataset, individual_to_class,
            transform, device
        )

        print(f"Epoch {epoch+1:3d}/{best_epoch}: Loss={train_loss:.4f}, Test R@1={test_r1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_r1': test_r1,
        })

    training_time = time.time() - start_time
    final_r1 = epoch_history[-1]['test_r1'] if epoch_history else 0.0

    print(f"\nTraining complete in {training_time:.1f}s")
    print(f"Final Test R@1: {final_r1:.4f}")

    # Step G: Save result JSON
    result = {
        'model_name': model_name,
        'backbone': model_config['backbone_label'],
        'config': {
            'learning_rate': best_lr,
            'image_size': best_size,
            'embedding_dim': best_embedding_dim,
            'epochs': best_epoch,
            'gallery_threshold': gallery_threshold,
            'query_threshold': query_threshold,
            'cosine_similarity_threshold': cosine_sim_threshold,
            'gallery_size_from_hygiene': gallery_size_from_hygiene,
            'loss': 'ArcFace',
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'optimizer': 'AdamW',
            'batch_k': BATCH_K,
            'seed': 0,
        },
        'dataset': {
            'train_total': len(train_dataset),
            'test_total': len(test_dataset),
            'shared_individuals': shared_individuals,
            'train_per_individual': train_per_individual,
            'test_per_individual': test_per_individual,
        },
        'results': {
            'test_recall_at_1': final_r1,
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

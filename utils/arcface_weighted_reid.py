"""
Shared utilities for Weighted ArcFace open-set experiments across all backbones.

Parallel to utils/adaface_reid.py but replaces QualityAdaFaceLoss with standard
ArcFace loss + per-sample weighting based on pelage quality.

Weight formula: weight = 1 + (quality * alpha)
- alpha=0 -> weight=1 for all (unweighted baseline, same as standard ArcFace)
- alpha=1 -> weights in [1, 2] (moderate quality emphasis)
- alpha=2 -> weights in [1, 3] (strong quality emphasis)

Opt scripts (LR, embedding_dim sweeps) use alpha=0. Hygiene sweep varies alpha.
"""

import os
import sys
import json
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.dataset import set_all_seeds

# Re-export from submodules for backward compatibility
from utils.reid_config import (  # noqa: F401
    MODEL_CONFIGS, ARCFACE_MARGIN, ARCFACE_SCALE, TRAIN_BATCH_SIZE,
    TARGET_SAMPLES_PER_INDIVIDUAL, QUERY_QUALITY_THRESHOLDS,
    EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    create_arcface_model, create_reid_transform, create_transform_for_model,
)
from utils.reid_data import (  # noqa: F401
    ArcFaceDataset, RareIndividualsDataset,
    load_feasibility_config, load_reidentification_dataset,
    load_reidentification_test_dataset, build_metadata_cache,
    create_ymdh_split_dataset, filter_training_pool_by_quality,
    get_rare_individual_indices, create_filtered_gallery_dataset,
    get_qualified_individuals,
    get_event_index_map, subsample_to_match_filtered,
)
from utils.reid_evaluation import (  # noqa: F401
    evaluate_recall_simple, compute_rare_embeddings, compute_validation_loss,
    compute_cosine_similarity,
    compute_open_set_metrics_per_individual_threshold,
    evaluate_recall_with_openset, compute_arcface_center_thresholds,
)


# ============================================================================
# Weighted ArcFace experiment directory mapping
# ============================================================================

def _weighted_experiment_dir(model_name):
    """Return arcface_weighted experiment dir for a model (parallel to reid_openset/)."""
    config = MODEL_CONFIGS[model_name]
    # reid_openset/dinov3 -> arcface_weighted/dinov3
    original = config["experiment_dir"]
    return original.replace("reid_openset/", "arcface_weighted/", 1)


# ============================================================================
# Training utilities
# ============================================================================

def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_weighted(model, train_loader, optimizer, criterion, device, alpha):
    """Train for one epoch with weighted ArcFace loss.

    Args:
        model: The model to train
        train_loader: DataLoader yielding (images, labels, quality) tuples
        optimizer: Optimizer
        criterion: ArcFaceLoss instance
        device: Device string or torch.device
        alpha: Quality weighting strength. 0 = unweighted (standard ArcFace),
               >0 = weight = 1 + quality * alpha
    """
    model.train()
    criterion.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        quality = batch[2].to(device)

        embeddings = model(images)

        optimizer.zero_grad()
        if alpha > 0:
            weights = 1.0 + quality * alpha
            loss = criterion(embeddings, labels, sample_weights=weights)
        else:
            loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================================
# Hyperparameter loading
# ============================================================================

def _select_best_param(results_dir, glob_pattern, param_key):
    """Select best param value using cross-seed best-epoch mean R@1.

    Groups results by param value, for each group averages test_recall_at_1
    across seeds at each epoch, picks the epoch with highest mean, and
    returns the param value with the best cross-seed score.
    """
    files = glob.glob(os.path.join(results_dir, glob_pattern))
    if not files:
        return None, 0.0

    # Group by param value
    groups = {}
    for f in files:
        with open(f, 'r') as fp:
            result = json.load(fp)
        value = result['config'][param_key]
        groups.setdefault(value, []).append(result)

    best_value = None
    best_score = 0.0
    for value, results in groups.items():
        histories = [r['results']['epoch_history'] for r in results]
        n_epochs = len(histories[0])
        best_mean = 0.0
        for i in range(n_epochs):
            mean_r1 = sum(h[i]['test_recall_at_1'] for h in histories) / len(histories)
            if mean_r1 > best_mean:
                best_mean = mean_r1
        if best_mean > best_score:
            best_score = best_mean
            best_value = value

    return best_value, best_score


def load_best_hyperparams(model_name):
    """
    Load best LR, image size, and embedding dim from Weighted ArcFace optimization sweep results.
    Falls back to defaults if results not found.

    Returns: (best_lr, best_size, best_embedding_dim)
    """
    config = MODEL_CONFIGS[model_name]
    experiment_dir = _weighted_experiment_dir(model_name)

    best_lr = config["default_lr"]
    best_size = config["native_size"]
    best_embedding_dim = 128  # default

    # LR selection
    lr_results_dir = os.path.join(experiment_dir, 'results/opt/lr')
    selected_lr, lr_score = _select_best_param(lr_results_dir, 'lr=*_seed=*.json', 'learning_rate')
    if selected_lr is not None:
        best_lr = selected_lr
        print(f"Loaded best LR from sweep: {best_lr} (cross-seed R@1={lr_score:.4f})")
    else:
        print(f"No LR sweep results found, using default: {best_lr}")

    print(f"Image size (native): {best_size}")

    # Embedding dim selection
    emb_results_dir = os.path.join(experiment_dir, 'results/opt/embedding_dim')
    selected_emb, emb_score = _select_best_param(emb_results_dir, 'embedding_dim=*_seed=*.json', 'embedding_dim')
    if selected_emb is not None:
        best_embedding_dim = selected_emb
        print(f"Loaded best embedding dim from sweep: {best_embedding_dim} (cross-seed R@1={emb_score:.4f})")
    else:
        print(f"No embedding dim sweep results found, using default: {best_embedding_dim}")

    return best_lr, best_size, best_embedding_dim


# ============================================================================
# Hygiene sweep: job distribution and training
# ============================================================================

ALPHAS = [0, 1, 2]
GALLERY_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def get_job_combinations(job_idx, max_jobs=24):
    """Map job index to list of (alpha, gallery_size, seed) tuples."""
    all_combinations = []
    for alpha in ALPHAS:
        for gallery_size in GALLERY_SIZES:
            for seed in SEEDS:
                all_combinations.append((alpha, gallery_size, seed))

    total = len(all_combinations)
    configs_per_job = total // max_jobs
    remainder = total % max_jobs

    if job_idx >= max_jobs or total == 0:
        return []

    if job_idx < remainder:
        start = job_idx * (configs_per_job + 1)
        end = start + configs_per_job + 1
    else:
        start = remainder * (configs_per_job + 1) + (job_idx - remainder) * configs_per_job
        end = start + configs_per_job

    if start >= total:
        return []

    return all_combinations[start:min(end, total)]


def train_single_config(model_name, alpha, gallery_size, seed, args,
                         dataset, config, metadata_cache,
                         learning_rate, image_size, embedding_dim, epochs=50):
    """Train one hygiene sweep configuration and return results."""
    from utils.arcface import ArcFaceLoss, BalancedBatchSampler
    from utils.training import check_result_exists

    model_config = MODEL_CONFIGS[model_name]

    set_all_seeds(seed)

    filename = f"alpha={alpha}_gallery={gallery_size}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)

    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: alpha={alpha}, gallery_size={gallery_size}, seed={seed}")
    print(f"{'='*60}")

    # Get qualified individuals from config (preprocessing already enforces criteria)
    feasible_individuals = config.get('qualified_individuals', [])
    promoted_individuals = config.get('promoted_to_rare', [])
    excluded_individuals = config.get('excluded_entirely', [])

    if len(feasible_individuals) < 2:
        print(f"Not enough individuals ({len(feasible_individuals)}) for training (need >= 2)")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create filtered gallery dataset — always threshold=0.0 (no quality filtering)
    train_dataset, val_dataset, individual_to_class, dataset_info = create_filtered_gallery_dataset(
        dataset, feasible_individuals, gallery_size, 0.0, seed,
        metadata_cache, config
    )

    if train_dataset is None or len(train_dataset) == 0:
        print(f"No training data available after filtering")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=embedding_dim,
                                           image_size=image_size, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_transform_for_model(model_name, size=image_size)
    train_torch_dataset = ArcFaceDataset(train_dataset, transform, individual_to_class)

    # Create balanced batch sampler
    sampler = BalancedBatchSampler(
        labels=train_torch_dataset.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create ArcFace loss (standard — weighting handled in train loop)
    criterion = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(criterion)
    trainable_params = {
        'head': head_params,
        'arcface': arcface_params,
        'total': head_params + arcface_params
    }

    print(f"  Model parameters (trainable):")
    print(f"  Projection head: {trainable_params['head']:,}")
    print(f"  ArcFace centers: {trainable_params['arcface']:,}")
    print(f"  Total: {trainable_params['total']:,}")

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=learning_rate
    )

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_recall_epoch = 0
    best_balanced_accuracy = 0.0
    best_ba_epoch = 0
    best_harmonic_mean = 0.0
    best_hm_epoch = 0

    for epoch in range(epochs):
        train_loss = train_epoch_weighted(model, train_loader, optimizer, criterion,
                                          device, alpha=alpha)

        query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
            model, train_dataset, val_dataset, individual_to_class, transform, device,
            dataset, feasible_individuals, metadata_cache,
            criterion=criterion, embedding_dim=emb_dim,
            promoted_individuals=promoted_individuals,
            excluded_individuals=excluded_individuals
        )

        recall_1 = query_quality_metrics["q>=0.0"]["recall_at_1"]

        if recall_1 > best_recall:
            best_recall = recall_1
            best_recall_epoch = epoch + 1

        by_quality = open_set_metrics.get('by_quality', {})
        ba_q0 = by_quality.get('q>=0.0', {}).get('balanced_accuracy', 0.0)
        if ba_q0 > best_balanced_accuracy:
            best_balanced_accuracy = ba_q0
            best_ba_epoch = epoch + 1

        if (recall_1 + ba_q0) > 0:
            hm = 2 * recall_1 * ba_q0 / (recall_1 + ba_q0)
        else:
            hm = 0.0
        if hm > best_harmonic_mean:
            best_harmonic_mean = hm
            best_hm_epoch = epoch + 1

        thresh_mean = by_quality.get('q>=0.0', {}).get('cosine_threshold', 0.0)

        print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, thresh={thresh_mean:.3f}")

        for q_thresh in QUERY_QUALITY_THRESHOLDS:
            q_key = f"q>={q_thresh}"
            r1 = query_quality_metrics[q_key]['recall_at_1']
            count = query_quality_metrics[q_key]['count']
            ba_data = by_quality.get(q_key, {})
            ba = ba_data.get('balanced_accuracy', 0.0)
            kar = ba_data.get('known_accept_rate', 0.0)
            urr = ba_data.get('unknown_reject_rate', 0.0)
            n_k_ind = ba_data.get('n_known_individuals', 0)
            n_u_ind = ba_data.get('n_unknown_individuals', 0)
            print(f"  {q_key}: R@1={r1:.4f} (n={count:3d}), BA={ba:.4f} (K={kar:.2f}[{n_k_ind}], U={urr:.2f}[{n_u_ind}])")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'query_quality_metrics': query_quality_metrics,
            'open_set': open_set_metrics
        })

    training_time = time.time() - start_time

    result = {
        'config': {
            'alpha': alpha,
            'gallery_size': gallery_size,
            'seed': seed,
            'loss': 'WeightedArcFace',
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'embedding_dim': emb_dim,
            'optimizer': 'AdamW',
            'scheduler': 'None (fixed LR)',
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': TRAIN_BATCH_SIZE,
            'backbone': model_config['backbone_label'],
            'score_normalization': 'Raw Cosine'
        },
        'trainable_params': trainable_params,
        'dataset': {
            'individuals': feasible_individuals,
            'gallery_samples_per_individual': dataset_info['gallery_samples_per_individual'],
            'eligible_pool_per_individual': dataset_info['eligible_pool_per_individual'],
            'query_samples_per_individual': dataset_info['query_samples_per_individual'],
            'total_gallery_size': len(train_dataset),
            'total_query_size': len(val_dataset)
        },
        'epoch_history': epoch_history,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx,
            'transform': f'resize_{image_size}_{model_name}_norm',
            'best_recall_epoch': best_recall_epoch,
            'best_recall_at_1': best_recall,
            'best_ba_epoch': best_ba_epoch,
            'best_balanced_accuracy': best_balanced_accuracy,
            'best_hm_epoch': best_hm_epoch,
            'best_harmonic_mean': best_harmonic_mean,
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best R@1 epoch: {best_recall_epoch}, R@1={best_recall:.4f}")
    print(f"Best BA epoch:  {best_ba_epoch}, BA={best_balanced_accuracy:.4f}")
    print(f"Best H-Mean epoch: {best_hm_epoch}, H-Mean={best_harmonic_mean:.4f} (recommended)")

    return filename


def run_hygiene_sweep(model_name, args):
    """
    Main entry point for running a Weighted ArcFace hygiene sweep experiment.

    Loads dataset, config, best hyperparams, distributes work across SLURM jobs,
    and trains each configuration.
    """
    config = MODEL_CONFIGS[model_name]

    print("=" * 80)
    print(f"Open-Set {config['backbone_label']} + Weighted ArcFace + Raw Cosine Gallery Hygiene Sweep - Job {args.idx}")
    print("=" * 80)

    # Load best hyperparameters
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)

    # Load datasets and config
    print("\nLoading datasets...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    feasibility_config = load_feasibility_config()

    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    total_combinations = len(ALPHAS) * len(GALLERY_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Model: {config['backbone_label']} + Projection Head ({best_embedding_dim}-d) + Weighted ArcFace")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE}) + quality weighting")
    print(f"  Score Normalization: Raw Cosine Similarity (L2-normalized)")
    print(f"  Optimizer: AdamW (lr={best_lr}, default settings)")
    print(f"  Embedding: {best_embedding_dim}-d (trainable projection)")
    print(f"  Image size: {best_size}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")
    print(f"  Open-set evaluation: Enabled (rare individuals, raw cosine scores)")

    combinations = get_job_combinations(args.idx)

    if not combinations:
        print(f"\nNo combinations assigned to job {args.idx}")
        return

    print(f"\nJob {args.idx} processing {len(combinations)} configurations:")
    for alpha, gallery_size, seed in combinations[:5]:
        print(f"  alpha={alpha}, gallery_size={gallery_size}, seed={seed}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more")

    results_summary = []
    for i, (alpha, gallery_size, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(
                model_name, alpha, gallery_size, seed, args,
                dataset, feasibility_config, metadata_cache,
                learning_rate=best_lr, image_size=best_size,
                embedding_dim=best_embedding_dim
            )
            if result:
                results_summary.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"Job {args.idx} completed!")
    print(f"Successfully processed {len(results_summary)} configurations")
    print(f"Results saved to: {args.output_dir}")


# ============================================================================
# Opt sweep shared training logic
# ============================================================================

def run_opt_training(model_name, sweep_param_name, sweep_param_value, args,
                     dataset, config, metadata_cache,
                     learning_rate=None, image_size=None, embedding_dim=128,
                     epochs=20, seed=0):
    """
    Shared training loop for all opt sweep scripts (LR, embedding_dim).

    Uses standard ArcFace loss with alpha=0 (no quality weighting).
    """
    from utils.arcface import ArcFaceLoss, BalancedBatchSampler

    model_config = MODEL_CONFIGS[model_name]
    if learning_rate is None:
        learning_rate = model_config["default_lr"]
    if image_size is None:
        image_size = model_config["native_size"]

    set_all_seeds(seed)

    # Output path
    if sweep_param_name == 'lr':
        filename = f"lr={sweep_param_value:.6f}_seed={seed}.json"
    elif sweep_param_name == 'embedding_dim':
        filename = f"embedding_dim={sweep_param_value}_seed={seed}.json"
    else:
        filename = f"{sweep_param_name}={sweep_param_value}_seed={seed}.json"

    output_path = os.path.join(args.output_dir, filename)

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: {sweep_param_name}={sweep_param_value}")
    print(f"{'='*60}")

    # Get valid individuals
    qualified_individuals = get_qualified_individuals(config)
    if qualified_individuals is None:
        return None
    print(f"Using {len(qualified_individuals)} individuals: {', '.join(qualified_individuals)}")

    # Create temporal split dataset
    train_dataset, test_dataset, individual_to_class, dataset_info = create_ymdh_split_dataset(
        dataset, qualified_individuals, metadata_cache, seed=seed
    )

    if train_dataset is None or len(train_dataset) == 0:
        print("No training data available")
        return None
    if test_dataset is None or len(test_dataset) == 0:
        print("No test data available")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=embedding_dim,
                                           image_size=image_size, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_transform_for_model(model_name, size=image_size)
    train_torch_dataset = ArcFaceDataset(train_dataset, transform, individual_to_class)

    # Create balanced batch sampler
    sampler = BalancedBatchSampler(
        labels=train_torch_dataset.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create standard ArcFace loss (no quality weighting for opt sweeps)
    criterion = ArcFaceLoss(
        num_classes=len(qualified_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(criterion)
    print(f"  Trainable params: head={head_params:,}, arcface={arcface_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=learning_rate
    )

    # Training loop — alpha=0 (standard ArcFace, no weighting)
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_epoch_weighted(model, train_loader, optimizer, criterion,
                                          device, alpha=0)

        recall_at_1 = evaluate_recall_simple(
            model, train_dataset, test_dataset, individual_to_class, transform, device
        )

        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:3d}/{epochs}: loss={train_loss:.4f}, test_R@1={recall_at_1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_recall_at_1': recall_at_1
        })

    training_time = time.time() - start_time

    # Prepare result
    result = {
        'config': {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'seed': seed,
            'backbone': model_config['backbone_label'],
            'loss': 'ArcFace',
            'target_samples_per_individual': TARGET_SAMPLES_PER_INDIVIDUAL,
            'image_size': image_size,
            'embedding_dim': embedding_dim,
        },
        'dataset': {
            'individuals': qualified_individuals,
            'train_samples_per_individual': dataset_info['train_samples_per_individual'],
            'test_samples_per_individual': dataset_info['test_samples_per_individual'],
            'total_train': dataset_info['total_train'],
            'total_test': dataset_info['total_test'],
            'split_method': 'ymdh_temporal_50_50_train_priority'
        },
        'results': {
            'best_epoch': best_epoch,
            'best_test_recall_at_1': best_recall,
            'epoch_history': epoch_history
        },
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best R@1: {best_recall:.4f} at epoch {best_epoch}")

    return filename



# ============================================================================
# Final test evaluation
# ============================================================================

def load_best_hygiene_config(model_name, criterion='harmonic_mean', alpha_filter=None):
    """
    Read all hygiene sweep results for a backbone and select the best operating point.

    Groups by (alpha, gallery_size), extracts best epoch per seed by criterion,
    averages across seeds, returns the (alpha, gallery_size, epoch) with highest mean.

    Args:
        model_name: Key in MODEL_CONFIGS
        criterion: 'recall', 'balanced_accuracy', or 'harmonic_mean'
        alpha_filter: None = any alpha, 'weighted' = alpha > 0 only

    Returns:
        dict with {alpha, gallery_size, best_epoch, score}
    """
    experiment_dir = _weighted_experiment_dir(model_name)
    results_dir = os.path.join(experiment_dir, 'results', 'hygiene')

    files = glob.glob(os.path.join(results_dir, 'alpha=*_gallery=*_seed=*.json'))
    if not files:
        raise FileNotFoundError(f"No hygiene results found in {results_dir}")

    # Load all results
    results = []
    for f in files:
        with open(f, 'r') as fp:
            results.append(json.load(fp))

    # Group by (alpha, gallery_size)
    groups = defaultdict(list)
    for r in results:
        key = (r['config']['alpha'], r['config']['gallery_size'])
        groups[key].append(r)

    # Optionally filter by alpha
    if alpha_filter == 'weighted':
        groups = {k: v for k, v in groups.items() if k[0] > 0}
        if not groups:
            raise ValueError(f"No hygiene results with alpha > 0 in {results_dir}")

    best_overall_score = -1.0
    best_config = None

    # For alpha=0, only evaluate at q>=0.0
    if alpha_filter is None:
        q_thresholds_to_search = QUERY_QUALITY_THRESHOLDS
    else:
        q_thresholds_to_search = QUERY_QUALITY_THRESHOLDS

    for (alpha, gallery_size), group_results in groups.items():
        for q_thresh in q_thresholds_to_search:
            q_key = f"q>={q_thresh}"

            all_histories = [r['epoch_history'] for r in group_results]

            eval_epochs = []
            for entry in all_histories[0]:
                if 'query_quality_metrics' in entry:
                    eval_epochs.append(entry['epoch'])

            best_mean = -1.0
            best_epoch_num = eval_epochs[0] if eval_epochs else 1

            for epoch_num in eval_epochs:
                scores = []
                for history in all_histories:
                    for entry in history:
                        if entry['epoch'] == epoch_num:
                            if criterion == 'recall':
                                s = entry.get('query_quality_metrics', {}).get(q_key, {}).get('recall_at_1')
                            elif criterion == 'balanced_accuracy':
                                s = entry.get('open_set', {}).get('by_quality', {}).get(q_key, {}).get('balanced_accuracy')
                            else:
                                raise ValueError(f"Unknown criterion: {criterion}")
                            if s is not None:
                                scores.append(s)
                            break

                if scores:
                    epoch_mean = np.mean(scores)
                    if epoch_mean > best_mean:
                        best_mean = epoch_mean
                        best_epoch_num = epoch_num

            # Extract per-individual thresholds at best epoch
            all_per_individual_thresholds = defaultdict(list)
            cosine_thresholds = []
            for history in all_histories:
                for entry in history:
                    if entry['epoch'] == best_epoch_num:
                        open_set = entry.get('open_set', {})
                        thresh_cal = open_set['threshold_calibration']

                        for name, thresh in thresh_cal['per_individual'].items():
                            all_per_individual_thresholds[name].append(thresh)
                        cosine_thresholds.append(thresh_cal['global_threshold'])
                        break

            mean_per_individual = {name: float(np.mean(vals))
                                   for name, vals in all_per_individual_thresholds.items()}
            mean_cosine_thresh = float(np.mean(cosine_thresholds)) if cosine_thresholds else None

            if best_mean > best_overall_score:
                best_overall_score = best_mean
                best_config = {
                    'alpha': alpha,
                    'gallery_size': gallery_size,
                    'query_quality_threshold': q_thresh,
                    'best_epoch': best_epoch_num,
                    'score': float(best_mean),
                    'criterion': criterion,
                    'n_seeds': len(group_results),
                    'cosine_threshold': mean_cosine_thresh,
                    'per_individual_thresholds': mean_per_individual,
                }

    print(f"Best hygiene config ({criterion}): alpha={best_config['alpha']}, "
          f"gallery_size={best_config['gallery_size']}, "
          f"query_q>={best_config['query_quality_threshold']}, "
          f"epoch={best_config['best_epoch']}, "
          f"score={best_config['score']:.4f} (n={best_config['n_seeds']} seeds)")

    return best_config


# Each task selects its own optimal (alpha, gallery_size, epoch) from the
# hygiene sweep using the criterion and alpha constraint that match its goal.
#
# "Filtered" tasks consider any alpha (including 0 = unweighted baseline).
# "Weighted" tasks only consider alpha > 0 (quality-weighted loss).
TEST_TASKS = {
    'closed_filtered': {'criterion': 'recall',            'alpha_filter': None},        # idx 0
    'closed_weighted': {'criterion': 'recall',            'alpha_filter': 'weighted'},   # idx 1
    'open_filtered':   {'criterion': 'balanced_accuracy', 'alpha_filter': None},        # idx 2
    'open_weighted':   {'criterion': 'balanced_accuracy', 'alpha_filter': 'weighted'},   # idx 3
}


def run_final_test(model_name, args, seed, task_name):
    """
    Final test evaluation using the best operating point from Weighted ArcFace hygiene sweep.

    Gallery always uses threshold=0.0 (all data). Quality weighting is in the loss.

    Args:
        model_name: Key in MODEL_CONFIGS
        args: Argparse namespace
        seed: Random seed for gallery sampling and training
        task_name: Key in TEST_TASKS
    """
    from utils.arcface import ArcFaceLoss, BalancedBatchSampler
    from utils.training import check_result_exists

    task_cfg = TEST_TASKS[task_name]
    criterion = task_cfg['criterion']
    alpha_filter = task_cfg['alpha_filter']

    model_config = MODEL_CONFIGS[model_name]
    experiment_dir = _weighted_experiment_dir(model_name)

    # Load best hygiene config
    if alpha_filter is not None:
        # For weighted tasks, load both the unconstrained and constrained configs
        unconstrained_config = load_best_hygiene_config(model_name, criterion=criterion,
                                                         alpha_filter=None)
        best_config = load_best_hygiene_config(model_name, criterion=criterion,
                                                alpha_filter=alpha_filter)
        best_alpha = best_config['alpha']
        best_epoch = best_config['best_epoch']
        query_quality_threshold = best_config['query_quality_threshold']
    else:
        best_config = load_best_hygiene_config(model_name, criterion=criterion,
                                                alpha_filter=alpha_filter)
        best_alpha = best_config['alpha']
        best_epoch = best_config['best_epoch']
        query_quality_threshold = best_config['query_quality_threshold']

    set_all_seeds(seed)

    # Output
    output_dir = os.path.join(experiment_dir, 'results', 'test')
    filename = f"test_seed={seed}_{task_name}.json"
    output_path = os.path.join(output_dir, filename)

    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Final Test Evaluation: seed={seed}, task={task_name}")
    print(f"  criterion={criterion}, alpha={best_alpha}, "
          f"query_q>={query_quality_threshold}, epochs={best_epoch}")
    print(f"  gallery: threshold=0.0 (all data, quality handled by loss weighting)")
    print(f"{'='*60}")

    # Load best hyperparameters
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_reidentification_dataset()
    train_metadata_cache = build_metadata_cache(train_dataset)
    test_dataset = load_reidentification_test_dataset()
    test_metadata_cache = build_metadata_cache(test_dataset)

    feasibility_config = load_feasibility_config()
    if not feasibility_config:
        print("Failed to load feasibility config")
        return None

    feasible_individuals = feasibility_config.get('qualified_individuals', [])
    promoted_individuals = feasibility_config.get('promoted_to_rare', [])
    excluded_individuals = feasibility_config.get('excluded_entirely', [])

    if len(feasible_individuals) < 2:
        print(f"Not enough individuals ({len(feasible_individuals)}) for training (need >= 2)")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create gallery from TRAIN split — always threshold=0.0 (no quality filtering)
    id_to_indices = train_metadata_cache['id_to_indices']
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}

    all_gallery_indices = []
    gallery_info = {}

    for ind_id in feasible_individuals:
        all_ind_indices = list(id_to_indices.get(ind_id, []))

        # Use ALL training images (no quality filtering)
        sampled = all_ind_indices
        gallery_info[ind_id] = {
            'sampled': len(sampled),
            'total': len(all_ind_indices),
            'mode': 'all_data',
        }
        print(f"  {ind_id}: {len(sampled)} gallery (all data, threshold=0.0)")

        all_gallery_indices.extend(sampled)

    gallery_dataset = train_dataset.select(all_gallery_indices)

    # Query (known): ALL test images for gallery-eligible individuals
    test_id_to_indices = test_metadata_cache['id_to_indices']
    all_query_indices = []
    query_info = {}
    for ind_id in feasible_individuals:
        test_indices = test_id_to_indices.get(ind_id, [])
        all_query_indices.extend(test_indices)
        query_info[ind_id] = len(test_indices)

    query_dataset = test_dataset.select(all_query_indices)
    print(f"\nTotal: {len(all_gallery_indices)} gallery, {len(all_query_indices)} query (known)")

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    # Create transforms and training dataset
    transform = create_transform_for_model(model_name, size=best_size)
    train_torch_dataset = ArcFaceDataset(gallery_dataset, transform, individual_to_class)

    # Create balanced batch sampler
    sampler = BalancedBatchSampler(
        labels=train_torch_dataset.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create standard ArcFace loss (weighting applied in train loop)
    arcface_loss = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(arcface_loss.parameters()),
        lr=best_lr
    )

    # Train for exactly best_epoch epochs
    print(f"\nTraining for {best_epoch} epochs (alpha={best_alpha})...")
    start_time = time.time()
    epoch_history = []

    for epoch in range(best_epoch):
        train_loss = train_epoch_weighted(model, train_loader, optimizer, arcface_loss,
                                          device, alpha=best_alpha)
        print(f"Epoch {epoch+1:3d}/{best_epoch}: Loss={train_loss:.4f}")
        epoch_history.append({'epoch': epoch + 1, 'train_loss': train_loss})

    training_time = time.time() - start_time

    # Evaluate on FULL test split
    print("\nEvaluating on test split...")
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    # Gallery embeddings
    gallery_torch = ArcFaceDataset(gallery_dataset, transform, individual_to_class)
    gallery_loader = DataLoader(gallery_torch, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                                num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, _ in gallery_loader:
            images = images.to(device, non_blocking=True)
            gallery_embeddings.append(model(images))
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).float()
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Compute fresh per-individual thresholds from this model's ArcFace centers
    class_to_name = {v: k for k, v in individual_to_class.items()}
    per_individual_thresholds, global_threshold = compute_arcface_center_thresholds(
        gallery_embeddings, gallery_labels, arcface_loss, class_to_name, device
    )

    print(f"\nFresh ArcFace center p2 thresholds (global mean={global_threshold:.4f}):")
    for name, t in sorted(per_individual_thresholds.items()):
        print(f"    {name}: {t:.4f}")

    # Query embeddings (known individuals from test split)
    query_torch = ArcFaceDataset(query_dataset, transform, individual_to_class)
    query_loader = DataLoader(query_torch, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    query_embeddings = []
    query_labels = []
    query_quality = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, quality in query_loader:
            images = images.to(device, non_blocking=True)
            query_embeddings.append(model(images))
            query_labels.extend(labels.tolist())
            query_quality.extend(quality.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0).float()
    query_labels = torch.tensor(query_labels).to(device)
    query_quality = np.array(query_quality)
    query_labels_np = query_labels.cpu().numpy()

    # Compute cosine similarity: query vs gallery
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    # Shared config for result JSON
    result_config = {
        'seed': seed,
        'task': task_name,
        'criterion': criterion,
        'alpha': best_alpha,
        'alpha_filter': alpha_filter,
        'query_quality_threshold': query_quality_threshold,
        'best_epoch': best_epoch,
        'hygiene_score': best_config['score'],
        'learning_rate': best_lr,
        'image_size': best_size,
        'embedding_dim': best_embedding_dim,
        'loss': 'WeightedArcFace',
        'arcface_margin': ARCFACE_MARGIN,
        'arcface_scale': ARCFACE_SCALE,
        'backbone': model_config['backbone_label'],
        'gallery_mode': 'all_data',
        'cosine_threshold': global_threshold,
        'per_individual_thresholds': per_individual_thresholds,
    }

    dataset_info = {
        'individuals': feasible_individuals,
        'gallery_info': gallery_info,
        'query_info': query_info,
        'total_gallery': len(all_gallery_indices),
        'total_query_known': len(all_query_indices),
        'query_source': 'hf_test_split',
    }

    if criterion == 'recall':
        # ---- CLOSED-SET: R@1 only ----
        q_mask = query_quality >= query_quality_threshold
        mask_indices = np.where(q_mask)[0]
        n_query = int(q_mask.sum())

        pred_indices = scores_known[mask_indices].argmax(dim=1)
        pred_labels = gallery_labels[pred_indices].cpu().numpy()
        true_labels = query_labels_np[mask_indices]
        correct = (pred_labels == true_labels)

        individual_correct = {}
        individual_total = {}
        for label in np.unique(true_labels):
            label_mask = (true_labels == label)
            individual_total[label] = int(label_mask.sum())
            individual_correct[label] = int(correct[label_mask].sum())

        per_ind_recall = [
            individual_correct.get(label, 0) / individual_total[label]
            for label in individual_total
        ]
        recall_at_1 = float(np.mean(per_ind_recall)) if per_ind_recall else 0.0

        print(f"\nTest Results (seed={seed}, task={task_name}):")
        print(f"  R@1 = {recall_at_1:.4f} (n={n_query}, q>={query_quality_threshold})")

        result = {
            'config': result_config,
            'dataset': dataset_info,
            'results': {
                'recall_at_1': recall_at_1,
                'query_quality_threshold': query_quality_threshold,
                'n_query': n_query,
                'n_individuals': len(individual_total),
            },
            'epoch_history': epoch_history,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'job_idx': args.idx,
            }
        }

    else:
        # ---- OPEN-SET: BA with fresh per-individual thresholds ----

        # Rare/Unknown from TEST split only
        rare_indices, rare_quality_arr, rare_labels_str = get_rare_individual_indices(
            test_metadata_cache, feasible_individuals, quality_threshold=0.0,
            promoted_individuals=promoted_individuals,
            excluded_individuals=excluded_individuals
        )

        if rare_indices:
            rare_emb, rare_quality_vals, rare_labels_arr = compute_rare_embeddings(
                model, test_dataset, rare_indices, rare_quality_arr, rare_labels_str,
                transform, device, embedding_dim=emb_dim
            )
            rare_emb = rare_emb.to(device)
        else:
            rare_emb = torch.empty(0, emb_dim).to(device)
            rare_quality_vals = np.array([])
            rare_labels_arr = np.array([])

        if len(rare_emb) > 0:
            scores_unknown = compute_cosine_similarity(rare_emb, gallery_embeddings)
        else:
            scores_unknown = torch.empty(0, len(gallery_embeddings)).to(device)

        # BA using fresh per-individual thresholds
        ba_metrics = compute_open_set_metrics_per_individual_threshold(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_vals,
            unknown_scores=scores_unknown,
            per_individual_thresholds=per_individual_thresholds,
            quality_thresholds=[query_quality_threshold],
            gallery_labels=gallery_labels,
            class_to_name=class_to_name
        )

        q_key = f"q>={query_quality_threshold}"
        ba_data = ba_metrics.get(q_key, {})
        ba = ba_data.get('balanced_accuracy', 0.0)
        kar = ba_data.get('known_accept_rate', 0.0)
        urr = ba_data.get('unknown_reject_rate', 0.0)
        n_k = ba_data.get('n_known_individuals', 0)
        n_u = ba_data.get('n_unknown_individuals', 0)

        print(f"\nTest Results (seed={seed}, task={task_name}):")
        print(f"  Fresh threshold (global mean): {global_threshold:.4f}")
        print(f"  {q_key}: BA={ba:.4f} (K={kar:.2f}[{n_k}], U={urr:.2f}[{n_u}])")

        dataset_info.update({
            'total_query_unknown': len(rare_indices) if rare_indices else 0,
            'n_unknown_individuals': len(set(rare_labels_str)) if rare_labels_str else 0,
            'unknown_source': 'hf_test_split',
        })

        result = {
            'config': result_config,
            'dataset': dataset_info,
            'results': {
                'balanced_accuracy': ba,
                'known_accept_rate': kar,
                'unknown_reject_rate': urr,
                'query_quality_threshold': query_quality_threshold,
                'n_known_individuals': n_k,
                'n_unknown_individuals': n_u,
            },
            'epoch_history': epoch_history,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'job_idx': args.idx,
            }
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)

    print(f"\nSaved results to: {output_path}")
    return filename


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/data/hf_cache"

    parser = argparse.ArgumentParser(description="Weighted ArcFace open-set experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments for all subcommands
    def add_common_args(sub):
        sub.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
        sub.add_argument("--idx", type=int, required=True, help="SLURM array task ID")
        sub.add_argument("--output_dir", type=str, required=True)
        sub.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
        sub.add_argument("--overwrite", action="store_true")

    # hygiene subcommand
    sub_hygiene = subparsers.add_parser("hygiene", help="Run hygiene sweep")
    add_common_args(sub_hygiene)

    # opt_lr subcommand
    sub_lr = subparsers.add_parser("opt_lr", help="Learning rate sweep")
    add_common_args(sub_lr)
    sub_lr.add_argument("--values", type=float, nargs="+", required=True, help="Learning rates to sweep")
    sub_lr.add_argument("--image-size", type=int, required=True, help="Fixed image size")
    sub_lr.add_argument("--embedding-dim", type=int, default=128, help="Fixed embedding dimension")
    sub_lr.add_argument("--epochs", type=int, default=20)
    sub_lr.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # opt_embedding_dim subcommand
    sub_emb = subparsers.add_parser("opt_embedding_dim", help="Embedding dimension sweep")
    add_common_args(sub_emb)
    sub_emb.add_argument("--values", type=int, nargs="+", required=True, help="Embedding dims to sweep")
    sub_emb.add_argument("--lr", type=float, required=True, help="Fixed learning rate")
    sub_emb.add_argument("--image-size", type=int, required=True, help="Fixed image size")
    sub_emb.add_argument("--epochs", type=int, default=20)
    sub_emb.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # test_eval subcommand
    sub_test = subparsers.add_parser("test_eval", help="Final test evaluation")
    add_common_args(sub_test)

    args = parser.parse_args()
    model_name = args.model
    config = MODEL_CONFIGS[model_name]

    if args.command == "hygiene":
        run_hygiene_sweep(model_name, args)

    elif args.command == "opt_lr":
        lrs = args.values
        seeds = args.seeds
        total_configs = len(lrs) * len(seeds)

        print("=" * 80)
        print(f"Weighted ArcFace LR Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
        print(f"  {len(lrs)} LRs x {len(seeds)} seeds = {total_configs} configs")
        print("=" * 80)

        if args.idx >= total_configs:
            print(f"Job {args.idx} has no work (only {total_configs} configs)")
            sys.exit(0)

        dataset = load_reidentification_dataset()
        metadata_cache = build_metadata_cache(dataset)
        feasibility_config = load_feasibility_config()
        if not feasibility_config:
            sys.exit(1)

        value_idx, seed_idx = divmod(args.idx, len(seeds))
        lr = lrs[value_idx]
        seed = seeds[seed_idx]
        print(f"\nLR: {lr}, Seed: {seed}, Image size: {args.image_size}, Epochs: {args.epochs}")

        try:
            run_opt_training(
                model_name, "lr", lr, args, dataset, feasibility_config, metadata_cache,
                learning_rate=lr, image_size=args.image_size,
                embedding_dim=args.embedding_dim,
                epochs=args.epochs, seed=seed,
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} (opt_lr) completed!")

    elif args.command == "opt_embedding_dim":
        dims = args.values
        seeds = args.seeds
        total_configs = len(dims) * len(seeds)

        print("=" * 80)
        print(f"Weighted ArcFace Embedding Dim Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
        print(f"  {len(dims)} dims x {len(seeds)} seeds = {total_configs} configs")
        print("=" * 80)

        if args.idx >= total_configs:
            print(f"Job {args.idx} has no work (only {total_configs} configs)")
            sys.exit(0)

        dataset = load_reidentification_dataset()
        metadata_cache = build_metadata_cache(dataset)
        feasibility_config = load_feasibility_config()
        if not feasibility_config:
            sys.exit(1)

        value_idx, seed_idx = divmod(args.idx, len(seeds))
        emb_dim = dims[value_idx]
        seed = seeds[seed_idx]
        print(f"\nEmbedding dim: {emb_dim}, Seed: {seed}, LR: {args.lr}, Size: {args.image_size}, Epochs: {args.epochs}")

        try:
            run_opt_training(
                model_name, "embedding_dim", emb_dim, args, dataset, feasibility_config, metadata_cache,
                learning_rate=args.lr, image_size=args.image_size, embedding_dim=emb_dim,
                epochs=args.epochs, seed=seed,
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} (opt_embedding_dim) completed!")

    elif args.command == "test_eval":
        task_names = list(TEST_TASKS.keys())
        seed = 0  # single seed

        print("=" * 80)
        print(f"Final Test Evaluation - {config['backbone_label']} Weighted ArcFace Re-ID - Job {args.idx}")
        print(f"  {len(task_names)} tasks (one per SLURM node), seed={seed}")
        print(f"  Tasks: {', '.join(f'{i}={name}' for i, name in enumerate(task_names))}")
        print("=" * 80)

        if args.idx >= len(task_names):
            print(f"Job {args.idx} has no work (only {len(task_names)} tasks)")
            sys.exit(0)

        task_name = task_names[args.idx]
        print(f"\nTask: {task_name} (seed={seed})")

        try:
            run_final_test(model_name, args, seed, task_name=task_name)
        except Exception as e:
            print(f"Error ({task_name}): {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} (test_eval) completed!")

"""
Shared utilities for reid open-set experiments across all backbones.

Consolidates duplicated code from reid_openset/{dinov3,megadescriptor,bioclip2}
into a single module with a registry-based architecture.
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
    create_train_transform_for_model,
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
# Training utilities
# ============================================================================

def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ============================================================================
# Hyperparameter loading
# ============================================================================

def _select_best_param(results_dir, glob_pattern, param_key):
    """Select best param value using cross-seed best-step mean R@1.

    Groups results by param value, for each group averages test_recall_at_1
    across seeds at each step, picks the step with highest mean, and
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
        histories = [r['results']['step_history'] for r in results]
        n_steps = len(histories[0])
        best_mean = 0.0
        for i in range(n_steps):
            mean_r1 = sum(h[i]['test_recall_at_1'] for h in histories) / len(histories)
            if mean_r1 > best_mean:
                best_mean = mean_r1
        if best_mean > best_score:
            best_score = best_mean
            best_value = value

    return best_value, best_score


def load_best_hyperparams(model_name):
    """
    Load best LR, image size, and embedding dim from optimization sweep results.
    Falls back to defaults if results not found.

    Returns: (best_lr, best_size, best_embedding_dim)
    """
    config = MODEL_CONFIGS[model_name]
    experiment_dir = config["experiment_dir"]

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

THRESHOLDS = [0.0, 0.25, 0.5]
GALLERY_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def get_job_combinations(job_idx, max_jobs=24):
    """Map job index to list of (threshold, gallery_size, seed) tuples."""
    all_combinations = []
    for threshold in THRESHOLDS:
        for gallery_size in GALLERY_SIZES:
            for seed in SEEDS:
                all_combinations.append((threshold, gallery_size, seed))

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


def train_single_config(model_name, threshold, gallery_size, seed, args,
                         dataset, config, metadata_cache,
                         learning_rate, image_size, embedding_dim,
                         total_steps=300, eval_every=10,
                         aug_flags=None,
                         test_dataset=None, test_metadata_cache=None,
                         combined_rare_dataset=None, combined_rare_metadata=None,
                         rare_excluded=None):
    """Train one hygiene sweep configuration and return results."""
    from utils.arcface import ArcFaceLoss, CoverageSampler
    from utils.training import check_result_exists

    model_config = MODEL_CONFIGS[model_name]

    set_all_seeds(seed)

    filename = f"threshold={threshold:.2f}_gallery={gallery_size}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)

    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: threshold={threshold}, gallery_size={gallery_size}, seed={seed}")
    if aug_flags and any(aug_flags.values()):
        active = [k for k, v in aug_flags.items() if v]
        print(f"  Augmentations: {', '.join(active)}")
    else:
        print(f"  Augmentations: none")
    print(f"{'='*60}")

    # Get qualified individuals from config (preprocessing already enforces criteria)
    feasible_individuals = config.get('qualified_individuals', [])
    promoted_individuals = config.get('promoted_to_rare', [])
    excluded_individuals = config.get('excluded_entirely', [])

    if len(feasible_individuals) < 2:
        print(f"Not enough individuals ({len(feasible_individuals)}) for training (need >= 2)")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create filtered gallery dataset (gallery from train minus validation, query from validation)
    train_dataset, val_dataset, individual_to_class, dataset_info = create_filtered_gallery_dataset(
        dataset, feasible_individuals, gallery_size, threshold, seed,
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
    eval_transform = create_transform_for_model(model_name, size=image_size)
    if aug_flags and any(aug_flags.values()):
        train_transform = create_train_transform_for_model(model_name, size=image_size, **aug_flags)
    else:
        train_transform = eval_transform
    train_torch_dataset = ArcFaceDataset(train_dataset, train_transform, individual_to_class)

    # Create coverage sampler (infinite iterator for step-based training)
    sampler = CoverageSampler(
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

    # Create ArcFace loss
    criterion = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
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

    # Build test query dataset (known individuals only) if test data provided
    test_query_dataset = None
    test_query_info = {}
    if test_dataset is not None and test_metadata_cache is not None:
        test_id_to_indices = test_metadata_cache['id_to_indices']
        all_test_query_indices = []
        for ind_id in feasible_individuals:
            t_indices = test_id_to_indices.get(ind_id, [])
            all_test_query_indices.extend(t_indices)
            test_query_info[ind_id] = len(t_indices)
        if all_test_query_indices:
            test_query_dataset = test_dataset.select(all_test_query_indices)
            print(f"  Test query: {len(all_test_query_indices)} images for {len(feasible_individuals)} individuals")

    # Determine rare source for test-set BA evaluation
    test_rare_dataset = combined_rare_dataset if combined_rare_dataset is not None else test_dataset
    test_rare_metadata = combined_rare_metadata if combined_rare_metadata is not None else test_metadata_cache
    test_rare_excluded = rare_excluded if rare_excluded is not None else excluded_individuals

    # Step-based training loop
    start_time = time.time()
    step_history = []
    best_recall = 0.0
    best_recall_step = 0
    n_evals = total_steps // eval_every

    model.train()
    criterion.train()
    train_iter = iter(train_loader)
    running_loss = 0.0

    for step in range(1, total_steps + 1):
        batch = next(train_iter)
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)
        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % eval_every != 0:
            continue

        # Eval checkpoint
        avg_loss = running_loss / eval_every
        running_loss = 0.0
        eval_num = step // eval_every

        # Val: R@1 only (no open-set BA)
        query_quality_metrics, _, val_loss = evaluate_recall_with_openset(
            model, train_dataset, val_dataset, individual_to_class, eval_transform, device,
            dataset, feasible_individuals, metadata_cache,
            criterion=criterion, embedding_dim=emb_dim,
            skip_open_set=True
        )

        # Test-set evaluation with combined rare pool for BA
        test_query_quality_metrics = None
        test_open_set_metrics = None
        if test_query_dataset is not None:
            test_query_quality_metrics, test_open_set_metrics, _ = evaluate_recall_with_openset(
                model, train_dataset, test_query_dataset, individual_to_class, eval_transform, device,
                test_rare_dataset, feasible_individuals, test_rare_metadata,
                criterion=criterion, embedding_dim=emb_dim,
                promoted_individuals=promoted_individuals,
                excluded_individuals=test_rare_excluded
            )

        recall_1 = query_quality_metrics["q>=0.0"]["recall_at_1"]

        if recall_1 > best_recall:
            best_recall = recall_1
            best_recall_step = step

        print(f"Step {step:3d}/{total_steps} [{eval_num}/{n_evals}]: Loss={avg_loss:.4f}, ValLoss={val_loss:.4f}")

        for q_thresh in QUERY_QUALITY_THRESHOLDS:
            q_key = f"q>={q_thresh}"
            r1 = query_quality_metrics[q_key]['recall_at_1']
            count = query_quality_metrics[q_key]['count']
            print(f"  VAL  {q_key}: R@1={r1:.4f} (n={count:3d})")

        if test_query_quality_metrics is not None:
            test_by_q = test_open_set_metrics.get('by_quality', {})
            for q_thresh in QUERY_QUALITY_THRESHOLDS:
                q_key = f"q>={q_thresh}"
                t_r1 = test_query_quality_metrics[q_key]['recall_at_1']
                t_ba = test_by_q.get(q_key, {}).get('balanced_accuracy', 0.0)
                t_kar = test_by_q.get(q_key, {}).get('known_accept_rate', 0.0)
                t_urr = test_by_q.get(q_key, {}).get('unknown_reject_rate', 0.0)
                t_n_k = test_by_q.get(q_key, {}).get('n_known_individuals', 0)
                t_n_u = test_by_q.get(q_key, {}).get('n_unknown_individuals', 0)
                print(f"  TEST {q_key}: R@1={t_r1:.4f}, BA={t_ba:.4f} (K={t_kar:.2f}[{t_n_k}], U={t_urr:.2f}[{t_n_u}])")

        model.train()
        criterion.train()

        entry = {
            'step': step,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'query_quality_metrics': query_quality_metrics,
        }
        if test_query_quality_metrics is not None:
            entry['test_query_quality_metrics'] = test_query_quality_metrics
            entry['test_open_set'] = test_open_set_metrics
        step_history.append(entry)

    training_time = time.time() - start_time

    result = {
        'config': {
            'threshold': threshold,
            'gallery_size': gallery_size,
            'seed': seed,
            'loss': 'ArcFace',
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'embedding_dim': emb_dim,
            'optimizer': 'AdamW',
            'scheduler': 'None (fixed LR)',
            'learning_rate': learning_rate,
            'total_steps': total_steps,
            'eval_every': eval_every,
            'batch_size': TRAIN_BATCH_SIZE,
            'backbone': model_config['backbone_label'],
            'score_normalization': 'Raw Cosine',
            'augmentations': aug_flags or {"blur": False, "jitter": False, "ir_sim": False}
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
        'step_history': step_history,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx,
            'transform': f'resize_{image_size}_{model_name}_norm',
            'best_recall_step': best_recall_step,
            'best_recall_at_1': best_recall,
        }
    }

    if test_query_dataset is not None:
        result['dataset']['test_query_samples_per_individual'] = test_query_info
        result['dataset']['total_test_query_size'] = len(test_query_dataset)
        if best_recall_step > 0:
            # Find the step_history entry for the best step
            best_idx = best_recall_step // eval_every - 1
            hist_entry = step_history[best_idx]
            result['test_at_best_recall_step'] = {
                'step': best_recall_step,
                'test_query_quality_metrics': hist_entry.get('test_query_quality_metrics'),
                'test_open_set': hist_entry.get('test_open_set'),
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best val R@1 step: {best_recall_step}, R@1={best_recall:.4f}")
    if test_query_dataset is not None and best_recall_step > 0:
        best_idx = best_recall_step // eval_every - 1
        hist_entry = step_history[best_idx]
        test_qm = hist_entry.get('test_query_quality_metrics', {})
        test_os = hist_entry.get('test_open_set', {})
        t_r1 = test_qm.get('q>=0.0', {}).get('recall_at_1', 0.0)
        t_ba = test_os.get('by_quality', {}).get('q>=0.0', {}).get('balanced_accuracy', 0.0)
        print(f"Test @ step {best_recall_step}: R@1={t_r1:.4f}, BA={t_ba:.4f}")

    return filename


def run_hygiene_sweep(model_name, args):
    """
    Main entry point for running a hygiene sweep experiment.

    Loads dataset, config, best hyperparams, distributes work across SLURM jobs,
    and trains each configuration.
    """
    config = MODEL_CONFIGS[model_name]

    print("=" * 80)
    print(f"Open-Set {config['backbone_label']} + ArcFace + Raw Cosine Gallery Hygiene Sweep - Job {args.idx}")
    print("=" * 80)

    # Load best hyperparameters; augmentations are always on
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)
    aug_flags = {"blur": True, "jitter": True, "ir_sim": True, "hflip": True, "rotation": True}

    # Load datasets and config
    print("\nLoading datasets...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    test_dataset_full = load_reidentification_test_dataset()
    test_metadata_cache = build_metadata_cache(test_dataset_full)
    feasibility_config = load_feasibility_config()

    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    # Load unknown assignment: combined train+test rare source for test-time BA
    unknown_config_path = 'preprocessing/results/unknown_assignment.json'
    try:
        with open(unknown_config_path, 'r') as f:
            unknown_config = json.load(f)
        from datasets import concatenate_datasets
        combined_rare_dataset = concatenate_datasets([dataset, test_dataset_full])
        combined_rare_metadata = build_metadata_cache(combined_rare_dataset)
        rare_excluded = ['HLC21-H1']  # only exclude individuals too sparse for unknown eval
        print(f"  Combined rare pool: {unknown_config['summary']['combined_unknown_images']} images "
              f"from {unknown_config['summary']['n_unknown_individuals']} unknown individuals")
    except FileNotFoundError:
        print(f"Warning: {unknown_config_path} not found, test BA will use test-split unknowns only")
        combined_rare_dataset = test_dataset_full
        combined_rare_metadata = test_metadata_cache
        rare_excluded = feasibility_config.get('excluded_entirely', [])

    total_combinations = len(THRESHOLDS) * len(GALLERY_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Model: {config['backbone_label']} + Projection Head ({best_embedding_dim}-d) + ArcFace")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    print(f"  Score Normalization: Raw Cosine Similarity (L2-normalized)")
    print(f"  Optimizer: AdamW (lr={best_lr}, default settings)")
    print(f"  Embedding: {best_embedding_dim}-d (trainable projection)")
    print(f"  Image size: {best_size}")
    print(f"  Augmentations: always on (hflip, rotation, blur, jitter, ir_sim)")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")
    print(f"  Val: R@1 only (no open-set BA)")
    print(f"  Test: R@1 + BA (combined train+test unknown pool)")

    combinations = get_job_combinations(args.idx)

    if not combinations:
        print(f"\nNo combinations assigned to job {args.idx}")
        return

    print(f"\nJob {args.idx} processing {len(combinations)} configurations:")
    for threshold, gallery_size, seed in combinations[:5]:
        print(f"  threshold={threshold}, gallery_size={gallery_size}, seed={seed}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more")

    results_summary = []
    for i, (threshold, gallery_size, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(
                model_name, threshold, gallery_size, seed, args,
                dataset, feasibility_config, metadata_cache,
                learning_rate=best_lr, image_size=best_size,
                embedding_dim=best_embedding_dim, aug_flags=aug_flags,
                test_dataset=test_dataset_full,
                test_metadata_cache=test_metadata_cache,
                combined_rare_dataset=combined_rare_dataset,
                combined_rare_metadata=combined_rare_metadata,
                rare_excluded=rare_excluded,
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
                     total_steps=100, eval_every=10, seed=0, train_transform=None):
    """
    Shared training loop for all opt sweep scripts (LR, resize, embedding_dim, augmentation).

    Args:
        model_name: Key in MODEL_CONFIGS
        sweep_param_name: Name of swept param ('lr', 'resize', 'embedding_dim', 'augmentation')
        sweep_param_value: Value of swept param
        args: Argparse namespace (needs .output_dir, .overwrite, .device, .idx)
        dataset: HuggingFace dataset
        config: Feasibility config
        metadata_cache: Metadata cache dict
        learning_rate: LR to use (overrides default)
        image_size: Image size to use (overrides default)
        embedding_dim: Embedding dimension
        total_steps: Total number of training steps
        eval_every: Evaluate every N steps
        seed: Random seed
        train_transform: Optional augmented transform for training data.
            If None, uses the same (clean) eval transform for both.

    Returns:
        filename if saved, None if skipped
    """
    from utils.arcface import ArcFaceLoss, CoverageSampler

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

    # Create transforms: eval is always clean, train may have augmentations
    eval_transform = create_transform_for_model(model_name, size=image_size)
    t_transform = train_transform if train_transform is not None else eval_transform
    train_torch_dataset = ArcFaceDataset(train_dataset, t_transform, individual_to_class)

    # Create coverage sampler (infinite iterator for step-based training)
    sampler = CoverageSampler(
        labels=train_torch_dataset.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create ArcFace loss
    criterion = ArcFaceLoss(
        num_classes=len(qualified_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(criterion)
    print(f"  Trainable params: head={head_params:,}, arcface={arcface_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=learning_rate
    )

    # Step-based training loop
    start_time = time.time()
    step_history = []
    best_recall = 0.0
    best_step = 0
    n_evals = total_steps // eval_every

    model.train()
    criterion.train()
    train_iter = iter(train_loader)
    running_loss = 0.0

    for step in range(1, total_steps + 1):
        batch = next(train_iter)
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)
        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % eval_every != 0:
            continue

        avg_loss = running_loss / eval_every
        running_loss = 0.0
        eval_num = step // eval_every

        recall_at_1 = evaluate_recall_simple(
            model, train_dataset, test_dataset, individual_to_class, eval_transform, device
        )

        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            best_step = step

        print(f"Step {step:3d}/{total_steps} [{eval_num}/{n_evals}]: loss={avg_loss:.4f}, test_R@1={recall_at_1:.4f}")

        step_history.append({
            'step': step,
            'train_loss': avg_loss,
            'test_recall_at_1': recall_at_1
        })

        model.train()
        criterion.train()

    training_time = time.time() - start_time

    # Prepare result
    result = {
        'config': {
            'learning_rate': learning_rate,
            'total_steps': total_steps,
            'eval_every': eval_every,
            'seed': seed,
            'backbone': model_config['backbone_label'],
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
            'best_step': best_step,
            'best_test_recall_at_1': best_recall,
            'step_history': step_history
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
    print(f"Best R@1: {best_recall:.4f} at step {best_step}")

    return filename



# ============================================================================
# Final test evaluation
# ============================================================================

def load_best_hygiene_config(model_name, criterion='harmonic_mean', threshold_filter=None):
    """
    Read all hygiene sweep results for a backbone and select the best operating point.

    Groups by (threshold, gallery_size), extracts best step per seed by criterion,
    averages across seeds, returns the (threshold, gallery_size, step) with highest mean.

    Args:
        model_name: Key in MODEL_CONFIGS
        criterion: 'recall', 'balanced_accuracy', or 'harmonic_mean'
        threshold_filter: If set, only consider groups with this threshold value

    Returns:
        dict with {threshold, gallery_size, best_step, score}
    """
    config = MODEL_CONFIGS[model_name]
    results_dir = os.path.join(config['experiment_dir'], 'results', 'hygiene')

    files = glob.glob(os.path.join(results_dir, 'threshold=*_gallery=*_seed=*.json'))
    if not files:
        raise FileNotFoundError(f"No hygiene results found in {results_dir}")

    # Load all results
    results = []
    for f in files:
        with open(f, 'r') as fp:
            results.append(json.load(fp))

    # Group by (threshold, gallery_size)
    groups = defaultdict(list)
    for r in results:
        key = (r['config']['threshold'], r['config']['gallery_size'])
        groups[key].append(r)

    # Optionally filter to a specific threshold
    if threshold_filter is not None:
        groups = {k: v for k, v in groups.items() if k[0] == threshold_filter}
        if not groups:
            raise ValueError(f"No hygiene results with threshold={threshold_filter} in {results_dir}")

    best_overall_score = -1.0
    best_config = None

    # For unfiltered tasks (threshold_filter=0.0), only evaluate at q>=0.0.
    # For filtered tasks, search across all query quality thresholds to find
    # the best (threshold, gallery_size, query_q_threshold, epoch) combo.
    if threshold_filter is not None and threshold_filter == 0.0:
        q_thresholds_to_search = [0.0]
    else:
        q_thresholds_to_search = QUERY_QUALITY_THRESHOLDS

    for (threshold, gallery_size), group_results in groups.items():
        for q_thresh in q_thresholds_to_search:
            q_key = f"q>={q_thresh}"

            # Build per-seed lookup: step_num -> score
            # Only include steps that have evaluation data.
            all_histories = [r['step_history'] for r in group_results]

            eval_steps = []
            for entry in all_histories[0]:
                if 'query_quality_metrics' in entry:
                    eval_steps.append(entry['step'])

            # For each eval step, average the criterion across seeds
            best_mean = -1.0
            best_step_num = eval_steps[0] if eval_steps else 1

            for step_num in eval_steps:
                scores = []
                for history in all_histories:
                    for entry in history:
                        if entry['step'] == step_num:
                            # Always select by val recall (the only val metric;
                            # val skips open-set BA).  Test metrics are reported
                            # but never used for model/step selection.
                            s = entry.get('query_quality_metrics', {}).get(q_key, {}).get('recall_at_1')
                            if s is not None:
                                scores.append(s)
                            break

                if scores:
                    step_mean = np.mean(scores)
                    if step_mean > best_mean:
                        best_mean = step_mean
                        best_step_num = step_num

            # Extract per-individual thresholds at best step (for open-set tasks)
            all_per_individual_thresholds = defaultdict(list)
            cosine_thresholds = []
            for history in all_histories:
                for entry in history:
                    if entry['step'] == best_step_num:
                        open_set = entry.get('test_open_set', {})
                        thresh_cal = open_set['threshold_calibration']

                        for name, thresh in thresh_cal['per_individual'].items():
                            all_per_individual_thresholds[name].append(thresh)
                        cosine_thresholds.append(thresh_cal['global_threshold'])
                        break

            # Average each individual's threshold across seeds
            mean_per_individual = {name: float(np.mean(vals))
                                   for name, vals in all_per_individual_thresholds.items()}
            mean_cosine_thresh = float(np.mean(cosine_thresholds)) if cosine_thresholds else None

            if best_mean > best_overall_score:
                best_overall_score = best_mean
                best_config = {
                    'threshold': threshold,
                    'gallery_size': gallery_size,
                    'query_quality_threshold': q_thresh,
                    'best_step': best_step_num,
                    'score': float(best_mean),
                    'criterion': criterion,
                    'n_seeds': len(group_results),
                    'cosine_threshold': mean_cosine_thresh,
                    'per_individual_thresholds': mean_per_individual,
                }

    print(f"Best hygiene config ({criterion}): threshold={best_config['threshold']}, "
          f"gallery_size={best_config['gallery_size']}, "
          f"query_q>={best_config['query_quality_threshold']}, "
          f"step={best_config['best_step']}, "
          f"score={best_config['score']:.4f} (n={best_config['n_seeds']} seeds)")

    return best_config


# Each task selects its own optimal (threshold, gallery_size, epoch) from the
# hygiene sweep using the criterion and threshold constraint that match its goal.
#
# Filtered tasks use ALL images above the quality threshold (no gallery_size cap).
# Matched tasks use event-matched subsampling: for each individual's YMDH event,
# count how many images pass the quality filter, then randomly sample that count
# from the full event (no filter). This isolates the effect of quality filtering.
TEST_TASKS = {
    'closed_filtered': {'criterion': 'recall',            'threshold_filter': None},   # idx 0
    'closed_matched':  {'criterion': 'recall',            'threshold_filter': 0.0},    # idx 1
    'open_filtered':   {'criterion': 'balanced_accuracy', 'threshold_filter': None},   # idx 2
    'open_matched':    {'criterion': 'balanced_accuracy', 'threshold_filter': 0.0},    # idx 3
}


def run_final_test(model_name, args, seed, task_name):
    """
    Final test evaluation using the best operating point from hygiene sweep.

    Gallery construction:
      - Filtered tasks: ALL images above quality threshold (no gallery_size cap)
      - Matched tasks: event-matched subsampling from the full pool to match
        the filtered gallery's per-event image counts

    Thresholds are computed fresh from the trained model's own ArcFace centers
    (p5 percentile), not imported from hygiene averages.

    Args:
        model_name: Key in MODEL_CONFIGS
        args: Argparse namespace
        seed: Random seed for gallery sampling and training
        task_name: Key in TEST_TASKS
    """
    from utils.arcface import ArcFaceLoss, CoverageSampler
    from utils.training import check_result_exists

    task_cfg = TEST_TASKS[task_name]
    criterion = task_cfg['criterion']
    threshold_filter = task_cfg['threshold_filter']
    is_matched = threshold_filter is not None  # matched tasks have threshold_filter=0.0

    model_config = MODEL_CONFIGS[model_name]
    experiment_dir = model_config['experiment_dir']

    # For matched tasks, we need TWO hygiene configs:
    # 1. The filtered config (same criterion, threshold_filter=None) to determine
    #    what "filtered" means (quality threshold) and build the filtered gallery
    # 2. The unfiltered config (same criterion, threshold_filter=0.0) for
    #    best_step and query_quality_threshold
    if is_matched:
        filtered_config = load_best_hygiene_config(model_name, criterion=criterion,
                                                    threshold_filter=None)
        best_config = load_best_hygiene_config(model_name, criterion=criterion,
                                               threshold_filter=threshold_filter)
        quality_threshold = filtered_config['threshold']  # defines "filtered"
        best_step = best_config['best_step']
        query_quality_threshold = best_config['query_quality_threshold']
    else:
        best_config = load_best_hygiene_config(model_name, criterion=criterion,
                                               threshold_filter=threshold_filter)
        quality_threshold = best_config['threshold']
        best_step = best_config['best_step']
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
    print(f"  criterion={criterion}, quality_threshold={quality_threshold}, "
          f"query_q>={query_quality_threshold}, steps={best_step}")
    if is_matched:
        print(f"  gallery_mode=event_matched (matching filtered threshold={quality_threshold})")
    else:
        print(f"  gallery_mode=full_filtered (all images >= {quality_threshold})")
    print(f"{'='*60}")

    # Load best hyperparameters; augmentations are always on
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)
    best_aug = {"blur": True, "jitter": True, "ir_sim": True, "hflip": True, "rotation": True}

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

    # Create gallery from TRAIN split
    id_to_indices = train_metadata_cache['id_to_indices']
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}

    all_gallery_indices = []
    gallery_info = {}

    for ind_id in feasible_individuals:
        all_ind_indices = list(id_to_indices.get(ind_id, []))

        # Build filtered gallery: ALL images above quality threshold (no cap)
        filtered_pool = filter_training_pool_by_quality(
            train_metadata_cache, all_ind_indices, quality_threshold
        )

        if is_matched:
            # Event-matched subsampling: sample from full pool to match
            # the filtered gallery's per-event image counts
            sampled = subsample_to_match_filtered(
                train_metadata_cache, all_ind_indices, filtered_pool
            )
            gallery_info[ind_id] = {
                'sampled': len(sampled),
                'filtered_count': len(filtered_pool),
                'total': len(all_ind_indices),
                'mode': 'event_matched',
            }
            print(f"  {ind_id}: {len(sampled)} gallery (event-matched to {len(filtered_pool)} filtered)")
        else:
            # Filtered: use ALL eligible images
            sampled = filtered_pool
            gallery_info[ind_id] = {
                'sampled': len(sampled),
                'eligible': len(filtered_pool),
                'total': len(all_ind_indices),
                'mode': 'full_filtered',
            }
            if len(sampled) == 0:
                print(f"  WARNING: {ind_id} has NO samples above threshold {quality_threshold}")
            else:
                print(f"  {ind_id}: {len(sampled)}/{len(all_ind_indices)} gallery (filtered >= {quality_threshold})")

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
    eval_transform = create_transform_for_model(model_name, size=best_size)
    if any(best_aug.values()):
        train_transform = create_train_transform_for_model(model_name, size=best_size, **best_aug)
    else:
        train_transform = eval_transform
    train_torch_dataset = ArcFaceDataset(gallery_dataset, train_transform, individual_to_class)

    # Create coverage sampler (infinite iterator for step-based training)
    sampler = CoverageSampler(
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

    # Create ArcFace loss
    arcface_loss = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(arcface_loss.parameters()),
        lr=best_lr
    )

    # Step-based training loop
    print(f"\nTraining for {best_step} steps...")
    start_time = time.time()

    model.train()
    arcface_loss.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    for step in range(1, best_step + 1):
        batch = next(train_iter)
        images = batch[0].to(device)
        labels = batch[1].to(device)
        embeddings = model(images)
        optimizer.zero_grad()
        loss = arcface_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 10 == 0:
            print(f"Step {step:3d}/{best_step}: Loss={running_loss / 10:.4f}")
            running_loss = 0.0

    training_time = time.time() - start_time

    # Evaluate on FULL test split
    print("\nEvaluating on test split...")
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    # Gallery embeddings
    gallery_torch = ArcFaceDataset(gallery_dataset, eval_transform, individual_to_class)
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

    print(f"\nFresh ArcFace center p10 thresholds (global mean={global_threshold:.4f}):")
    for name, t in sorted(per_individual_thresholds.items()):
        print(f"    {name}: {t:.4f}")

    # Query embeddings (known individuals from test split)
    query_torch = ArcFaceDataset(query_dataset, eval_transform, individual_to_class)
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
        'threshold_filter': threshold_filter,
        'quality_threshold': quality_threshold,
        'query_quality_threshold': query_quality_threshold,
        'best_step': best_step,
        'hygiene_score': best_config['score'],
        'learning_rate': best_lr,
        'image_size': best_size,
        'embedding_dim': best_embedding_dim,
        'loss': 'ArcFace',
        'arcface_margin': ARCFACE_MARGIN,
        'arcface_scale': ARCFACE_SCALE,
        'backbone': model_config['backbone_label'],
        'gallery_mode': 'event_matched' if is_matched else 'full_filtered',
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
        # Filter queries to those meeting the selected quality threshold
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
                eval_transform, device, embedding_dim=emb_dim
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

        # BA using fresh per-individual thresholds from this model's ArcFace centers
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
# Step-count sweep: fixed step budget training
# ============================================================================

STEP_COUNT_GALLERY_THRESHOLDS = [0.0, 0.25, 0.5]
STEP_COUNT_SEEDS = [0, 1, 2, 3, 4]
STEP_COUNT_BUDGET = 3000
STEP_COUNT_EVAL_EVERY = 100


def _build_step_count_job_configs():
    """Build idx -> (label, gallery_threshold, seed) mapping."""
    configs = {}
    idx = 0
    for gt in STEP_COUNT_GALLERY_THRESHOLDS:
        for seed in STEP_COUNT_SEEDS:
            configs[idx] = (f"g{gt:.2f}_seed{seed}", gt, seed)
            idx += 1
    return configs


def run_step_count_sweep(model_name, args):
    """
    Main entry point for step-budget training sweep.

    Fixed gradient step budget with periodic evaluation.  The CoverageSampler
    uses ALL images per individual, shuffled, and reshuffles per-individual
    once coverage is complete.
    """
    from utils.arcface import ArcFaceLoss, CoverageSampler
    from tqdm import tqdm

    model_config = MODEL_CONFIGS[model_name]
    experiment_dir = model_config["experiment_dir"]

    job_configs = _build_step_count_job_configs()
    total_configs = len(job_configs)

    step_budget = STEP_COUNT_BUDGET
    eval_every = STEP_COUNT_EVAL_EVERY

    if args.idx not in job_configs:
        print(f"idx={args.idx} has no work assigned ({total_configs} configs), exiting.")
        return

    mode_label, gallery_threshold, seed = job_configs[args.idx]

    output_path = os.path.join(
        args.output_dir,
        f"g{gallery_threshold:.2f}_seed={seed}.json",
    )

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — config only, no training")
        print("=" * 70)
        print(f"  idx:                {args.idx}")
        print(f"  mode:               {mode_label}")
        print(f"  gallery_threshold:  {gallery_threshold}")
        print(f"  step_budget:        {step_budget}")
        print(f"  eval_every:         {eval_every}")
        print(f"  seed:               {seed}")
        print(f"  output_path:        {output_path}")
        print(f"\nAll {total_configs} configs:")
        for i, (lbl, gt, s) in sorted(job_configs.items()):
            marker = " <-- this job" if i == args.idx else ""
            print(f"  idx={i}: {lbl}{marker}")
        return

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Result exists: {output_path}. Use --overwrite to replace.")
        return

    set_all_seeds(seed)

    print("=" * 70)
    print(f"Tune Step Count [{mode_label}]: {model_config['backbone_label']}, seed={seed}, idx={args.idx}")
    print(f"  step_budget={step_budget}, eval_every={eval_every}")
    print("=" * 70)

    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)
    aug_flags = {"blur": True, "jitter": True, "ir_sim": True, "hflip": True, "rotation": True}

    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)

    feasibility_config = load_feasibility_config()
    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    feasible_individuals = feasibility_config.get("qualified_individuals", [])
    promoted_individuals = feasibility_config.get("promoted_to_rare", [])
    excluded_individuals = feasibility_config.get("excluded_entirely", [])

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")
    print(f"\nMode: {mode_label} (threshold>={gallery_threshold})")

    train_ds, val_ds, individual_to_class, dataset_info = create_filtered_gallery_dataset(
        dataset, feasible_individuals, gallery_size=None, threshold=gallery_threshold,
        seed=seed, metadata_cache=metadata_cache, config=feasibility_config,
    )
    dataset_info["mode"] = mode_label
    dataset_info["gallery_threshold"] = gallery_threshold

    if train_ds is None or len(train_ds) == 0:
        print(f"No training data for {mode_label}, skipping.")
        return

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    eval_transform = create_transform_for_model(model_name, size=best_size)
    if any(aug_flags.values()):
        train_transform = create_train_transform_for_model(model_name, size=best_size, **aug_flags)
    else:
        train_transform = eval_transform
    train_torch = ArcFaceDataset(train_ds, train_transform, individual_to_class)

    sampler = CoverageSampler(
        labels=train_torch.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )
    class_sizes = {l: len(v) for l, v in sampler.label_to_all_indices.items()}
    print(f"  Sampler: CoverageSampler, batch_size={TRAIN_BATCH_SIZE}, "
          f"k={sampler.k}/class, pool min={min(class_sizes.values())}, "
          f"pool max={max(class_sizes.values())}")
    print(f"  Step budget: {step_budget}, eval every {eval_every} steps")

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(train_torch, batch_sampler=sampler,
                              num_workers=0, collate_fn=collate_fn)

    arcface_loss = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(arcface_loss)
    print(f"Trainable: head={head_params:,}, arcface={arcface_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(arcface_loss.parameters()),
        lr=best_lr,
    )

    # Warm up dataloader
    print("\nWarming up dataloader (first batch)...", flush=True)
    train_iter = iter(train_loader)
    _warmup_batch = next(train_iter)
    del _warmup_batch
    print("Dataloader ready.")

    print(f"\nTraining for {step_budget} steps (eval every {eval_every} steps)...")
    start_time = time.time()
    step_history = []

    q_keys = [f"q>={q}" for q in QUERY_QUALITY_THRESHOLDS]
    best_metrics = {}
    for q_key in q_keys:
        best_metrics[q_key] = {
            "best_recall": 0.0, "best_recall_step": 0,
        }

    global_step = 0
    running_loss = 0.0
    running_batches = 0

    model.train()
    arcface_loss.train()

    pbar = tqdm(total=step_budget, desc="Training", leave=True)
    for batch in train_iter:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)
        optimizer.zero_grad()
        loss = arcface_loss(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_batches += 1
        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{running_loss / running_batches:.4f}", step=global_step)

        is_eval_step = (global_step % eval_every == 0) or (global_step == step_budget)

        if is_eval_step:
            train_loss = running_loss / max(running_batches, 1)

            query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
                model, train_ds, val_ds, individual_to_class, eval_transform, device,
                dataset, feasible_individuals, metadata_cache,
                criterion=arcface_loss, embedding_dim=emb_dim,
                promoted_individuals=promoted_individuals,
                excluded_individuals=excluded_individuals,
                skip_open_set=True,
            )

            # Coverage stats summary
            cstats = sampler.coverage_stats
            min_resets = min(v["resets"] for v in cstats.values())
            max_resets = max(v["resets"] for v in cstats.values())
            print(f"\nStep {global_step:5d}/{step_budget}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
                  f"coverage_resets=[{min_resets},{max_resets}]")

            step_entry = {
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "query_quality_metrics": query_quality_metrics,
                "open_set_metrics": open_set_metrics,
            }

            for q_key in q_keys:
                r1 = query_quality_metrics.get(q_key, {}).get("recall_at_1", 0.0)

                bm = best_metrics[q_key]
                if r1 > bm["best_recall"]:
                    bm["best_recall"] = r1
                    bm["best_recall_step"] = global_step

                n_q = query_quality_metrics.get(q_key, {}).get("count", 0)
                print(f"  {q_key}: R@1={r1:.4f} (n={n_q})")

            step_history.append(step_entry)

            running_loss = 0.0
            running_batches = 0

            model.train()
            arcface_loss.train()

        if global_step >= step_budget:
            break

    pbar.close()
    training_time = time.time() - start_time

    print(f"\nBest metrics by query quality:")
    for q_key in q_keys:
        bm = best_metrics[q_key]
        print(f"  {q_key}: R@1={bm['best_recall']:.4f} (step {bm['best_recall_step']})")

    # Final coverage summary
    final_coverage = sampler.coverage_stats
    coverage_summary = {
        label: {"pool_size": v["pool_size"], "resets": v["resets"]}
        for label, v in final_coverage.items()
    }

    result = {
        "config": {
            "mode": mode_label,
            "gallery_threshold": gallery_threshold,
            "step_budget": step_budget,
            "eval_every": eval_every,
            "seed": seed,
            "learning_rate": best_lr,
            "image_size": best_size,
            "embedding_dim": best_embedding_dim,
            "augmentations": aug_flags,
            "loss": "ArcFace",
            "arcface_margin": ARCFACE_MARGIN,
            "arcface_scale": ARCFACE_SCALE,
            "backbone": model_config["backbone_label"],
            "sampler": "CoverageSampler",
            "coverage_summary": coverage_summary,
        },
        "dataset": dataset_info,
        "step_history": step_history,
        "best_metrics": best_metrics,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "training_time_seconds": training_time,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else o)

    print(f"\nSaved: {output_path}")
    print(f"Training time: {training_time:.1f}s")


# ============================================================================
# Test evaluation from step-count sweep
# ============================================================================

def load_eval_schedule(model_name, gallery_threshold):
    """
    For each query_q, find the best step by max-mean R@1
    across 5 seeds from the step-count sweep.

    Returns: list of (q_thresh, best_step, mean_r1)
    """
    config = MODEL_CONFIGS[model_name]
    results_dir = os.path.join(config["experiment_dir"], "results", "step_count")

    # Load all seed results for this gallery_threshold
    seed_results = []
    for seed in STEP_COUNT_SEEDS:
        path = os.path.join(results_dir, f"g{gallery_threshold:.2f}_seed={seed}.json")
        with open(path) as f:
            seed_results.append(json.load(f))

    schedule = []
    for q_thresh in QUERY_QUALITY_THRESHOLDS:
        q_key = f"q>={q_thresh}"

        # Build step -> [r1_seed0, r1_seed1, ...] mapping
        step_scores = defaultdict(list)
        for sr in seed_results:
            for entry in sr['step_history']:
                step = entry['step']
                score = entry['query_quality_metrics'].get(q_key, {}).get('recall_at_1', 0.0)
                step_scores[step].append(score)

        # Pick step with highest mean R@1 across seeds
        best_step = None
        best_mean = -1.0
        for step, scores in step_scores.items():
            mean_score = sum(scores) / len(scores)
            if mean_score > best_mean:
                best_mean = mean_score
                best_step = step

        if best_step is not None:
            schedule.append((q_thresh, best_step, best_mean))

    return schedule


def _evaluate_and_save(model, arcface_loss, gallery_dataset, query_dataset,
                       test_dataset, test_metadata_cache, eval_transform,
                       individual_to_class, feasible_individuals,
                       promoted_individuals, excluded_individuals,
                       device, emb_dim, schedule_entries, output_dir,
                       gallery_threshold, step_history,
                       training_time, best_lr, best_size, best_embedding_dim,
                       aug_flags, model_config, seed, gallery_info, query_info,
                       all_gallery_indices, all_query_indices, rare_indices_cache,
                       export_dir=None, anchor_dataset=None, export_meta=None):
    """
    Evaluate at a scheduled step and save one JSON per query_q combo.

    schedule_entries: list of (q_thresh, step, sweep_score) for this step.
    rare_indices_cache: dict with pre-fetched rare/unknown data.
    export_dir: if set, save gallery/query/rare/anchor embeddings for this
        step as one .npz. anchor_dataset holds the last training event per
        individual (embedded unfiltered, regardless of gallery_threshold);
        export_meta holds pre-sliced filename/id/ymdh/quality arrays aligned
        to each embedding block.
    """
    model.eval()
    arcface_loss.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    class_to_name = {v: k for k, v in individual_to_class.items()}

    # Gallery embeddings
    gallery_torch = ArcFaceDataset(gallery_dataset, eval_transform, individual_to_class)
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

    # Per-individual thresholds from ArcFace centers
    per_individual_thresholds, global_threshold = compute_arcface_center_thresholds(
        gallery_embeddings, gallery_labels, arcface_loss, class_to_name, device
    )

    print(f"\n  ArcFace center thresholds (global mean={global_threshold:.4f}):")
    for name, t in sorted(per_individual_thresholds.items()):
        print(f"      {name}: {t:.4f}")

    # Query embeddings (known individuals from test split)
    query_torch = ArcFaceDataset(query_dataset, eval_transform, individual_to_class)
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

    # Cosine similarity: query vs gallery
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    # Rare/Unknown from TEST split (use cached indices, recompute embeddings)
    rare_indices = rare_indices_cache['rare_indices']
    rare_labels_str = rare_indices_cache['rare_labels_str']

    # Compute rare embeddings with current model state
    if rare_indices:
        rare_quality_arr = rare_indices_cache['rare_quality_arr']
        rare_emb, rare_quality_vals, rare_labels_arr = compute_rare_embeddings(
            model, test_dataset, rare_indices, rare_quality_arr, rare_labels_str,
            eval_transform, device, embedding_dim=emb_dim
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

    # ---- Embedding export: one .npz per (gallery_threshold, step) ----
    if export_dir is not None:
        step = schedule_entries[0][1]

        anchor_torch = ArcFaceDataset(anchor_dataset, eval_transform, individual_to_class)
        anchor_loader = DataLoader(anchor_torch, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                                   num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)
        anchor_embeddings = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for images, _, _ in anchor_loader:
                images = images.to(device, non_blocking=True)
                anchor_embeddings.append(model(images))
        anchor_embeddings = torch.cat(anchor_embeddings, dim=0).float()

        os.makedirs(export_dir, exist_ok=True)
        npz_path = os.path.join(
            export_dir, f"emb_g{gallery_threshold:.2f}_step{step}_seed={seed}.npz")
        np.savez_compressed(
            npz_path,
            gallery_emb=gallery_embeddings.cpu().numpy().astype(np.float16),
            query_emb=query_embeddings.cpu().numpy().astype(np.float16),
            rare_emb=rare_emb.cpu().numpy().astype(np.float16),
            anchor_emb=anchor_embeddings.cpu().numpy().astype(np.float16),
            per_individual_thresholds=json.dumps(
                {k: float(v) for k, v in per_individual_thresholds.items()}),
            global_threshold=float(global_threshold),
            step=step,
            gallery_threshold=gallery_threshold,
            backbone=model_config["backbone_label"],
            **export_meta,
        )
        print(f"    Exported embeddings: {npz_path} "
              f"(gallery={len(gallery_embeddings)}, query={len(query_embeddings)}, "
              f"rare={len(rare_emb)}, anchor={len(anchor_embeddings)})")

    # Evaluate each q_thresh combo at this step
    for q_thresh, step, sweep_score in schedule_entries:
        q_key = f"q>={q_thresh}"

        # ---- R@1 (closed-set, macro-averaged) ----
        q_mask = query_quality >= q_thresh
        mask_indices = np.where(q_mask)[0]
        n_query = int(q_mask.sum())

        if n_query > 0:
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
        else:
            recall_at_1 = 0.0

        # ---- BA (open-set) ----
        ba_metrics = compute_open_set_metrics_per_individual_threshold(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_vals,
            unknown_scores=scores_unknown,
            per_individual_thresholds=per_individual_thresholds,
            quality_thresholds=[q_thresh],
            gallery_labels=gallery_labels,
            class_to_name=class_to_name
        )

        ba_data = ba_metrics.get(q_key, {})
        ba = ba_data.get('balanced_accuracy', 0.0)
        kar = ba_data.get('known_accept_rate', 0.0)
        urr = ba_data.get('unknown_reject_rate', 0.0)
        n_known = ba_data.get('n_known_individuals', 0)
        n_unknown = ba_data.get('n_unknown_individuals', 0)

        print(f"    {q_key}: R@1={recall_at_1:.4f} (n={n_query}), "
              f"BA={ba:.4f} (K={kar:.2f}[{n_known}], U={urr:.2f}[{n_unknown}])")

        # ---- Save JSON ----
        result = {
            "config": {
                "gallery_threshold": gallery_threshold,
                "query_threshold": q_thresh,
                "best_step": step,
                "cv_recall_at_1": sweep_score,
                "learning_rate": best_lr,
                "image_size": best_size,
                "embedding_dim": best_embedding_dim,
                "backbone": model_config["backbone_label"],
                "augmentations": aug_flags,
                "loss": "ArcFace",
                "arcface_margin": ARCFACE_MARGIN,
                "arcface_scale": ARCFACE_SCALE,
                "seed": seed,
                "cosine_threshold": global_threshold,
                "per_individual_thresholds": per_individual_thresholds,
            },
            "dataset": {
                "individuals": feasible_individuals,
                "gallery_info": gallery_info,
                "query_info": query_info,
                "total_gallery": len(all_gallery_indices),
                "total_query_known": len(all_query_indices),
                "total_query_unknown": len(rare_indices) if rare_indices else 0,
                "n_unknown_individuals": len(set(rare_labels_str)) if rare_labels_str else 0,
            },
            "results": {
                "recall_at_1": recall_at_1,
                "balanced_accuracy": ba,
                "known_accept_rate": kar,
                "unknown_reject_rate": urr,
                "n_query": n_query,
                "n_known_individuals": n_known,
                "n_unknown_individuals": n_unknown,
            },
            "step_history": step_history,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "training_time_seconds": training_time,
            },
        }

        output_path = os.path.join(
            output_dir, f"g{gallery_threshold:.2f}_q{q_thresh:.2f}_seed={seed}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, np.floating)
                      else int(o) if isinstance(o, np.integer) else o)
        print(f"    Saved: {output_path}")

    model.train()
    arcface_loss.train()


def run_test_from_step_sweep(model_name, args, gallery_threshold):
    """
    Final test evaluation using per-query_q optimal checkpoints.

    Trains up to the max scheduled step, evaluating at each scheduled step
    for the query_q combos assigned to this gallery_threshold.
    Best steps are selected by max-mean R@1 across 5 seeds.
    """
    from utils.arcface import ArcFaceLoss, CoverageSampler
    from tqdm import tqdm

    model_config = MODEL_CONFIGS[model_name]
    experiment_dir = model_config["experiment_dir"]
    # Per-seed replicates live beside (not inside) test_eval/ so
    # aggregate_results.py's flat glob over test_eval/*.json never
    # double-counts them.
    output_dir = os.path.join(experiment_dir, "results", "test_eval_seeds")

    seed = args.seed

    # Load eval schedule
    print(f"\nLoading eval schedule for g>={gallery_threshold:.2f}:")
    schedule = load_eval_schedule(model_name, gallery_threshold)

    if not schedule:
        print(f"  No combos won for this config. Exiting gracefully.")
        return

    for q_thresh, step, score in schedule:
        print(f"  q>={q_thresh}: step={step}, cv_r1={score:.4f}")

    # Check for already-completed outputs and filter schedule
    remaining = []
    for entry in schedule:
        q_thresh, step, score = entry
        out_path = os.path.join(
            output_dir, f"g{gallery_threshold:.2f}_q{q_thresh:.2f}_seed={seed}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  SKIP (exists): {out_path}")
        else:
            remaining.append(entry)

    if not remaining:
        print("All outputs exist. Use --overwrite to replace.")
        return

    schedule = remaining

    # Group schedule by step
    step_to_evals = defaultdict(list)
    for entry in schedule:
        q_thresh, step, score = entry
        step_to_evals[step].append(entry)
    eval_steps = sorted(step_to_evals.keys())
    max_step = max(eval_steps)

    set_all_seeds(seed)
    # Deterministic mode is scoped to this subcommand: the frozen backbone
    # never sees a backward pass, so the deterministic kernels cost little
    # and each seed replicate is exactly re-runnable on one GPU model.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print(f"Test Step Eval: {model_config['backbone_label']}, "
          f"g>={gallery_threshold:.2f}, seed={seed}")
    print(f"  {len(schedule)} combos across {len(eval_steps)} eval steps, "
          f"max_step={max_step}")
    print("=" * 70)

    # Load best hyperparameters
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)
    aug_flags = {"blur": True, "jitter": True, "ir_sim": True, "hflip": True, "rotation": True}

    print(f"\nLoading datasets...")
    train_dataset = load_reidentification_dataset()
    train_metadata_cache = build_metadata_cache(train_dataset)
    test_dataset = load_reidentification_test_dataset()
    test_metadata_cache = build_metadata_cache(test_dataset)

    feasibility_config = load_feasibility_config()
    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    feasible_individuals = feasibility_config.get("qualified_individuals", [])
    promoted_individuals = feasibility_config.get("promoted_to_rare", [])
    excluded_individuals = feasibility_config.get("excluded_entirely", [])

    # Load unknown assignment: combined train+test rare source for BA
    unknown_config_path = 'preprocessing/results/unknown_assignment.json'
    try:
        with open(unknown_config_path, 'r') as f:
            unknown_config = json.load(f)
        from datasets import concatenate_datasets
        combined_rare_dataset = concatenate_datasets([train_dataset, test_dataset])
        combined_rare_metadata = build_metadata_cache(combined_rare_dataset)
        rare_excluded = ['HLC21-H1']
        print(f"  Combined rare pool: {unknown_config['summary']['combined_unknown_images']} images "
              f"from {unknown_config['summary']['n_unknown_individuals']} unknown individuals")
    except FileNotFoundError:
        print(f"Warning: {unknown_config_path} not found, BA will use test-split unknowns only")
        combined_rare_dataset = test_dataset
        combined_rare_metadata = test_metadata_cache
        rare_excluded = excluded_individuals

    if len(feasible_individuals) < 2:
        print(f"Not enough individuals ({len(feasible_individuals)}) for training (need >= 2)")
        return

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # ---- Gallery construction (full_filtered, no val carve-out) ----
    id_to_indices = train_metadata_cache['id_to_indices']
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}

    all_gallery_indices = []
    gallery_info = {}

    for ind_id in feasible_individuals:
        all_ind_indices = list(id_to_indices.get(ind_id, []))
        filtered_pool = filter_training_pool_by_quality(
            train_metadata_cache, all_ind_indices, gallery_threshold
        )
        sampled = filtered_pool
        gallery_info[ind_id] = {
            'sampled': len(sampled),
            'eligible': len(filtered_pool),
            'total': len(all_ind_indices),
            'mode': 'full_filtered',
        }
        if len(sampled) == 0:
            print(f"  WARNING: {ind_id} has NO samples above threshold {gallery_threshold}")
        else:
            print(f"  {ind_id}: {len(sampled)}/{len(all_ind_indices)} gallery "
                  f"(filtered >= {gallery_threshold})")
        all_gallery_indices.extend(sampled)

    gallery_dataset = train_dataset.select(all_gallery_indices)

    # ---- Query (known): ALL test images for known individuals ----
    test_id_to_indices = test_metadata_cache['id_to_indices']
    all_query_indices = []
    query_info = {}
    for ind_id in feasible_individuals:
        test_indices = test_id_to_indices.get(ind_id, [])
        all_query_indices.extend(test_indices)
        query_info[ind_id] = len(test_indices)

    query_dataset = test_dataset.select(all_query_indices)
    print(f"\nTotal: {len(all_gallery_indices)} gallery, {len(all_query_indices)} query (known)")

    # ---- Pre-fetch rare/unknown indices (used at every eval step) ----
    rare_indices, rare_quality_arr, rare_labels_str = get_rare_individual_indices(
        combined_rare_metadata, feasible_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=rare_excluded
    )
    rare_indices_cache = {
        'rare_indices': rare_indices,
        'rare_quality_arr': rare_quality_arr,
        'rare_labels_str': rare_labels_str,
    }

    # ---- Optional embedding export (anchor = last training event per individual) ----
    export_dir = None
    anchor_dataset = None
    export_meta = None
    if getattr(args, 'export_embeddings', False):
        export_dir = os.path.join(experiment_dir, "results", "embeddings")
        anchor_indices = []
        for ind_id in feasible_individuals:
            ind_indices = id_to_indices.get(ind_id, [])
            if not ind_indices:
                continue
            last_event = max(train_metadata_cache['ymdh'][i] for i in ind_indices)
            anchor_indices.extend(
                i for i in ind_indices if train_metadata_cache['ymdh'][i] == last_event
            )
        anchor_dataset = train_dataset.select(anchor_indices)
        export_meta = {}
        for prefix, cache, indices in [
            ('gallery', train_metadata_cache, all_gallery_indices),
            ('query', test_metadata_cache, all_query_indices),
            ('rare', combined_rare_metadata, rare_indices),
            ('anchor', train_metadata_cache, anchor_indices),
        ]:
            export_meta[f'{prefix}_filenames'] = np.asarray(cache['filenames'][indices], dtype=str)
            export_meta[f'{prefix}_ids'] = np.asarray(cache['ids'][indices], dtype=str)
            export_meta[f'{prefix}_ymdh'] = np.asarray(cache['ymdh'][indices])
            export_meta[f'{prefix}_quality'] = np.asarray(cache['quality_scores'][indices])
        print(f"  Embedding export ON: anchor={len(anchor_indices)} images "
              f"(last train event per individual) -> {export_dir}")

    # ---- Create model ----
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=best_embedding_dim,
                                           image_size=best_size, device=device)
    print(f"Using device: {device}")

    # ---- Transforms and training dataset ----
    eval_transform = create_transform_for_model(model_name, size=best_size)
    if any(aug_flags.values()):
        train_transform = create_train_transform_for_model(model_name, size=best_size, **aug_flags)
    else:
        train_transform = eval_transform
    train_torch = ArcFaceDataset(gallery_dataset, train_transform, individual_to_class)

    # ---- CoverageSampler ----
    sampler = CoverageSampler(
        labels=train_torch.get_labels(),
        batch_size=TRAIN_BATCH_SIZE,
    )
    class_sizes = {l: len(v) for l, v in sampler.label_to_all_indices.items()}
    print(f"  Sampler: CoverageSampler, batch_size={TRAIN_BATCH_SIZE}, "
          f"k={sampler.k}/class, pool min={min(class_sizes.values())}, "
          f"pool max={max(class_sizes.values())}")
    print(f"  Training for up to {max_step} gradient steps "
          f"(eval at steps: {eval_steps})")

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(train_torch, batch_sampler=sampler,
                              num_workers=0, collate_fn=collate_fn)

    # ---- ArcFace loss ----
    arcface_loss = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=emb_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE,
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(arcface_loss)
    print(f"Trainable: head={head_params:,}, arcface={arcface_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(arcface_loss.parameters()),
        lr=best_lr,
    )

    # ---- Training loop (step-based, with mid-training eval) ----
    print("\nWarming up dataloader (first batch)...", flush=True)
    train_iter = iter(train_loader)
    _warmup_batch = next(train_iter)
    del _warmup_batch
    print("Dataloader ready.")

    print(f"\nTraining for up to {max_step} steps...")
    start_time = time.time()
    step_history = []

    global_step = 0
    running_loss = 0.0
    running_batches = 0

    model.train()
    arcface_loss.train()

    pbar = tqdm(total=max_step, desc="Training", leave=True)
    for batch in train_iter:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)
        optimizer.zero_grad()
        loss = arcface_loss(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_batches += 1
        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{running_loss / running_batches:.4f}", step=global_step)

        # Log periodically
        if global_step % 50 == 0 or global_step == max_step:
            train_loss = running_loss / max(running_batches, 1)
            step_history.append({"step": global_step, "train_loss": train_loss})

        # Evaluate at scheduled steps
        if global_step in step_to_evals:
            elapsed = time.time() - start_time
            print(f"\n  Evaluating at step {global_step} "
                  f"({len(step_to_evals[global_step])} combos, "
                  f"{elapsed:.1f}s elapsed)...")
            _evaluate_and_save(
                model, arcface_loss, gallery_dataset, query_dataset,
                combined_rare_dataset, combined_rare_metadata, eval_transform,
                individual_to_class, feasible_individuals,
                promoted_individuals, rare_excluded,
                device, emb_dim, step_to_evals[global_step], output_dir,
                gallery_threshold, list(step_history),
                elapsed, best_lr, best_size, best_embedding_dim,
                aug_flags, model_config, seed, gallery_info, query_info,
                all_gallery_indices, all_query_indices, rare_indices_cache,
                export_dir=export_dir, anchor_dataset=anchor_dataset,
                export_meta=export_meta,
            )

        if global_step >= max_step:
            break

    pbar.close()
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")

    # Summary
    print(f"\nSummary: saved {len(schedule)} JSONs to {output_dir}")
    for q_thresh, step, score in schedule:
        print(f"  g{gallery_threshold:.2f}_q{q_thresh:.2f}.json "
              f"(step={step}, cv_r1={score:.4f})")


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/data/user_data/kdoherty/hf_cache")
    # Required for deterministic cuBLAS matmuls (test_step_eval turns on
    # torch.use_deterministic_algorithms); must be set before CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    parser = argparse.ArgumentParser(description="Reid open-set experiments")
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
    sub_lr.add_argument("--total-steps", type=int, default=100)
    sub_lr.add_argument("--eval-every", type=int, default=10)
    sub_lr.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # opt_embedding_dim subcommand
    sub_emb = subparsers.add_parser("opt_embedding_dim", help="Embedding dimension sweep")
    add_common_args(sub_emb)
    sub_emb.add_argument("--values", type=int, nargs="+", required=True, help="Embedding dims to sweep")
    sub_emb.add_argument("--lr", type=float, required=True, help="Fixed learning rate")
    sub_emb.add_argument("--image-size", type=int, required=True, help="Fixed image size")
    sub_emb.add_argument("--total-steps", type=int, default=100)
    sub_emb.add_argument("--eval-every", type=int, default=10)
    sub_emb.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # test_eval subcommand
    sub_test = subparsers.add_parser("test_eval", help="Final test evaluation")
    add_common_args(sub_test)

    # tune_step_count subcommand
    sub_step = subparsers.add_parser("tune_step_count",
                                      help="Step-budget training sweep (gallery threshold × seed)")
    add_common_args(sub_step)
    sub_step.add_argument("--dry-run", action="store_true",
                          help="Print config and exit without training")

    # test_step_eval subcommand
    sub_tstep = subparsers.add_parser("test_step_eval",
                                       help="Final test from step-count sweep")
    add_common_args(sub_tstep)
    sub_tstep.add_argument("--seed", type=int, default=0)
    sub_tstep.add_argument("--export-embeddings", action="store_true",
                           help="Save gallery/query/rare/anchor embeddings per "
                                "eval step to results/embeddings/")

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
        print(f"LR Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
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
        print(f"\nLR: {lr}, Seed: {seed}, Image size: {args.image_size}, Steps: {args.total_steps}")

        try:
            run_opt_training(
                model_name, "lr", lr, args, dataset, feasibility_config, metadata_cache,
                learning_rate=lr, image_size=args.image_size,
                embedding_dim=args.embedding_dim,
                total_steps=args.total_steps, eval_every=args.eval_every, seed=seed,
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
        print(f"Embedding Dim Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
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
        print(f"\nEmbedding dim: {emb_dim}, Seed: {seed}, LR: {args.lr}, Size: {args.image_size}, Steps: {args.total_steps}")

        try:
            run_opt_training(
                model_name, "embedding_dim", emb_dim, args, dataset, feasibility_config, metadata_cache,
                learning_rate=args.lr, image_size=args.image_size, embedding_dim=emb_dim,
                total_steps=args.total_steps, eval_every=args.eval_every, seed=seed,
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
        print(f"Final Test Evaluation - {config['backbone_label']} Re-ID - Job {args.idx}")
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

    elif args.command == "tune_step_count":
        print("=" * 80)
        print(f"Step-Budget Training Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
        print("=" * 80)

        try:
            run_step_count_sweep(model_name, args)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} (tune_step_count) completed!")

    elif args.command == "test_step_eval":
        if args.idx >= len(STEP_COUNT_GALLERY_THRESHOLDS):
            print(f"Job {args.idx} has no work ({len(STEP_COUNT_GALLERY_THRESHOLDS)} gallery thresholds)")
            sys.exit(0)

        gallery_threshold = STEP_COUNT_GALLERY_THRESHOLDS[args.idx]

        print("=" * 80)
        print(f"Test Step Eval - {config['backbone_label']} Re-ID - Job {args.idx}")
        print(f"  gallery_q>={gallery_threshold:.2f}")
        print("=" * 80)

        try:
            run_test_from_step_sweep(model_name, args, gallery_threshold)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} (test_step_eval) completed!")


#!/usr/bin/env python3
"""
Script 00: Quality-Weighted Triplet Loss Sweep
Re-identification experiment with triplet loss weighted by anchor image quality.

Hypothesis: High-quality anchor images (pelage visible) provide more reliable
training signal, improving individual wolverine re-identification.

Grid: 6 alpha values × 6 sample sizes × 8 seeds = 288 configurations
Distributed across 24 SLURM jobs (12 configs/job).
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
from PIL import Image

# Control parallelism and set HuggingFace cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

# Add project root to path
sys.path.append('.')

from utils.dataset import set_all_seeds
from utils.triplet import (
    create_embedding_model,
    QualityWeightedTripletLoss,
    PKBatchSampler,
    mine_random_triplets,
    compute_recall_at_k,
    compute_recall_at_k_by_quality_bin,
    compute_mean_average_precision
)
from utils.training import check_result_exists

import datasets
datasets.config.NUM_PROC = 1


# Experiment parameters
ALPHA_VALUES = [0, 1, 10, 100]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
MARGIN = 0.3
EMBEDDING_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_K = 8  # Samples per identity in PK batch
MIN_P = 5  # Minimum identities per batch


def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
    """Map job index to list of (alpha, sample_size, seed) tuples."""
    all_combinations = []
    for alpha in ALPHA_VALUES:
        for sample_size in SAMPLE_SIZES:
            for seed in SEEDS:
                all_combinations.append((alpha, sample_size, seed))

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


def load_feasibility_config():
    """Load the feasibility configuration from individual_id experiment."""
    config_path = 'individual_id/results/feasible_individuals.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Please run individual_id/00_count_individuals.py first.")
        return None


def load_reidentification_dataset():
    """Load the wolverines dataset with reidentification configuration."""
    print("Loading wolverines dataset (reidentification config)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    print(f"Loaded {len(dataset)} samples")
    return dataset


def build_id_to_indices(dataset):
    """Build individual ID to dataset indices mapping ONCE."""
    print("Building ID to indices mapping...")
    id_to_indices = {}
    for idx, sample in enumerate(dataset):
        ind_id = sample['id']
        if ind_id not in id_to_indices:
            id_to_indices[ind_id] = []
        id_to_indices[ind_id].append(idx)
    print(f"  Mapped {len(id_to_indices)} individuals")
    return id_to_indices


def create_temporal_dataset(dataset, individuals, sample_size, seed, config, id_to_indices):
    """
    Create temporal train/val split for triplet learning.

    Training: N samples per individual from all years except final
    Validation: All samples from final year (used as queries)
    """
    import random

    set_all_seeds(seed)

    # Pool all data
    print(f"Creating temporal dataset: {sample_size} samples/class, seed={seed}")

    # Build training and validation sets using pre-computed indices
    all_train_indices = []
    all_val_indices = []
    individual_to_class = {ind: i for i, ind in enumerate(sorted(individuals))}

    dataset_info = {}

    for ind_id in individuals:
        # Get validation indices from config
        val_indices = config['validation_indices'][ind_id]['indices']

        # Get all indices for this individual
        all_ind_indices = set(id_to_indices.get(ind_id, []))

        # Training candidates: exclude validation indices
        train_candidates = list(all_ind_indices - set(val_indices))

        if len(train_candidates) < sample_size:
            print(f"Warning: {ind_id} has only {len(train_candidates)} training samples, need {sample_size}")
            train_sampled = train_candidates
        else:
            random.shuffle(train_candidates)
            train_sampled = train_candidates[:sample_size]

        all_train_indices.extend(train_sampled)
        all_val_indices.extend(val_indices)

        dataset_info[ind_id] = {
            'train_samples': len(train_sampled),
            'val_samples': len(val_indices)
        }

        print(f"  {ind_id}: {len(train_sampled)} train, {len(val_indices)} val")

    # Create dataset subsets
    train_dataset = dataset.select(all_train_indices)
    val_dataset = dataset.select(all_val_indices)

    print(f"Total: {len(train_dataset)} train, {len(val_dataset)} val")

    return train_dataset, val_dataset, individual_to_class, dataset_info


class TripletDataset(TorchDataset):
    """PyTorch dataset for triplet learning with quality scores."""

    def __init__(self, hf_dataset, transform, individual_to_class):
        self.dataset = hf_dataset
        self.transform = transform
        self.individual_to_class = individual_to_class

        # Pre-compute labels and quality scores
        self.labels = []
        self.quality_scores = []
        for sample in hf_dataset:
            self.labels.append(individual_to_class[sample['id']])
            self.quality_scores.append(sample['pelage_score'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        label = self.individual_to_class[sample['id']]
        quality = sample['pelage_score']
        return image, label, quality

    def get_labels(self):
        return self.labels


def create_dinov3_transform():
    """Create DINOv3-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with triplet loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, labels, quality_scores) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        quality_scores = quality_scores.to(device).float()

        # Get embeddings
        embeddings = model(images)

        # Mine triplets
        anchor_emb, pos_emb, neg_emb, anchor_quality, pos_quality = mine_random_triplets(
            embeddings, labels, quality_scores
        )

        if anchor_emb.size(0) == 0:
            continue

        # Compute loss (product weighting: uses both anchor and positive quality)
        optimizer.zero_grad()
        loss = criterion(anchor_emb, pos_emb, neg_emb, anchor_quality, pos_quality)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, train_dataset, val_dataset, individual_to_class, transform, device, batch_size=32):
    """
    Evaluate model using Recall@K.

    Gallery: All training embeddings
    Queries: All validation embeddings
    """
    model.eval()

    # Create datasets
    train_torch = TripletDataset(train_dataset, transform, individual_to_class)
    val_torch = TripletDataset(val_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute gallery embeddings
    gallery_embeddings = []
    gallery_labels = []

    with torch.no_grad():
        for images, labels, _ in train_loader:
            images = images.to(device)
            emb = model(images)
            gallery_embeddings.append(emb.cpu())
            gallery_labels.extend(labels.tolist())

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.tensor(gallery_labels)

    # Compute query embeddings
    query_embeddings = []
    query_labels = []
    query_quality = []

    with torch.no_grad():
        for images, labels, quality in val_loader:
            images = images.to(device)
            emb = model(images)
            query_embeddings.append(emb.cpu())
            query_labels.extend(labels.tolist())
            query_quality.extend(quality.tolist())

    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels = torch.tensor(query_labels)
    query_quality = np.array(query_quality)

    # Compute metrics
    recall_at_1 = compute_recall_at_k(query_embeddings, gallery_embeddings,
                                      query_labels, gallery_labels, k=1)
    mAP = compute_mean_average_precision(query_embeddings, gallery_embeddings,
                                         query_labels, gallery_labels)

    # Compute by quality bin
    bin_metrics = compute_recall_at_k_by_quality_bin(
        query_embeddings, gallery_embeddings,
        query_labels, gallery_labels,
        query_quality, k=1
    )

    # Compute distances for raw predictions (optional, for post-hoc analysis)
    distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)

    return {
        'recall_at_1': recall_at_1,
        'mean_avg_precision': mAP,
        'by_quality_bin': bin_metrics,
        'raw_predictions': {
            'distances': distances.numpy().tolist(),
            'query_labels': query_labels.tolist(),
            'gallery_labels': gallery_labels.tolist(),
            'query_quality_scores': query_quality.tolist()
        }
    }


def train_single_config(alpha: float, sample_size: int, seed: int, args, dataset, config, id_to_indices) -> dict:
    """Train one configuration and return results."""
    set_all_seeds(seed)

    # Output path
    filename = f"alpha={alpha:.2f}_samples={sample_size}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)

    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: alpha={alpha}, samples={sample_size}, seed={seed}")
    print(f"{'='*60}")

    # Get valid individuals from config
    valid_individuals = config.get('valid_individuals', [])

    # Check feasibility for this sample size
    feasible_individuals = []
    for ind_id in valid_individuals:
        training_compat = config['training_compatibility'].get(ind_id, {})
        threshold_compat = training_compat.get('threshold_compatibility', {}).get('threshold_0.00', {})
        compatible_sizes = threshold_compat.get('compatible_training_sizes', [])
        if sample_size in compatible_sizes:
            feasible_individuals.append(ind_id)

    if len(feasible_individuals) < MIN_P:
        print(f"Not enough individuals ({len(feasible_individuals)}) for PK sampling (need {MIN_P})")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create temporal dataset
    train_dataset, val_dataset, individual_to_class, dataset_info = create_temporal_dataset(
        dataset, feasible_individuals, sample_size, seed, config, id_to_indices
    )

    # Use effective_k based on sample_size for small datasets
    effective_k = min(BATCH_K, sample_size)
    if len(train_dataset) < MIN_P * effective_k:
        print(f"Not enough training samples ({len(train_dataset)}) for PK batching")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model = create_embedding_model(embedding_dim=EMBEDDING_DIM, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_dinov3_transform()
    train_torch_dataset = TripletDataset(train_dataset, transform, individual_to_class)

    # Create PK batch sampler
    try:
        pk_sampler = PKBatchSampler(
            labels=train_torch_dataset.get_labels(),
            p=min(MIN_P, len(feasible_individuals)),
            k=effective_k,
            drop_last=True
        )
    except ValueError as e:
        print(f"Cannot create PK sampler: {e}")
        return None

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=pk_sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create loss and optimizer
    criterion = QualityWeightedTripletLoss(margin=MARGIN, alpha=alpha)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LEARNING_RATE)

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate every epoch for best epoch selection
        metrics = evaluate(model, train_dataset, val_dataset, individual_to_class,
                           transform, device)
        recall_1 = metrics['recall_at_1']

        if recall_1 > best_recall:
            best_recall = recall_1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Loss={train_loss:.4f}, R@1={recall_1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_recall_at_1': recall_1,
            'learning_rate': LEARNING_RATE
        })

    training_time = time.time() - start_time

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, train_dataset, val_dataset, individual_to_class,
                             transform, device)

    # Prepare result
    result = {
        'config': {
            'alpha': alpha,
            'samples_per_class': sample_size,
            'seed': seed,
            'margin': MARGIN,
            'embedding_dim': EMBEDDING_DIM,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size_k': BATCH_K
        },
        'dataset': {
            'individuals': feasible_individuals,
            'train_samples_per_individual': {k: v['train_samples'] for k, v in dataset_info.items()},
            'val_samples_per_individual': {k: v['val_samples'] for k, v in dataset_info.items()},
            'gallery_size': len(train_dataset),
            'query_size': len(val_dataset)
        },
        'final_metrics': {
            'recall_at_1': final_metrics['recall_at_1'],
            'mean_avg_precision': final_metrics['mean_avg_precision'],
            'by_quality_bin': final_metrics['by_quality_bin']
        },
        'epoch_history': epoch_history,
        'raw_predictions': final_metrics['raw_predictions'],
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx,
            'transform': 'resize_224_imagenet_norm',
            'best_epoch': best_epoch,
            'best_recall_at_1': best_recall
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Final R@1={final_metrics['recall_at_1']:.4f}, mAP={final_metrics['mean_avg_precision']:.4f}")
    print(f"By quality bin:")
    for bin_name, bin_data in final_metrics['by_quality_bin'].items():
        if bin_data['count'] > 0:
            print(f"  {bin_name}: R@1={bin_data['recall_at_1']:.4f} (n={bin_data['count']})")

    return filename


def main():
    parser = argparse.ArgumentParser(description='Quality-weighted triplet loss sweep')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_triplet/results/triplet_sweep',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Quality-Weighted Triplet Loss Experiment - Job {args.idx}")
    print("=" * 80)

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    id_to_indices = build_id_to_indices(dataset)  # Build index lookup ONCE
    config = load_feasibility_config()

    if not config:
        print("Failed to load feasibility config")
        return

    # Show experiment info
    total_combinations = len(ALPHA_VALUES) * len(SAMPLE_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Alpha values: {ALPHA_VALUES}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")

    # Get combinations for this job
    combinations = get_job_combinations(args.idx)

    if not combinations:
        print(f"\nNo combinations assigned to job {args.idx}")
        return

    print(f"\nJob {args.idx} processing {len(combinations)} configurations:")
    for alpha, sample_size, seed in combinations[:5]:
        print(f"  alpha={alpha}, samples={sample_size}, seed={seed}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more")

    # Train each configuration
    results_summary = []
    for i, (alpha, sample_size, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(alpha, sample_size, seed, args, dataset, config, id_to_indices)
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


if __name__ == "__main__":
    main()

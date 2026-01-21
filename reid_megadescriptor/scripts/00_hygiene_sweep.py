#!/usr/bin/env python3
"""
Script 00: Gallery Hygiene Sweep with MegaDescriptor Backbone

Replicates reid_hygiene_filter experiment using MegaDescriptor-L-384 backbone
instead of DINOv3.

Key differences from DINOv3:
- Input size: 384x384 (vs 224x224)
- Normalization: [0.5,0.5,0.5] (vs ImageNet stats)
- Feature extraction: Direct output (vs CLS token)
- Feature dimension: ~2048 (vs 768)

Grid: 6 thresholds x 6 gallery sizes x 8 seeds = 288 configurations
Distributed across 24 SLURM jobs (12 configs/job).
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset
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
    create_megadescriptor_embedding_model,
    PKBatchSampler,
    mine_random_triplets,
    compute_recall_at_k,
)
from utils.training import check_result_exists

import datasets
datasets.config.NUM_PROC = 1


# Experiment parameters
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0.0 = no filtering (baseline)
GALLERY_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
MARGIN = 0.3
EMBEDDING_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_K = 8  # Samples per identity in PK batch
MIN_P = 5  # Minimum identities per batch

# Query quality thresholds for evaluation
# q>=0.0 includes ALL queries (equivalent to overall recall)
QUERY_QUALITY_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def get_job_combinations(job_idx: int, max_jobs: int = 24) -> list:
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


def filter_training_pool_by_quality(dataset, indices, threshold):
    """
    Return indices where pelage_score >= threshold.

    Args:
        dataset: HuggingFace dataset
        indices: List of dataset indices to filter
        threshold: Minimum pelage_score to include

    Returns:
        List of indices meeting the threshold
    """
    if threshold <= 0.0:
        return indices  # No filtering needed for threshold 0.0

    filtered = []
    for idx in indices:
        if dataset[idx]['pelage_score'] >= threshold:
            filtered.append(idx)
    return filtered


def create_filtered_gallery_dataset(dataset, individuals, gallery_size, threshold, seed, config, id_to_indices):
    """
    Create gallery/query split with filtered gallery.

    Gallery: Training samples filtered by pelage_score >= threshold, then sampled
    Query: ALL validation samples (unfiltered)

    Returns:
        train_dataset: Gallery dataset (filtered + sampled)
        val_dataset: Query dataset (all validation samples)
        individual_to_class: Label mapping
        dataset_info: Statistics about the split
    """
    import random

    set_all_seeds(seed)

    print(f"Creating filtered gallery: threshold={threshold}, gallery_size={gallery_size}, seed={seed}")

    all_train_indices = []
    all_val_indices = []
    individual_to_class = {ind: i for i, ind in enumerate(sorted(individuals))}

    dataset_info = {
        'gallery_samples_per_individual': {},
        'eligible_pool_per_individual': {},
        'query_samples_per_individual': {}
    }

    for ind_id in individuals:
        # Get validation indices (queries) - these are UNFILTERED
        val_indices = config['validation_indices'][ind_id]['indices']
        all_val_indices.extend(val_indices)
        dataset_info['query_samples_per_individual'][ind_id] = len(val_indices)

        # Get all indices for this individual
        all_ind_indices = set(id_to_indices.get(ind_id, []))

        # Training candidates: exclude validation indices
        train_candidates = list(all_ind_indices - set(val_indices))

        # FILTER by quality threshold
        eligible_pool = filter_training_pool_by_quality(dataset, train_candidates, threshold)
        dataset_info['eligible_pool_per_individual'][ind_id] = len(eligible_pool)

        # Sample from filtered pool
        if len(eligible_pool) == 0:
            print(f"  WARNING: {ind_id} has NO samples above threshold {threshold}")
            train_sampled = []
        elif len(eligible_pool) < gallery_size:
            print(f"  WARNING: {ind_id} has only {len(eligible_pool)} eligible samples (need {gallery_size}), using all")
            train_sampled = eligible_pool
        else:
            random.shuffle(eligible_pool)
            train_sampled = eligible_pool[:gallery_size]

        all_train_indices.extend(train_sampled)
        dataset_info['gallery_samples_per_individual'][ind_id] = len(train_sampled)

        print(f"  {ind_id}: {len(train_sampled)}/{len(eligible_pool)} gallery (threshold>={threshold}), {len(val_indices)} query")

    # Create dataset subsets
    train_dataset = dataset.select(all_train_indices) if all_train_indices else None
    val_dataset = dataset.select(all_val_indices)

    print(f"Total: {len(all_train_indices)} gallery, {len(all_val_indices)} query")

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


def create_megadescriptor_transform():
    """Create MegaDescriptor-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(384, 384), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with standard triplet loss (no quality weighting)."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, labels, quality_scores) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        quality_scores = quality_scores.to(device).float()

        # Get embeddings
        embeddings = model(images)

        # Mine triplets (quality scores returned but not used for weighting)
        anchor_emb, pos_emb, neg_emb, _, _ = mine_random_triplets(
            embeddings, labels, quality_scores
        )

        if anchor_emb.size(0) == 0:
            continue

        # Standard triplet loss - NO quality weighting
        optimizer.zero_grad()
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def compute_recall_by_query_quality(query_emb, gallery_emb, query_labels, gallery_labels, query_quality):
    """
    Compute Recall@1 for queries at different quality thresholds.

    q>=0.0 includes ALL queries (equivalent to overall recall).

    Args:
        query_emb: Query embeddings tensor
        gallery_emb: Gallery embeddings tensor
        query_labels: Query labels tensor
        gallery_labels: Gallery labels tensor
        query_quality: Query quality scores (numpy array)

    Returns:
        dict: {"q>=0.0": {"recall_at_1": float, "count": int}, "q>=0.1": {...}, ...}
    """
    results = {}
    query_quality = np.array(query_quality)

    for thresh in QUERY_QUALITY_THRESHOLDS:
        mask = query_quality >= thresh
        count = int(mask.sum())

        if count == 0:
            results[f"q>={thresh}"] = {"recall_at_1": 0.0, "count": 0}
            continue

        # Filter queries by quality threshold
        filtered_query_emb = query_emb[mask]
        filtered_query_labels = query_labels[torch.tensor(mask)] if isinstance(query_labels, torch.Tensor) else query_labels[mask]

        # Compute recall against FULL gallery (already filtered by experiment threshold)
        recall = compute_recall_at_k(filtered_query_emb, gallery_emb,
                                     filtered_query_labels, gallery_labels, k=1)
        results[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    return results


def evaluate_recall(model, train_dataset, val_dataset, individual_to_class, transform, device, batch_size=32):
    """
    Evaluate model computing Recall@1 for different query quality thresholds.

    Gallery: All training embeddings (filtered by threshold)
    Queries: All validation embeddings (unfiltered - includes all quality levels)

    Returns:
        dict: query_quality_metrics with recall at each quality threshold
              (q>=0.0 is the overall recall)
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

    # Compute query embeddings and collect quality scores
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

    # Compute recall by query quality thresholds
    query_quality_metrics = compute_recall_by_query_quality(
        query_embeddings, gallery_embeddings,
        query_labels, gallery_labels,
        query_quality
    )

    return query_quality_metrics


def train_single_config(threshold: float, gallery_size: int, seed: int, args, dataset, config, id_to_indices) -> dict:
    """Train one configuration and return results."""
    set_all_seeds(seed)

    # Output path
    filename = f"threshold={threshold:.2f}_gallery={gallery_size}_seed={seed}.json"
    output_path = os.path.join(args.output_dir, filename)

    if check_result_exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: threshold={threshold}, gallery_size={gallery_size}, seed={seed}")
    print(f"{'='*60}")

    # Get valid individuals from config
    valid_individuals = config.get('valid_individuals', [])

    # Check feasibility for this threshold and gallery_size
    feasible_individuals = []
    for ind_id in valid_individuals:
        training_compat = config['training_compatibility'].get(ind_id, {})

        # Find the appropriate threshold key (config uses 0.00, 0.25, 0.50, 0.75)
        # Our thresholds are finer-grained, so use the floor
        threshold_key = f"threshold_{int(threshold * 4) * 0.25:.2f}"
        if threshold_key not in training_compat.get('threshold_compatibility', {}):
            threshold_key = "threshold_0.00"  # Fallback to no filtering

        threshold_compat = training_compat.get('threshold_compatibility', {}).get(threshold_key, {})
        compatible_sizes = threshold_compat.get('compatible_training_sizes', [])

        if gallery_size in compatible_sizes:
            feasible_individuals.append(ind_id)

    if len(feasible_individuals) < MIN_P:
        print(f"Not enough individuals ({len(feasible_individuals)}) for PK sampling (need {MIN_P})")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create filtered gallery dataset
    train_dataset, val_dataset, individual_to_class, dataset_info = create_filtered_gallery_dataset(
        dataset, feasible_individuals, gallery_size, threshold, seed, config, id_to_indices
    )

    if train_dataset is None or len(train_dataset) == 0:
        print(f"No training data available after filtering")
        return None

    # Use effective_k based on actual samples available
    effective_k = min(BATCH_K, gallery_size)
    if len(train_dataset) < MIN_P * effective_k:
        print(f"Not enough training samples ({len(train_dataset)}) for PK batching")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model = create_megadescriptor_embedding_model(embedding_dim=EMBEDDING_DIM, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_megadescriptor_transform()
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

    # Create STANDARD triplet loss (no quality weighting)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LEARNING_RATE)

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluation - returns metrics for each query quality threshold
        metrics = evaluate_recall(model, train_dataset, val_dataset,
                                  individual_to_class, transform, device)

        # Overall recall is at q>=0.0 (includes all queries)
        recall_1 = metrics["q>=0.0"]["recall_at_1"]

        if recall_1 > best_recall:
            best_recall = recall_1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Loss={train_loss:.4f}, R@1={recall_1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'query_quality_metrics': metrics  # q>=0.0 is the overall recall
        })

    training_time = time.time() - start_time

    # Prepare result
    result = {
        'config': {
            'threshold': threshold,
            'gallery_size': gallery_size,
            'seed': seed,
            'margin': MARGIN,
            'embedding_dim': EMBEDDING_DIM,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size_k': BATCH_K,
            'backbone': 'MegaDescriptor-L-384'
        },
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
            'transform': 'resize_384_megadescriptor_norm',
            'best_epoch': best_epoch,
            'best_recall_at_1': best_recall
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best epoch: {best_epoch}, R@1={best_recall:.4f}")

    return filename


def main():
    parser = argparse.ArgumentParser(description='Gallery hygiene sweep with MegaDescriptor backbone')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_megadescriptor/results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"MegaDescriptor Gallery Hygiene Sweep Experiment - Job {args.idx}")
    print("=" * 80)

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    id_to_indices = build_id_to_indices(dataset)
    config = load_feasibility_config()

    if not config:
        print("Failed to load feasibility config")
        return

    # Show experiment info
    total_combinations = len(THRESHOLDS) * len(GALLERY_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Backbone: MegaDescriptor-L-384")
    print(f"  Input size: 384x384")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")

    # Get combinations for this job
    combinations = get_job_combinations(args.idx)

    if not combinations:
        print(f"\nNo combinations assigned to job {args.idx}")
        return

    print(f"\nJob {args.idx} processing {len(combinations)} configurations:")
    for threshold, gallery_size, seed in combinations[:5]:
        print(f"  threshold={threshold}, gallery_size={gallery_size}, seed={seed}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more")

    # Train each configuration
    results_summary = []
    for i, (threshold, gallery_size, seed) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        try:
            result = train_single_config(threshold, gallery_size, seed, args, dataset, config, id_to_indices)
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

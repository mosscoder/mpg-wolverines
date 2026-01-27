#!/usr/bin/env python3
"""
Script 00: Open-Set Gallery Hygiene Sweep - DINOv3 + ArcFace

Extends reid_dinov3_arcface with open-set evaluation:
1. At each epoch, calibrate optimal distance threshold using validation set
2. Apply threshold to rare/unknown individuals from the broader dataset
3. Measure Correct Flag Rate - proportion of unknowns correctly rejected

Key metrics:
- Closed-set: Recall@1 on validation (known individuals)
- Open-set: Correct Flag Rate on rare unknowns (never seen during training)

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
from transformers import AutoModel

# Control parallelism and set HuggingFace cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

# Add project root to path
sys.path.append('.')

from utils.dataset import set_all_seeds
from utils.triplet import (
    ArcFaceLoss,
    PKBatchSampler,
    compute_recall_at_k,
    calibrate_threshold_loo,
    evaluate_open_set_balanced,
)
from utils.training import check_result_exists

import datasets
datasets.config.NUM_PROC = 1


class EmbeddingHead(nn.Module):
    """Trainable projection head for ArcFace."""
    def __init__(self, input_dim: int = 768, embedding_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# Experiment parameters
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0.0 = no filtering (baseline)
GALLERY_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

# ArcFace hyperparameters
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_K = 8  # Samples per identity in PK batch
MIN_P = 5  # Minimum identities per batch
EMBEDDING_DIM = 128  # Match reid_hygiene_filter for direct comparison

# Query quality thresholds for evaluation
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
    """Load the feasibility configuration from preprocessing."""
    config_path = 'preprocessing/results/feasible_individuals.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Run: python preprocessing/create_validation_splits.py")
        return None


def load_reidentification_dataset():
    """Load the wolverines dataset with reidentification configuration."""
    print("Loading wolverines dataset (reidentification config)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    print(f"Loaded {len(dataset)} samples")
    return dataset


def build_metadata_cache(dataset):
    """Build metadata cache using Arrow columnar access (optimized)."""
    from utils.arrow_cache import build_metadata_cache_arrow
    return build_metadata_cache_arrow(dataset)


def filter_training_pool_by_quality(metadata_cache, indices, threshold):
    """Use vectorized filtering with cached quality scores."""
    from utils.optimized_filters import filter_training_pool_by_quality_vectorized
    return filter_training_pool_by_quality_vectorized(
        metadata_cache['quality_scores'],
        indices,
        threshold
    )


def get_rare_individual_indices(metadata_cache, valid_individuals: list, quality_threshold: float = 0.0):
    """Use vectorized operations with metadata cache."""
    from utils.optimized_filters import get_rare_individual_indices_vectorized
    return get_rare_individual_indices_vectorized(
        metadata_cache,
        valid_individuals,
        quality_threshold
    )


def create_filtered_gallery_dataset(dataset, individuals, gallery_size, threshold, seed, config, metadata_cache):
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

    id_to_indices = metadata_cache['id_to_indices']

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

        # FILTER by quality threshold (VECTORIZED)
        eligible_pool = filter_training_pool_by_quality(metadata_cache, train_candidates, threshold)
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


class ArcFaceDataset(TorchDataset):
    """PyTorch dataset for ArcFace learning with quality scores."""

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


class RareIndividualsDataset(TorchDataset):
    """PyTorch dataset for rare/unknown individuals (open-set evaluation)."""

    def __init__(self, hf_dataset, indices, quality_scores, transform):
        self.dataset = hf_dataset
        self.indices = indices
        self.quality_scores = quality_scores
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        image = self.transform(sample['image'])
        quality = self.quality_scores[idx]
        return image, quality


def create_dinov3_transform():
    """Create DINOv3-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_dinov3_arcface_model(embedding_dim: int = 128, device="cuda"):
    """
    Create DINOv3 backbone with trainable projection head.

    Architecture: Frozen DINOv3 -> 768-d CLS -> EmbeddingHead (768->128) -> ArcFace

    The trainable projection head allows embeddings to improve during training,
    fixing the issue where recall was static with only frozen backbone outputs.

    Returns:
        Tuple of (model, embedding_dim)
    """
    backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Trainable projection head (768 -> 128 to match reid_hygiene_filter)
    head = EmbeddingHead(input_dim=768, embedding_dim=embedding_dim)

    class DINOv3WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                outputs = self.backbone(x)
                features = outputs.last_hidden_state[:, 0, :]
            return self.head(features)  # Trainable transformation

        def get_trainable_parameters(self):
            return self.head.parameters()

    model = DINOv3WithHead(backbone, head)

    # Move to device
    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    return model.to(device), embedding_dim


def train_epoch_arcface(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with ArcFace loss."""
    model.train()  # Projection head is trainable
    criterion.train()  # ArcFace weights are trainable
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Get embeddings (frozen backbone + trainable head)
        embeddings = model(images)

        # Compute ArcFace loss
        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
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


def compute_rare_embeddings(model, dataset, rare_indices, rare_quality, rare_labels, transform, device, batch_size=32):
    """
    Compute embeddings for rare/unknown individuals.

    Args:
        model: Embedding model
        dataset: HuggingFace dataset
        rare_indices: List of dataset indices for rare individuals
        rare_quality: List of quality scores for rare individuals
        rare_labels: List of individual IDs for rare individuals
        transform: Image transform
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        Tuple of (embeddings tensor, quality array, labels array)
    """
    model.eval()

    rare_dataset = RareIndividualsDataset(dataset, rare_indices, rare_quality, transform)
    rare_loader = DataLoader(rare_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = []
    qualities = []

    with torch.no_grad():
        for images, quality in rare_loader:
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu())
            qualities.extend(quality.tolist())

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
    else:
        embeddings = torch.empty(0, EMBEDDING_DIM)

    return embeddings, np.array(qualities), np.array(rare_labels)


def evaluate_recall_with_openset(model, train_dataset, val_dataset, individual_to_class,
                                  transform, device, dataset, valid_individuals, metadata_cache,
                                  gallery_threshold, batch_size=32):
    """
    Evaluate model computing Recall@1 and open-set metrics.

    Gallery: All training embeddings (filtered by gallery_threshold at training time)
    Queries: All validation embeddings (filtered at eval time by quality thresholds)
    Rare: All individuals not in valid_individuals (filtered at eval time by quality thresholds)

    Both closed-set and open-set metrics are computed across quality thresholds
    (0.0, 0.1, 0.2, 0.3, 0.4, 0.5).

    Threshold calibration: LOO within training gallery (no data leakage)
    Open-set metric: Balanced accuracy = (known_accept_rate + unknown_reject_rate) / 2

    Returns:
        dict: query_quality_metrics with recall at each quality threshold
              open_set metrics with balanced accuracy at each quality threshold
    """
    model.eval()

    # Create datasets
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    val_torch = ArcFaceDataset(val_dataset, transform, individual_to_class)

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

    # Compute recall by query quality thresholds (closed-set)
    query_quality_metrics = compute_recall_by_query_quality(
        query_embeddings, gallery_embeddings,
        query_labels, gallery_labels,
        query_quality
    )

    # --- Open-set evaluation with LOO threshold calibration ---

    # 1. Calibrate threshold via LOO within training gallery (no data leakage)
    optimal_thresh, thresh_std, loo_metrics = calibrate_threshold_loo(
        gallery_embeddings, gallery_labels
    )

    # 2. Get ALL rare/unknown individuals (no quality filtering - filter at eval time)
    rare_indices, rare_quality, rare_labels = get_rare_individual_indices(
        metadata_cache, valid_individuals,
        quality_threshold=0.0  # Get all, filter at eval time
    )

    # 3. Compute embeddings for rare individuals
    if rare_indices:
        rare_emb, rare_quality_arr, rare_labels_arr = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels, transform, device
        )
    else:
        rare_emb = torch.empty(0, EMBEDDING_DIM)
        rare_quality_arr = np.array([])
        rare_labels_arr = np.array([])

    # 4. Evaluate open-set with balanced accuracy at each quality threshold
    # Uses macro-averaging: per-individual rates are computed, then averaged
    balanced_metrics_by_quality = evaluate_open_set_balanced(
        known_query_emb=query_embeddings,
        known_query_labels=query_labels,
        known_query_quality=query_quality,
        unknown_query_emb=rare_emb,
        unknown_query_quality=rare_quality_arr,
        gallery_emb=gallery_embeddings,
        gallery_labels=gallery_labels,
        distance_threshold=optimal_thresh,
        quality_thresholds=QUERY_QUALITY_THRESHOLDS,
        unknown_query_labels=rare_labels_arr
    )

    open_set_metrics = {
        'threshold_calibration': {
            'method': 'loo',
            'threshold_mean': optimal_thresh,
            'threshold_std': thresh_std,
            'n_folds': loo_metrics.get('n_folds', 0)
        },
        'by_quality': balanced_metrics_by_quality
    }

    return query_quality_metrics, open_set_metrics


def train_single_config(threshold: float, gallery_size: int, seed: int, args, dataset, config, metadata_cache) -> dict:
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

        # Threshold key matches the config thresholds directly
        threshold_key = f"threshold_{threshold:.2f}"
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
        dataset, feasible_individuals, gallery_size, threshold, seed, config, metadata_cache
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
    model, embedding_dim = create_dinov3_arcface_model(device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_dinov3_transform()
    train_torch_dataset = ArcFaceDataset(train_dataset, transform, individual_to_class)

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

    # Create ArcFace loss with learnable class centers
    criterion = ArcFaceLoss(
        num_classes=len(feasible_individuals),
        embedding_size=embedding_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    # Train both projection head and ArcFace class centers
    optimizer = torch.optim.SGD(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        momentum=0.9
    )

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0
    best_balanced_accuracy = 0.0
    best_ba_epoch = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        # Evaluation - returns metrics for closed-set and open-set
        query_quality_metrics, open_set_metrics = evaluate_recall_with_openset(
            model, train_dataset, val_dataset, individual_to_class, transform, device,
            dataset, feasible_individuals, metadata_cache, gallery_threshold=threshold
        )

        # Overall recall is at q>=0.0 (includes all queries)
        recall_1 = query_quality_metrics["q>=0.0"]["recall_at_1"]

        if recall_1 > best_recall:
            best_recall = recall_1
            best_epoch = epoch + 1

        # Track best balanced accuracy for open-set (at q>=0.0 for overall metric)
        by_quality = open_set_metrics.get('by_quality', {})
        ba_q0 = by_quality.get('q>=0.0', {}).get('balanced_accuracy', 0.0)
        if ba_q0 > best_balanced_accuracy:
            best_balanced_accuracy = ba_q0
            best_ba_epoch = epoch + 1

        thresh_mean = open_set_metrics.get('threshold_calibration', {}).get('threshold_mean', 0.0)
        known_rate = by_quality.get('q>=0.0', {}).get('known_accept_rate', 0.0)
        unknown_rate = by_quality.get('q>=0.0', {}).get('unknown_reject_rate', 0.0)

        print(f"Epoch {epoch+1:3d}/{EPOCHS}: Loss={train_loss:.4f}, R@1={recall_1:.4f}, "
              f"BA={ba_q0:.4f} (K={known_rate:.2f}, U={unknown_rate:.2f}), thresh={thresh_mean:.3f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': LEARNING_RATE,  # Fixed LR (no scheduler)
            'query_quality_metrics': query_quality_metrics,  # q>=0.0 is the overall recall
            'open_set': open_set_metrics  # Open-set evaluation metrics with balanced accuracy
        })

    training_time = time.time() - start_time

    # Prepare result
    result = {
        'config': {
            'threshold': threshold,
            'gallery_size': gallery_size,
            'seed': seed,
            'loss': 'ArcFace',
            'arcface_margin': ARCFACE_MARGIN,
            'arcface_scale': ARCFACE_SCALE,
            'embedding_dim': EMBEDDING_DIM,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'scheduler': 'None (fixed LR)',
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size_k': BATCH_K,
            'backbone': 'DINOv3-ViT-B/16'
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
            'transform': 'resize_224_imagenet_norm',
            'best_epoch': best_epoch,
            'best_recall_at_1': best_recall,
            'best_ba_epoch': best_ba_epoch,
            'best_balanced_accuracy': best_balanced_accuracy
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best R@1 epoch: {best_epoch}, R@1={best_recall:.4f}")
    print(f"Best BA epoch: {best_ba_epoch}, BA={best_balanced_accuracy:.4f}")

    return filename


def main():
    parser = argparse.ArgumentParser(description='Open-Set DINOv3 + ArcFace Gallery Hygiene Sweep')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_arcface/results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Open-Set DINOv3 + ArcFace Gallery Hygiene Sweep - Job {args.idx}")
    print("=" * 80)

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    config = load_feasibility_config()

    if not config:
        print("Failed to load feasibility config")
        return

    # Show experiment info
    total_combinations = len(THRESHOLDS) * len(GALLERY_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    print(f"  Optimizer: SGD (momentum=0.9, lr={LEARNING_RATE}, fixed)")
    print(f"  Embedding: {EMBEDDING_DIM}-d (trainable projection from DINOv3 CLS)")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")
    print(f"  Open-set evaluation: Enabled (rare individuals)")

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
            result = train_single_config(threshold, gallery_size, seed, args, dataset, config, metadata_cache)
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

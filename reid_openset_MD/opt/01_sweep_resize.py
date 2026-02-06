#!/usr/bin/env python3
"""
Script 01: Image Size Sweep for MegaDescriptor Re-ID with Temporal (ymdh) Split

Uses frozen MegaDescriptor-L-384 backbone with ArcFace training and temporal (ymdh) train/test split.
Sweeps over 17 image sizes with fixed LR=5e-4.

Key design:
- Temporal split by capture event (ymdh) for each individual
- Earlier 50% of ymdh values -> train pool
- Later 50% of ymdh values -> test pool
- Train gets priority: sample 32 (or all if <32)
- Test gets remainder: sample 32 (or all if <32)
- Hygiene validation indices are EXCLUDED before any splitting

Grid: 17 image sizes (256-512, step 16) = 17 total configurations
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict

# Control parallelism and set HuggingFace cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

# Add project root to path
sys.path.append('.')

from utils.triplet import ArcFaceLoss, PKBatchSampler, create_megadescriptor_arcface_model
from utils.dataset import set_all_seeds

import datasets
datasets.config.NUM_PROC = 1


# Experiment parameters
RESIZE_SIZES = list(range(256, 513, 16))  # 256 to 512, step 16 = 17 sizes
LEARNING_RATE = 5e-4  # Fixed LR
SEED = 0
EPOCHS = 50
TARGET_SAMPLES_PER_INDIVIDUAL = 32

# ArcFace hyperparameters
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64
BATCH_K = 8  # Samples per identity in PK batch
MIN_P = 5    # Minimum identities per batch
EMBEDDING_DIM = 128


class ArcFaceDataset(TorchDataset):
    """PyTorch dataset for ArcFace learning."""

    def __init__(self, hf_dataset, transform, individual_to_class):
        self.dataset = hf_dataset
        self.transform = transform
        self.individual_to_class = individual_to_class

        # Pre-compute labels
        self.labels = []
        for sample in hf_dataset:
            self.labels.append(individual_to_class[sample['id']])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        label = self.individual_to_class[sample['id']]
        return image, label

    def get_labels(self):
        return self.labels


def create_megadescriptor_transform(size=384):
    """Create MegaDescriptor-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(size, size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_arcface(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with ArcFace loss."""
    model.train()
    criterion.train()
    total_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)

        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def load_feasibility_config():
    """Load the feasibility configuration from preprocessing."""
    config_path = 'preprocessing/results/feasible_individuals.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return None


def load_reidentification_dataset():
    """Load the wolverines dataset with reidentification configuration."""
    print("Loading wolverines dataset (reidentification config)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="train")
    print(f"Loaded {len(dataset)} samples")
    return dataset


def build_metadata_cache(dataset):
    """Build metadata cache including ymdh for temporal splitting."""
    n = len(dataset)

    # Extract columns using Arrow columnar access
    ids = np.array(dataset['id'], dtype=object)
    ymdh = np.array(dataset['ymdh'], dtype=np.int64)

    # Build ID-to-indices mapping
    id_to_indices = defaultdict(list)
    for idx, ind_id in enumerate(ids):
        id_to_indices[ind_id].append(idx)

    return {
        'ids': ids,
        'ymdh': ymdh,
        'id_to_indices': dict(id_to_indices),
        'dataset_size': n
    }


def create_ymdh_split_dataset(dataset, individuals, metadata_cache, config, seed=0):
    """
    Create train/test split based on capture events (ymdh) with no hygiene val leakage.

    For each individual:
    1. Get all sample indices
    2. EXCLUDE indices in config['validation_indices'][ind_id]['indices'] (hygiene val)
    3. From remaining, get unique ymdh values and sort chronologically
    4. Earlier 50% of ymdh -> train pool
    5. Later 50% of ymdh -> test pool
    6. Train gets priority: sample TARGET_SAMPLES_PER_INDIVIDUAL (or all available if <32)
    7. Test gets remainder: sample TARGET_SAMPLES_PER_INDIVIDUAL (or all available if <32)

    Returns: train_dataset, test_dataset, individual_to_class, dataset_info
    """
    import random

    set_all_seeds(seed)

    ymdh_arr = metadata_cache['ymdh']
    id_to_indices = metadata_cache['id_to_indices']

    print(f"Creating ymdh temporal split: seed={seed}")

    all_train_indices = []
    all_test_indices = []
    individual_to_class = {ind: i for i, ind in enumerate(sorted(individuals))}

    dataset_info = {
        'train_samples_per_individual': {},
        'test_samples_per_individual': {},
        'train_ymdh_per_individual': {},
        'test_ymdh_per_individual': {},
    }

    for ind_id in individuals:
        # Get all indices for this individual
        all_ind_indices = set(id_to_indices.get(ind_id, []))

        # EXCLUDE hygiene validation indices
        hygiene_val_indices = set(config['validation_indices'].get(ind_id, {}).get('indices', []))
        available_indices = list(all_ind_indices - hygiene_val_indices)

        if len(available_indices) == 0:
            print(f"  WARNING: {ind_id} has NO samples after excluding hygiene val")
            continue

        # Get unique ymdh values for available indices
        available_ymdh = [ymdh_arr[i] for i in available_indices]
        unique_ymdh = sorted(set(available_ymdh))

        # Split ymdh values: earlier 50% -> train, later 50% -> test
        n_ymdh = len(unique_ymdh)
        split_point = n_ymdh // 2
        train_ymdh_set = set(unique_ymdh[:split_point])
        test_ymdh_set = set(unique_ymdh[split_point:])

        # Partition indices by ymdh
        train_pool = [i for i in available_indices if ymdh_arr[i] in train_ymdh_set]
        test_pool = [i for i in available_indices if ymdh_arr[i] in test_ymdh_set]

        # Train gets priority: sample TARGET_SAMPLES_PER_INDIVIDUAL
        random.shuffle(train_pool)
        train_sampled = train_pool[:TARGET_SAMPLES_PER_INDIVIDUAL]

        # Test gets remainder: sample TARGET_SAMPLES_PER_INDIVIDUAL
        random.shuffle(test_pool)
        test_sampled = test_pool[:TARGET_SAMPLES_PER_INDIVIDUAL]

        all_train_indices.extend(train_sampled)
        all_test_indices.extend(test_sampled)

        dataset_info['train_samples_per_individual'][ind_id] = len(train_sampled)
        dataset_info['test_samples_per_individual'][ind_id] = len(test_sampled)
        dataset_info['train_ymdh_per_individual'][ind_id] = len(train_ymdh_set)
        dataset_info['test_ymdh_per_individual'][ind_id] = len(test_ymdh_set)

        print(f"  {ind_id}: train={len(train_sampled)}/{len(train_pool)} (ymdh: {len(train_ymdh_set)}), "
              f"test={len(test_sampled)}/{len(test_pool)} (ymdh: {len(test_ymdh_set)})")

    # Create dataset subsets
    train_dataset = dataset.select(all_train_indices) if all_train_indices else None
    test_dataset = dataset.select(all_test_indices) if all_test_indices else None

    dataset_info['total_train'] = len(all_train_indices)
    dataset_info['total_test'] = len(all_test_indices)

    print(f"Total: {len(all_train_indices)} train, {len(all_test_indices)} test")

    return train_dataset, test_dataset, individual_to_class, dataset_info


def evaluate_recall_simple(model, train_dataset, test_dataset, individual_to_class, transform, device, batch_size=32):
    """
    Compute macro-averaged Recall@1 using cosine similarity.
    No quality filtering, no open-set evaluation.

    Returns: recall_at_1 value
    """
    model.eval()

    # Create datasets
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    test_torch = ArcFaceDataset(test_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    # Gallery embeddings (train)
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            emb = model(images)
            gallery_embeddings.append(emb)
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query embeddings (test)
    query_embeddings = []
    query_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            emb = model(images)
            query_embeddings.append(emb)
            query_labels.extend(labels.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels_np = np.array(query_labels)

    # Compute cosine similarity
    query_emb_norm = F.normalize(query_embeddings, p=2, dim=1)
    gallery_emb_norm = F.normalize(gallery_embeddings, p=2, dim=1)
    similarity = torch.mm(query_emb_norm, gallery_emb_norm.t())  # Higher = better

    # Macro-averaged Recall@1
    individual_correct = defaultdict(int)
    individual_total = defaultdict(int)

    for i in range(len(query_labels_np)):
        true_label = query_labels_np[i]
        pred_idx = similarity[i].argmax().item()
        pred_label = gallery_labels[pred_idx].item()

        individual_total[true_label] += 1
        if pred_label == true_label:
            individual_correct[true_label] += 1

    per_ind_recall = [
        individual_correct.get(label, 0) / individual_total[label]
        for label in individual_total
    ]

    recall_at_1 = np.mean(per_ind_recall) if per_ind_recall else 0.0

    return recall_at_1


def train_single_resize(resize_size: int, args, dataset, config, metadata_cache) -> dict:
    """Train one configuration and return results."""
    set_all_seeds(SEED)

    # Output path
    filename = f"resize={resize_size}.json"
    output_path = os.path.join(args.output_dir, filename)

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: resize_size={resize_size}")
    print(f"{'='*60}")

    # Get valid individuals compatible with gallery_size=64 at threshold=0.0
    valid_individuals = []
    for ind_id in config.get('valid_individuals', []):
        training_compat = config['training_compatibility'].get(ind_id, {})
        threshold_compat = training_compat.get('threshold_compatibility', {}).get('threshold_0.00', {})
        compatible_sizes = threshold_compat.get('compatible_training_sizes', [])
        if 64 in compatible_sizes:
            valid_individuals.append(ind_id)

    if len(valid_individuals) < MIN_P:
        print(f"Not enough individuals ({len(valid_individuals)}) for PK sampling (need {MIN_P})")
        return None

    print(f"Using {len(valid_individuals)} individuals: {', '.join(valid_individuals)}")

    # Create temporal split dataset
    train_dataset, test_dataset, individual_to_class, dataset_info = create_ymdh_split_dataset(
        dataset, valid_individuals, metadata_cache, config, seed=SEED
    )

    if train_dataset is None or len(train_dataset) == 0:
        print(f"No training data available")
        return None

    if test_dataset is None or len(test_dataset) == 0:
        print(f"No test data available")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, embedding_dim = create_megadescriptor_arcface_model(embedding_dim=EMBEDDING_DIM, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset - use the resize_size parameter
    transform = create_megadescriptor_transform(size=resize_size)
    train_torch_dataset = ArcFaceDataset(train_dataset, transform, individual_to_class)

    # Create PK batch sampler
    effective_k = min(BATCH_K, TARGET_SAMPLES_PER_INDIVIDUAL)
    try:
        pk_sampler = PKBatchSampler(
            labels=train_torch_dataset.get_labels(),
            p=min(MIN_P, len(valid_individuals)),
            k=effective_k,
            drop_last=True
        )
    except ValueError as e:
        print(f"Cannot create PK sampler: {e}")
        return None

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels

    train_loader = DataLoader(
        train_torch_dataset,
        batch_sampler=pk_sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create ArcFace loss
    criterion = ArcFaceLoss(
        num_classes=len(valid_individuals),
        embedding_size=embedding_dim,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    head_params = count_trainable_parameters(model.head)
    arcface_params = count_trainable_parameters(criterion)

    print(f"  Trainable params: head={head_params:,}, arcface={arcface_params:,}")

    # Optimizer with fixed LR
    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE
    )

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        # Evaluation - need to use same transform for consistency
        recall_at_1 = evaluate_recall_simple(
            model, train_dataset, test_dataset, individual_to_class, transform, device
        )

        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:3d}/{EPOCHS}: loss={train_loss:.4f}, test_R@1={recall_at_1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_recall_at_1': recall_at_1
        })

    training_time = time.time() - start_time

    # Prepare result
    result = {
        'config': {
            'resize_size': resize_size,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'seed': SEED,
            'backbone': 'Frozen MegaDescriptor-L-384',
            'target_samples_per_individual': TARGET_SAMPLES_PER_INDIVIDUAL
        },
        'dataset': {
            'individuals': valid_individuals,
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


def main():
    parser = argparse.ArgumentParser(description='Image Size Sweep for MegaDescriptor Re-ID with Temporal Split')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-16 for 17 sizes)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_MD/opt/resize_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Image Size Sweep - MegaDescriptor Re-ID with Temporal Split - Job {args.idx}")
    print("=" * 80)

    # Check if this job has work to do
    if args.idx >= len(RESIZE_SIZES):
        print(f"Job {args.idx} has no work (only {len(RESIZE_SIZES)} sizes)")
        return

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    config = load_feasibility_config()

    if not config:
        print("Failed to load feasibility config")
        return

    # Get size for this job
    resize_size = RESIZE_SIZES[args.idx]

    print(f"\nExperiment parameters:")
    print(f"  Resize size: {resize_size}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Seed: {SEED}")
    print(f"  Target samples per individual: {TARGET_SAMPLES_PER_INDIVIDUAL}")

    # Train
    try:
        result = train_single_resize(resize_size, args, dataset, config, metadata_cache)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Job {args.idx} completed!")


if __name__ == "__main__":
    main()

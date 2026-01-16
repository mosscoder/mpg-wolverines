#!/usr/bin/env python3
"""
Script 00: Quality-Weighted Classification Sweep
Classification experiment with cross-entropy loss weighted by anchor image quality.

Comparison experiment to reid_triplet to test if quality weighting helps
classification but not retrieval.

Grid: 4 alpha values × 6 sample sizes × 8 seeds = 192 configurations
Distributed across 24 SLURM jobs (8 configs/job).
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
from sklearn.metrics import f1_score, accuracy_score

# Control parallelism and set HuggingFace cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/data/hf_cache"

# Add project root to path
sys.path.append('.')

from utils.dataset import set_all_seeds
from utils.models import create_model
from utils.training import check_result_exists

import datasets
datasets.config.NUM_PROC = 1


# Experiment parameters
ALPHA_VALUES = [0, 10, 50, 100]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 16


class QualityWeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss weighted by sample quality score."""

    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, labels, quality_scores):
        # Per-sample cross-entropy (no reduction)
        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
        # Quality weighting: weight = 1 + alpha * quality
        weights = 1 + self.alpha * quality_scores
        return (weights * ce_loss).mean()


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
        print(f"Error: {config_path} not found.")
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
    Create temporal train/val split for classification.

    Training: N samples per individual from all years except final
    Validation: All samples from final year
    """
    import random

    set_all_seeds(seed)

    print(f"Creating temporal dataset: {sample_size} samples/class, seed={seed}")

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


class ClassificationDataset(TorchDataset):
    """PyTorch dataset for classification with quality scores."""

    def __init__(self, hf_dataset, transform, individual_to_class):
        self.dataset = hf_dataset
        self.transform = transform
        self.individual_to_class = individual_to_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        label = self.individual_to_class[sample['id']]
        quality = sample['pelage_score']
        return image, label, quality


def create_dinov3_transform():
    """Create DINOv3-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with quality-weighted cross-entropy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, quality_scores in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        quality_scores = quality_scores.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels, quality_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device):
    """Evaluate model on validation set with per-bin metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_quality = []

    with torch.no_grad():
        for images, labels, quality_scores in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_quality.extend(quality_scores.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_quality = np.array(all_quality)

    # Overall metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-bin metrics
    bins = [
        ('[0.75,1.0]', 0.75, 1.01),
        ('[0.5,0.75)', 0.5, 0.75),
        ('[0.25,0.5)', 0.25, 0.5),
        ('[0,0.25)', 0.0, 0.25)
    ]

    bin_metrics = {}
    for bin_name, low, high in bins:
        mask = (all_quality >= low) & (all_quality < high)
        count = mask.sum()
        if count > 0:
            bin_preds = all_preds[mask]
            bin_labels = all_labels[mask]
            bin_f1 = f1_score(bin_labels, bin_preds, average='macro')
            bin_acc = accuracy_score(bin_labels, bin_preds)
            bin_metrics[bin_name] = {
                'f1_macro': float(bin_f1),
                'accuracy': float(bin_acc),
                'count': int(count)
            }
        else:
            bin_metrics[bin_name] = {'f1_macro': 0.0, 'accuracy': 0.0, 'count': 0}

    return {
        'f1_macro': float(f1_macro),
        'accuracy': float(accuracy),
        'by_quality_bin': bin_metrics
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

    if len(feasible_individuals) < 2:
        print(f"Not enough individuals ({len(feasible_individuals)}) for classification")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create temporal dataset
    train_dataset, val_dataset, individual_to_class, dataset_info = create_temporal_dataset(
        dataset, feasible_individuals, sample_size, seed, config, id_to_indices
    )

    num_classes = len(feasible_individuals)

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model = create_model(num_classes=num_classes, device=device)
    print(f"Using device: {device}")

    # Create transforms and datasets
    transform = create_dinov3_transform()
    train_torch_dataset = ClassificationDataset(train_dataset, transform, individual_to_class)
    val_torch_dataset = ClassificationDataset(val_dataset, transform, individual_to_class)

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        quality = torch.tensor([item[2] for item in batch])
        return images, labels, quality

    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create loss and optimizer
    criterion = QualityWeightedCrossEntropyLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate every epoch
        metrics = evaluate(model, val_loader, device)
        val_f1 = metrics['f1_macro']

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val F1={val_f1:.4f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_f1_macro': val_f1,
            'val_accuracy': metrics['accuracy'],
            'learning_rate': LEARNING_RATE
        })

    training_time = time.time() - start_time

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, val_loader, device)

    # Prepare result
    result = {
        'config': {
            'alpha': alpha,
            'samples_per_class': sample_size,
            'seed': seed,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE
        },
        'dataset': {
            'individuals': feasible_individuals,
            'train_samples_per_individual': {k: v['train_samples'] for k, v in dataset_info.items()},
            'val_samples_per_individual': {k: v['val_samples'] for k, v in dataset_info.items()},
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_classes': num_classes
        },
        'final_metrics': {
            'f1_macro': final_metrics['f1_macro'],
            'accuracy': final_metrics['accuracy'],
            'by_quality_bin': final_metrics['by_quality_bin']
        },
        'epoch_history': epoch_history,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'job_idx': args.idx,
            'transform': 'resize_224_imagenet_norm',
            'best_epoch': best_epoch,
            'best_f1_macro': best_f1
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Final F1={final_metrics['f1_macro']:.4f}, Acc={final_metrics['accuracy']:.4f}")
    print(f"By quality bin:")
    for bin_name, bin_data in final_metrics['by_quality_bin'].items():
        if bin_data['count'] > 0:
            print(f"  {bin_name}: F1={bin_data['f1_macro']:.4f} (n={bin_data['count']})")

    return filename


def main():
    parser = argparse.ArgumentParser(description='Quality-weighted classification sweep')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_classifier/results/classifier_sweep',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Quality-Weighted Classification Experiment - Job {args.idx}")
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

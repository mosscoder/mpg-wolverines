#!/usr/bin/env python3
"""
Script 00: Open-Set Gallery Hygiene Sweep - MegaDescriptor + ArcFace + Raw Cosine Similarity

Uses frozen MegaDescriptor-L-384 backbone with raw cosine similarity for score calibration:
1. Freezes MegaDescriptor backbone (only trainable projection head)
2. Computes L2-normalized cosine similarity scores
3. Calibrates threshold on raw cosine scores using LOO within gallery
4. Measures Balanced Accuracy = (Known Accept Rate + Unknown Reject Rate) / 2

Key differences from reid_openset_tnorm:
- Score normalization: Raw cosine similarity [0, 1] instead of T-Norm Z-scores
- Threshold scale: Cosine (e.g., 0.3, 0.4) vs Z-scores (e.g., 2.5, 3.8)
- Simpler computation, no imposter distribution statistics needed

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

from utils.triplet import (
    ArcFaceLoss,
    PKBatchSampler,
    evaluate_open_set_balanced,
    create_megadescriptor_arcface_model,
)
from utils.training import check_result_exists
from utils.dataset import set_all_seeds

import datasets
datasets.config.NUM_PROC = 1


def build_metadata_cache(dataset):
    """Build metadata cache using Arrow columnar access (optimized)."""
    from utils.arrow_cache import build_metadata_cache_arrow
    return build_metadata_cache_arrow(dataset)


# Experiment parameters
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0.0 = no filtering (baseline)
GALLERY_SIZES = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

# ArcFace hyperparameters
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64
LEARNING_RATE = 0.0005
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




def filter_training_pool_by_quality(metadata_cache, indices, threshold):
    """Use vectorized filtering with cached quality scores."""
    from utils.optimized_filters import filter_training_pool_by_quality_vectorized
    return filter_training_pool_by_quality_vectorized(
        metadata_cache['quality_scores'],
        indices,
        threshold
    )


def get_rare_individual_indices(metadata_cache, valid_individuals: list, quality_threshold: float = 0.0,
                                promoted_individuals: list = None, excluded_individuals: list = None):
    """Use vectorized operations with metadata cache."""
    from utils.optimized_filters import get_rare_individual_indices_vectorized
    return get_rare_individual_indices_vectorized(
        metadata_cache,
        valid_individuals,
        quality_threshold,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
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

        # Get all indices for this individual (from cache)
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


def create_megadescriptor_transform():
    """Create MegaDescriptor-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(384, 384), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_arcface(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with ArcFace loss."""
    model.train()  # Projection head is trainable
    criterion.train()  # ArcFace weights are trainable
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Get embeddings (trainable head)
        embeddings = model(images)

        # Compute ArcFace loss
        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


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


def compute_validation_loss(val_embeddings, val_labels, criterion, device):
    """Compute ArcFace loss on validation set."""
    criterion.eval()
    val_embeddings = val_embeddings.to(device)
    val_labels = val_labels.to(device)

    with torch.no_grad():
        loss = criterion(val_embeddings, val_labels)

    return loss.item()


def compute_cosine_similarity(query_emb, gallery_emb):
    """
    Compute L2-normalized cosine similarity matrix.

    Args:
        query_emb: Query embeddings (N_q, D)
        gallery_emb: Gallery embeddings (N_g, D)

    Returns:
        Similarity matrix (N_q, N_g) with values in [0, 1] after normalization
    """
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    return torch.mm(query_emb, gallery_emb.t())


def compute_open_set_metrics_cosine(
    known_query_labels: torch.Tensor,
    known_query_quality: np.ndarray,
    known_scores: torch.Tensor,
    unknown_query_labels: np.ndarray,
    unknown_query_quality: np.ndarray,
    unknown_scores: torch.Tensor,
    score_threshold: float,
    quality_thresholds: list,
    gallery_labels: torch.Tensor
):
    """
    Compute macro-averaged open-set metrics for raw cosine similarity scores.

    Uses per-individual averaging for both known accept rate and unknown reject rate.
    This ensures equal weight to each individual regardless of sample count.

    Args:
        known_query_labels: Labels for known queries (n_known,)
        known_query_quality: Quality scores for known queries (n_known,)
        known_scores: Cosine similarity matrix for known queries (n_known, n_gallery)
        unknown_query_labels: Individual IDs for unknown queries (n_unknown,)
        unknown_query_quality: Quality scores for unknown queries (n_unknown,)
        unknown_scores: Cosine similarity matrix for unknown queries (n_unknown, n_gallery)
        score_threshold: Accept threshold (accept if score >= threshold)
        quality_thresholds: List of quality thresholds to evaluate
        gallery_labels: Labels for gallery samples (n_gallery,)

    Returns:
        Dict mapping quality threshold to metrics
    """
    # Convert labels to numpy for grouping
    if isinstance(known_query_labels, torch.Tensor):
        known_labels_np = known_query_labels.cpu().numpy()
    else:
        known_labels_np = np.array(known_query_labels)

    results = {}

    for q_thresh in quality_thresholds:
        # --- Known accept rate (macro-averaged) ---
        k_mask = known_query_quality >= q_thresh
        per_ind_accept = []
        n_known_individuals = 0

        if k_mask.sum() > 0:
            known_indices = np.where(k_mask)[0]
            unique_known_labels = np.unique(known_labels_np[known_indices])

            for ind_label in unique_known_labels:
                # Get indices for this individual within quality-filtered set
                ind_mask = (known_labels_np == ind_label) & k_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # Detection only: accept if score >= threshold
                # Correct identification is measured separately by Recall@1
                accepted = 0
                for i in ind_indices:
                    max_score = known_scores[i].max().item()
                    if max_score >= score_threshold:  # Detection only
                        accepted += 1

                per_ind_accept.append(accepted / len(ind_indices))  # Per-individual rate

            n_known_individuals = len(per_ind_accept)

        known_accept_rate = np.mean(per_ind_accept) if per_ind_accept else 0.0

        # --- Unknown reject rate (macro-averaged) ---
        u_mask = unknown_query_quality >= q_thresh
        per_ind_reject = []
        n_unknown_individuals = 0

        if len(unknown_query_labels) > 0 and u_mask.sum() > 0:
            unknown_indices = np.where(u_mask)[0]
            unique_unknown_labels = np.unique(unknown_query_labels[unknown_indices])

            for ind_id in unique_unknown_labels:
                # Get indices for this individual within quality-filtered set
                ind_mask = (unknown_query_labels == ind_id) & u_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # For each sample: reject if max score < threshold
                rejected = 0
                for i in ind_indices:
                    max_score = unknown_scores[i].max().item()
                    if max_score < score_threshold:
                        rejected += 1

                per_ind_reject.append(rejected / len(ind_indices))

            n_unknown_individuals = len(per_ind_reject)

        unknown_reject_rate = np.mean(per_ind_reject) if per_ind_reject else 1.0

        # Balanced accuracy
        balanced_acc = (known_accept_rate + unknown_reject_rate) / 2.0

        results[f"q>={q_thresh}"] = {
            'balanced_accuracy': balanced_acc,
            'known_accept_rate': known_accept_rate,
            'unknown_reject_rate': unknown_reject_rate,
            'n_known_individuals': n_known_individuals,
            'n_unknown_individuals': n_unknown_individuals
        }

    return results


def evaluate_recall_with_openset(model, train_dataset, val_dataset, individual_to_class,
                                  transform, device, dataset, valid_individuals, metadata_cache,
                                  gallery_threshold, criterion, batch_size=32,
                                  promoted_individuals=None, excluded_individuals=None):
    """
    Evaluate model computing Recall@1 and open-set metrics with raw cosine similarity.

    Uses L2-normalized cosine similarity for both closed-set and open-set evaluation.

    Returns:
        dict: query_quality_metrics with recall at each quality threshold
              open_set metrics with balanced accuracy at each quality threshold
              val_loss: validation loss value
    """
    model.eval()

    # --- 1. Extract Embeddings ---
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    val_torch = ArcFaceDataset(val_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    # Gallery
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad():
        for images, labels, _ in train_loader:
            images = images.to(device)
            gallery_embeddings.append(model(images))
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query (Known)
    query_embeddings = []
    query_labels = []
    query_quality = []
    with torch.no_grad():
        for images, labels, quality in val_loader:
            images = images.to(device)
            query_embeddings.append(model(images))
            query_labels.extend(labels.tolist())
            query_quality.extend(quality.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels = torch.tensor(query_labels).to(device)
    query_quality = np.array(query_quality)

    # Compute validation loss using ArcFace criterion
    val_loss = compute_validation_loss(query_embeddings, query_labels, criterion, device)

    # Rare/Unknown (VECTORIZED) - includes promoted individuals, excludes excluded individuals
    rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
        metadata_cache, valid_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
    )
    if rare_indices:
        rare_emb, rare_quality_arr, rare_labels_arr = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels_str, transform, device
        )
        rare_emb = rare_emb.to(device)
    else:
        rare_emb = torch.empty(0, EMBEDDING_DIM).to(device)
        rare_quality_arr = np.array([])
        rare_labels_arr = np.array([])

    # --- 2. Compute Cosine Similarity Matrices ---
    # Known Query vs Gallery
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    # Unknown Query vs Gallery
    if len(rare_emb) > 0:
        scores_unknown = compute_cosine_similarity(rare_emb, gallery_embeddings)
    else:
        scores_unknown = torch.empty(0, len(gallery_embeddings)).to(device)

    # Gallery vs Gallery (for threshold calibration)
    scores_gal_gal = compute_cosine_similarity(gallery_embeddings, gallery_embeddings)

    # --- 3. Compute Metrics ---

    # A. Recall@1 (Closed Set) - macro-averaged, using cosine similarity scores
    query_quality_metrics = {}
    query_labels_np = query_labels.cpu().numpy()

    for thresh in QUERY_QUALITY_THRESHOLDS:
        mask = query_quality >= thresh
        count = int(mask.sum())
        if count == 0:
            query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": 0.0, "count": 0}
            continue

        # Compute macro-averaged Recall@1 using cosine similarity scores
        # Group by individual, compute per-individual accuracy, then average
        mask_indices = np.where(mask)[0]
        individual_correct = {}
        individual_total = {}

        for idx in mask_indices:
            true_label = query_labels_np[idx]
            # Find argmax of cosine scores (highest = best match)
            pred_idx = scores_known[idx].argmax().item()
            pred_label = gallery_labels[pred_idx].item()

            individual_total[true_label] = individual_total.get(true_label, 0) + 1
            if pred_label == true_label:
                individual_correct[true_label] = individual_correct.get(true_label, 0) + 1

        # Per-individual Recall@1, then mean (macro-averaging)
        per_ind_recall = [
            individual_correct.get(label, 0) / individual_total[label]
            for label in individual_total
        ]
        recall = np.mean(per_ind_recall) if per_ind_recall else 0.0

        query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    # B. Threshold Calibration (LOO on Cosine Similarity Gallery Scores)
    # Mask out self-matches (diagonal)
    N = len(gallery_labels)
    scores_gal_gal.fill_diagonal_(-1.0)  # Use -1 since cosine is in [0, 1]

    # Individual-level LOO Threshold Calibration
    #
    # For each held-out individual H:
    #   1. Gallery = all samples from other individuals
    #   2. Unknown queries = H's samples (truly unknown - no same-ID in gallery)
    #   3. Known queries = sample-level LOO within reduced gallery
    #   4. Grid search for optimal threshold for this fold
    #
    # Final threshold = mean across all folds

    unique_individuals = gallery_labels.unique().tolist()
    fold_results = []
    thresholds = np.linspace(0.0, 1.0, 101)  # Cosine similarity range [0, 1]

    for held_out_ind in unique_individuals:
        # Masks for this fold
        gallery_mask = (gallery_labels != held_out_ind)  # Other individuals
        unknown_mask = (gallery_labels == held_out_ind)  # Held-out individual

        # UNKNOWN: held-out individual queries the reduced gallery
        # These are truly unknown - no same-ID exists in gallery
        unknown_max_scores = scores_gal_gal[unknown_mask][:, gallery_mask].max(dim=1).values

        # KNOWN: sample-level LOO within the reduced gallery
        # Group by individual for macro-averaging
        known_results_by_ind = {}  # {ind_label: [(max_score, is_correct), ...]}
        gallery_indices = torch.where(gallery_mask)[0]

        for idx in gallery_indices:
            label_i = gallery_labels[idx].item()
            row = scores_gal_gal[idx]

            # Valid targets: in reduced gallery AND not self
            valid_mask = gallery_mask.clone()
            valid_mask[idx] = False

            if valid_mask.sum() > 0:
                scores_to_valid = row[valid_mask]
                max_score, local_idx = scores_to_valid.max(dim=0)

                # Map local index back to original index
                valid_indices = torch.where(valid_mask)[0]
                pred_idx = valid_indices[local_idx]
                pred_label = gallery_labels[pred_idx].item()
                is_correct = (pred_label == label_i)
                known_results_by_ind.setdefault(label_i, []).append((max_score.item(), is_correct))

        # Grid search for this fold
        fold_best_ba, fold_best_thresh = 0.0, 0.5
        fold_best_known_accept, fold_best_unknown_reject = 0.0, 0.0

        for thresh in thresholds:
            # Known accept: macro-averaged across individuals in reduced gallery
            # Detection only - correct identification measured separately by Recall@1
            per_ind_accept = []
            for ind_label, results in known_results_by_ind.items():
                accepted = sum(1 for s, c in results if s >= thresh)  # Detection only
                per_ind_accept.append(accepted / len(results))  # Per-individual rate
            known_accept = np.mean(per_ind_accept) if per_ind_accept else 0.0  # Macro-average

            # Unknown reject: all samples from held-out individual (single group)
            if len(unknown_max_scores) > 0:
                unknown_reject = (unknown_max_scores < thresh).float().mean().item()
            else:
                unknown_reject = 0.0

            ba = (known_accept + unknown_reject) / 2
            if ba > fold_best_ba:
                fold_best_ba, fold_best_thresh = ba, thresh
                fold_best_known_accept = known_accept
                fold_best_unknown_reject = unknown_reject

        fold_results.append({
            'individual': held_out_ind,
            'threshold': fold_best_thresh,
            'ba': fold_best_ba,
            'known_accept': fold_best_known_accept,
            'unknown_reject': fold_best_unknown_reject,
        })

    # Final threshold = mean across folds
    optimal_thresh = np.mean([f['threshold'] for f in fold_results])
    calibration_ba = np.mean([f['ba'] for f in fold_results])
    calibration_known_accept = np.mean([f['known_accept'] for f in fold_results])
    calibration_unknown_reject = np.mean([f['unknown_reject'] for f in fold_results])
    n_folds = len(fold_results)

    # C. Open Set Metrics (Balanced Accuracy) - MACRO-AVERAGED
    balanced_metrics_by_quality = compute_open_set_metrics_cosine(
        known_query_labels=query_labels,
        known_query_quality=query_quality,
        known_scores=scores_known,
        unknown_query_labels=rare_labels_arr,
        unknown_query_quality=rare_quality_arr,
        unknown_scores=scores_unknown,
        score_threshold=optimal_thresh,
        quality_thresholds=QUERY_QUALITY_THRESHOLDS,
        gallery_labels=gallery_labels
    )

    open_set_metrics = {
        'threshold_calibration': {
            'method': 'individual_loo_ba',
            'threshold': optimal_thresh,
            'threshold_std': np.std([f['threshold'] for f in fold_results]),
            'calibration_ba': calibration_ba,
            'calibration_known_accept': calibration_known_accept,
            'calibration_unknown_reject': calibration_unknown_reject,
            'n_folds': n_folds,
            'per_fold': fold_results,
        },
        'by_quality': balanced_metrics_by_quality
    }

    return query_quality_metrics, open_set_metrics, val_loss


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

    # Get valid individuals and diversity-based classification from config
    valid_individuals = config.get('valid_individuals', [])
    promoted_individuals = config.get('promoted_to_rare', [])
    excluded_individuals = config.get('excluded_entirely', [])

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

    # Create model with frozen backbone
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, embedding_dim = create_megadescriptor_arcface_model(embedding_dim=EMBEDDING_DIM, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_megadescriptor_transform()
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

    # Train projection head and ArcFace class centers with AdamW
    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE
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

    for epoch in range(EPOCHS):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        # Evaluation - returns metrics for closed-set and open-set
        query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
            model, train_dataset, val_dataset, individual_to_class, transform, device,
            dataset, feasible_individuals, metadata_cache, gallery_threshold=threshold,
            criterion=criterion,
            promoted_individuals=promoted_individuals,
            excluded_individuals=excluded_individuals
        )

        # Overall recall is at q>=0.0 (includes all queries)
        recall_1 = query_quality_metrics["q>=0.0"]["recall_at_1"]

        if recall_1 > best_recall:
            best_recall = recall_1
            best_recall_epoch = epoch + 1

        # Track best balanced accuracy for open-set (at q>=0.0 for overall metric)
        by_quality = open_set_metrics.get('by_quality', {})
        ba_q0 = by_quality.get('q>=0.0', {}).get('balanced_accuracy', 0.0)
        if ba_q0 > best_balanced_accuracy:
            best_balanced_accuracy = ba_q0
            best_ba_epoch = epoch + 1

        # Track best harmonic mean of R@1 and BA (balances closed-set and open-set)
        if (recall_1 + ba_q0) > 0:
            hm = 2 * recall_1 * ba_q0 / (recall_1 + ba_q0)
        else:
            hm = 0.0
        if hm > best_harmonic_mean:
            best_harmonic_mean = hm
            best_hm_epoch = epoch + 1

        thresh_mean = open_set_metrics.get('threshold_calibration', {}).get('threshold', 0.0)

        # Header with loss and threshold info
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, thresh={thresh_mean:.3f}")

        # Per-quality metrics
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
            'optimizer': 'AdamW',
            'scheduler': 'None (fixed LR)',
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size_k': BATCH_K,
            'backbone': 'Frozen MegaDescriptor-L-384',
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
            'transform': 'resize_384_megadescriptor_norm',
            # Individual metric bests (for reference)
            'best_recall_epoch': best_recall_epoch,
            'best_recall_at_1': best_recall,
            'best_ba_epoch': best_ba_epoch,
            'best_balanced_accuracy': best_balanced_accuracy,
            # Harmonic mean best (recommended for epoch selection)
            'best_hm_epoch': best_hm_epoch,
            'best_harmonic_mean': best_harmonic_mean,
            # Note: Final epoch selection should be done at analysis time
            # using 01_plot_hygiene_sweep.py which supports multiple criteria
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {filename}")
    print(f"Best R@1 epoch: {best_recall_epoch}, R@1={best_recall:.4f}")
    print(f"Best BA epoch:  {best_ba_epoch}, BA={best_balanced_accuracy:.4f}")
    print(f"Best H-Mean epoch: {best_hm_epoch}, H-Mean={best_harmonic_mean:.4f} (recommended)")

    return filename


def main():
    parser = argparse.ArgumentParser(description='Open-Set MegaDescriptor + ArcFace + Raw Cosine Gallery Hygiene Sweep')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_MD/results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Open-Set MegaDescriptor + ArcFace + Raw Cosine Gallery Hygiene Sweep - Job {args.idx}")
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
    print(f"  Model: Frozen MegaDescriptor-L-384 + Projection Head ({EMBEDDING_DIM}-d) + ArcFace")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    print(f"  Score Normalization: Raw Cosine Similarity (L2-normalized)")
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, default settings)")
    print(f"  Embedding: {EMBEDDING_DIM}-d (trainable projection from MegaDescriptor)")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")
    print(f"  Open-set evaluation: Enabled (rare individuals, raw cosine scores)")

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

#!/usr/bin/env python3
"""
Script 00: Open-Set Gallery Hygiene Sweep - DINOv3 + ArcFace + T-Norm

Uses frozen DINOv3 backbone with T-Norm (Test Normalization) for score calibration:
1. Freezes DINOv3 backbone (only trainable projection head)
2. Computes T-Norm statistics (mean, std) from imposter distributions in gallery
3. Normalizes similarity scores to Z-scores for better open-set discrimination
4. Calibrates threshold on normalized scores using LOO within gallery
5. Measures Balanced Accuracy = (Known Accept Rate + Unknown Reject Rate) / 2

Key differences from reid_openset_arcface:
- Score normalization: T-Norm Z-scores instead of raw cosine similarity
- Threshold scale: Z-scores (e.g., 2.5, 3.8) vs cosine (e.g., 0.3, 0.4)
- Better handling of low-quality gallery templates

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

from utils.triplet import (
    ArcFaceLoss,
    PKBatchSampler,
    compute_recall_at_k,
    evaluate_open_set_balanced,
)
from utils.training import check_result_exists
from utils.dataset import set_all_seeds

import datasets
datasets.config.NUM_PROC = 1


def build_metadata_cache(dataset):
    """Build metadata cache using Arrow columnar access (optimized)."""
    from utils.arrow_cache import build_metadata_cache_arrow
    return build_metadata_cache_arrow(dataset)


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


def create_dinov3_transform():
    """Create DINOv3-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dinov3_arcface_model(embedding_dim: int = 128, device="cuda"):
    """
    Create DINOv3 backbone with trainable projection head.

    Architecture: Frozen DINOv3 -> 768-d CLS -> EmbeddingHead (768->128) -> ArcFace

    The trainable projection head allows embeddings to improve during training,
    while keeping the backbone frozen for efficiency.

    Returns:
        Tuple of (model, embedding_dim)
    """
    backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Trainable projection head (768 -> 128)
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


def compute_tnorm_stats(gallery_emb, gallery_labels):
    """
    Compute T-Norm stats (mean, std) for each gallery sample using a
    'Leave-Other-Classes-Out' cohort strategy.

    Args:
        gallery_emb: Normalized embeddings (N, D)
        gallery_labels: Labels (N,)

    Returns:
        mu (N,), sigma (N,) tensors on the same device
    """
    # Ensure normalized for Cosine Similarity
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)

    # Compute full Gallery-to-Gallery Similarity Matrix
    # S[i, j] is sim between gallery item i and j
    sim_matrix = torch.mm(gallery_emb, gallery_emb.t())

    N = len(gallery_labels)
    mu = torch.zeros(N, device=gallery_emb.device)
    sigma = torch.ones(N, device=gallery_emb.device) # default to 1.0

    # We can vectorize this, but a loop is safer for variable class counts/logic
    # and fast enough for N < 1000
    unique_labels = torch.unique(gallery_labels)

    for i in range(N):
        label_i = gallery_labels[i]

        # Cohort = All samples NOT belonging to label_i
        # This is the "Imposter Distribution" for gallery item i
        cohort_mask = (gallery_labels != label_i)

        if cohort_mask.sum() > 0:
            cohort_scores = sim_matrix[i, cohort_mask]
            mu[i] = cohort_scores.mean()
            sigma[i] = cohort_scores.std()

            # Clamp sigma to avoid exploding scores on degenerate cohorts
            if sigma[i] < 1e-6:
                sigma[i] = 1e-6

    return mu, sigma


def apply_tnorm(query_emb, gallery_emb, mu, sigma):
    """
    Compute Normalized Similarity Matrix.
    S_norm(q, g) = ( Sim(q, g) - mu[g] ) / sigma[g]
    """
    # 1. Raw Cosine Similarity
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)

    raw_sim = torch.mm(query_emb, gallery_emb.t()) # (Num_Queries, Num_Gallery)

    # 2. Apply T-Norm
    # mu/sigma are shape (Num_Gallery,), we broadcast over queries (rows)
    norm_sim = (raw_sim - mu.unsqueeze(0)) / sigma.unsqueeze(0)

    return norm_sim


def compute_open_set_metrics_tnorm(
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
    Compute macro-averaged open-set metrics for T-normed scores.

    Uses per-individual averaging for both known accept rate and unknown reject rate.
    This matches the macro-averaging approach used in arcface/lora.

    Args:
        known_query_labels: Labels for known queries (n_known,)
        known_query_quality: Quality scores for known queries (n_known,)
        known_scores: T-normed score matrix for known queries (n_known, n_gallery)
        unknown_query_labels: Individual IDs for unknown queries (n_unknown,)
        unknown_query_quality: Quality scores for unknown queries (n_unknown,)
        unknown_scores: T-normed score matrix for unknown queries (n_unknown, n_gallery)
        score_threshold: Accept threshold (accept if score > threshold)
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

                # For each sample: accept if score > threshold AND correct match
                accepted_correct = 0
                for i in ind_indices:
                    max_score, max_idx = known_scores[i].max(dim=0)
                    pred_label = gallery_labels[max_idx].item()
                    if max_score.item() > score_threshold and pred_label == ind_label:
                        accepted_correct += 1

                per_ind_accept.append(accepted_correct / len(ind_indices))

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
    Evaluate model computing Recall@1 and open-set metrics with T-Norm.

    T-Norm normalizes similarity scores to Z-scores using imposter distributions,
    improving discrimination for open-set recognition.

    Returns:
        dict: query_quality_metrics with recall at each quality threshold
              open_set metrics with balanced accuracy at each quality threshold
              val_loss: validation loss value
    """
    model.eval()

    # --- 1. Extract Embeddings (Same as before) ---
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

    # --- 2. T-NORM CALCULATION ---
    # Calculate stats on the Gallery (Offline step)
    tnorm_mu, tnorm_sigma = compute_tnorm_stats(gallery_embeddings, gallery_labels)

    # Calculate Normalized Score Matrices
    # Known Query vs Gallery
    scores_known = apply_tnorm(query_embeddings, gallery_embeddings, tnorm_mu, tnorm_sigma)

    # Unknown Query vs Gallery
    if len(rare_emb) > 0:
        scores_unknown = apply_tnorm(rare_emb, gallery_embeddings, tnorm_mu, tnorm_sigma)
    else:
        scores_unknown = torch.empty(0, len(gallery_embeddings)).to(device)


    # --- 3. Compute Metrics ---

    # A. Recall@1 (Closed Set) - macro-averaged, same as arcface/lora
    # Uses compute_recall_at_k which normalizes embeddings and uses cosine-based ranking
    query_quality_metrics = {}
    for thresh in QUERY_QUALITY_THRESHOLDS:
        mask = query_quality >= thresh
        count = int(mask.sum())
        if count == 0:
            query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": 0.0, "count": 0}
            continue

        # Use macro-averaged recall (same as arcface/lora)
        recall = compute_recall_at_k(
            query_embeddings[mask],
            gallery_embeddings,
            query_labels[torch.tensor(mask)],
            gallery_labels,
            k=1
        )
        query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    # B. Threshold Calibration (LOO on T-Normed Gallery Scores)
    # We must calculate Gal-vs-Gal scores using T-Norm to be consistent
    scores_gal_gal = apply_tnorm(gallery_embeddings, gallery_embeddings, tnorm_mu, tnorm_sigma)

    # Mask out self-matches (diagonal)
    N = len(gallery_labels)
    scores_gal_gal.fill_diagonal_(-9999)

    # LOO Threshold Calibration via Max-Score Simulation
    #
    # For each gallery sample, simulate:
    # 1. KNOWN query: max score to gallery (excluding self) - check if correct match
    # 2. UNKNOWN query: max score to different-ID samples only (imposter simulation)
    #
    # Then grid search for threshold that maximizes balanced accuracy
    # where unknown detection is framed as the positive class.

    known_individual_results = {}   # {ind_label: [(max_score, is_correct_match), ...]}
    unknown_simulation_results = {} # {ind_label: [max_imposter_score, ...]}

    for i in range(N):
        label_i = gallery_labels[i].item()
        row = scores_gal_gal[i]  # Already has diagonal=-9999

        # --- KNOWN simulation: max score to gallery (excluding self) ---
        max_score, max_idx = row.max(dim=0)
        pred_label = gallery_labels[max_idx].item()
        is_correct = (pred_label == label_i)
        known_individual_results.setdefault(label_i, []).append((max_score.item(), is_correct))

        # --- UNKNOWN simulation: max score to different-ID only ---
        imposter_mask = (gallery_labels != gallery_labels[i])
        if imposter_mask.sum() > 0:
            max_imposter = row[imposter_mask].max().item()
            unknown_simulation_results.setdefault(label_i, []).append(max_imposter)

    # Grid search for optimal threshold
    thresholds = np.linspace(0, 10, 101)
    best_ba, best_thresh = 0.0, 4.0
    best_known_accept, best_unknown_reject = 0.0, 0.0

    for thresh in thresholds:
        # Macro-averaged known accept rate (TNR in unknown-positive framing)
        # Accept = max_score > thresh AND correct match
        per_ind_accept = []
        for ind_label, results in known_individual_results.items():
            accepted = sum(1 for max_s, correct in results if max_s > thresh and correct)
            per_ind_accept.append(accepted / len(results))
        known_accept_rate = np.mean(per_ind_accept) if per_ind_accept else 0.0

        # Macro-averaged unknown reject rate (TPR in unknown-positive framing)
        # Reject = max_imposter_score < thresh
        per_ind_reject = []
        for ind_label, max_imposters in unknown_simulation_results.items():
            rejected = sum(1 for max_s in max_imposters if max_s < thresh)
            per_ind_reject.append(rejected / len(max_imposters))
        unknown_reject_rate = np.mean(per_ind_reject) if per_ind_reject else 0.0

        ba = (known_accept_rate + unknown_reject_rate) / 2
        if ba > best_ba:
            best_ba, best_thresh = ba, thresh
            best_known_accept = known_accept_rate
            best_unknown_reject = unknown_reject_rate

    optimal_thresh = best_thresh
    calibration_ba = best_ba
    calibration_known_accept = best_known_accept
    calibration_unknown_reject = best_unknown_reject
    n_known_individuals = len(known_individual_results)

    # C. Open Set Metrics (Balanced Accuracy) - MACRO-AVERAGED
    # Uses per-individual averaging for consistency with arcface/lora
    balanced_metrics_by_quality = compute_open_set_metrics_tnorm(
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
            'method': 'max_score_loo',
            'threshold': optimal_thresh,
            'calibration_ba': calibration_ba,
            'calibration_known_accept': calibration_known_accept,
            'calibration_unknown_reject': calibration_unknown_reject,
            'n_known_individuals': n_known_individuals,
            'n_thresholds_searched': len(thresholds),
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
    best_epoch = 0
    best_balanced_accuracy = 0.0
    best_ba_epoch = 0

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
            best_epoch = epoch + 1

        # Track best balanced accuracy for open-set (at q>=0.0 for overall metric)
        by_quality = open_set_metrics.get('by_quality', {})
        ba_q0 = by_quality.get('q>=0.0', {}).get('balanced_accuracy', 0.0)
        if ba_q0 > best_balanced_accuracy:
            best_balanced_accuracy = ba_q0
            best_ba_epoch = epoch + 1

        thresh_mean = open_set_metrics.get('threshold_calibration', {}).get('threshold_mean', 0.0)

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
            'backbone': 'Frozen DINOv3-ViT-B/16',
            'score_normalization': 'T-Norm'
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
    parser = argparse.ArgumentParser(description='Open-Set DINOv3 + ArcFace + T-Norm Gallery Hygiene Sweep')
    parser.add_argument('--idx', type=int, required=True, help='Job index (0-23)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--output_dir', type=str,
                        default='reid_openset_tnorm/results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'],
                        default='gpu', help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Open-Set DINOv3 + ArcFace + T-Norm Gallery Hygiene Sweep - Job {args.idx}")
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
    print(f"  Model: Frozen DINOv3 + Projection Head ({EMBEDDING_DIM}-d) + ArcFace")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    print(f"  Score Normalization: T-Norm (imposter cohort statistics)")
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, default settings)")
    print(f"  Embedding: {EMBEDDING_DIM}-d (trainable projection from DINOv3 CLS)")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Gallery sizes: {GALLERY_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Configs per job: ~{total_combinations // 24}")
    print(f"  Open-set evaluation: Enabled (rare individuals, T-Norm scores)")

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

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
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image

from utils.dataset import set_all_seeds

import datasets
datasets.config.NUM_PROC = 1


# ============================================================================
# Model configuration registry
# ============================================================================

MODEL_CONFIGS = {
    "dinov3": {
        "factory": "create_dinov3_arcface_model",
        "native_size": 256,
        "patch_size": 16,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "backbone_label": "Frozen DINOv3-ViT-B/16",
        "experiment_dir": "reid_openset/dinov3",
        "default_lr": 0.0005,
    },
    "megadescriptor": {
        "factory": "create_megadescriptor_arcface_model",
        "native_size": 384,
        "patch_size": 16,
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
        "backbone_label": "Frozen MegaDescriptor-L-384",
        "experiment_dir": "reid_openset/megadescriptor",
        "default_lr": 0.0005,
    },
    "bioclip2": {
        "factory": "create_bioclip2_arcface_model",
        "native_size": 224,
        "patch_size": 14,
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
        "backbone_label": "Frozen BioCLIP-2 ViT-L/14",
        "experiment_dir": "reid_openset/bioclip2",
        "default_lr": 0.0005,
    },
}


# ============================================================================
# Shared constants
# ============================================================================

ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64
BATCH_K = 8
MIN_P = 5
TARGET_SAMPLES_PER_INDIVIDUAL = 32
QUERY_QUALITY_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================================
# Model creation dispatcher
# ============================================================================

def create_arcface_model(model_name, embedding_dim=128, image_size=None, device="cuda"):
    """Dispatch to the appropriate factory in utils/arcface.py."""
    import utils.arcface as arcface_module

    config = MODEL_CONFIGS[model_name]
    factory = getattr(arcface_module, config["factory"])

    kwargs = {"embedding_dim": embedding_dim, "device": device}
    if model_name == "bioclip2":
        kwargs["image_size"] = image_size or config["native_size"]

    return factory(**kwargs)


# ============================================================================
# Transform creation
# ============================================================================

def create_reid_transform(size, mean, std):
    """Create backbone-specific transform pipeline."""
    return T.Compose([
        T.Resize(size=(size, size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def create_transform_for_model(model_name, size=None):
    """Create transform using MODEL_CONFIGS defaults."""
    config = MODEL_CONFIGS[model_name]
    if size is None:
        size = config["native_size"]
    return create_reid_transform(size, config["norm_mean"], config["norm_std"])


# ============================================================================
# Dataset classes
# ============================================================================

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
            self.quality_scores.append(sample.get('pelage_score', 0.0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        label = self.individual_to_class[sample['id']]
        quality = sample.get('pelage_score', 0.0)
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


# ============================================================================
# Training utilities
# ============================================================================

def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch_arcface(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with ArcFace loss."""
    model.train()
    criterion.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        # Support both 2-tuple (image, label) and 3-tuple (image, label, quality)
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)

        optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================================
# Dataset loading and splitting
# ============================================================================

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
    ids = np.array(dataset['id'], dtype=object)
    ymdh = np.array(dataset['ymdh'], dtype=np.int64)

    # Quality scores if available
    quality_scores = None
    if 'pelage_score' in dataset.column_names:
        quality_scores = np.array(dataset['pelage_score'], dtype=np.float32)

    # Build ID-to-indices mapping
    id_to_indices = defaultdict(list)
    for idx, ind_id in enumerate(ids):
        id_to_indices[ind_id].append(idx)

    cache = {
        'ids': ids,
        'ymdh': ymdh,
        'id_to_indices': dict(id_to_indices),
        'dataset_size': len(dataset)
    }
    if quality_scores is not None:
        cache['quality_scores'] = quality_scores

    return cache


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
        all_ind_indices = set(id_to_indices.get(ind_id, []))

        # EXCLUDE hygiene validation indices
        hygiene_val_indices = set(config['validation_indices'].get(ind_id, {}).get('indices', []))
        available_indices = list(all_ind_indices - hygiene_val_indices)

        if len(available_indices) == 0:
            print(f"  WARNING: {ind_id} has NO samples after excluding hygiene val")
            continue

        available_ymdh = [ymdh_arr[i] for i in available_indices]
        unique_ymdh = sorted(set(available_ymdh))

        n_ymdh = len(unique_ymdh)
        split_point = n_ymdh // 2
        train_ymdh_set = set(unique_ymdh[:split_point])
        test_ymdh_set = set(unique_ymdh[split_point:])

        train_pool = [i for i in available_indices if ymdh_arr[i] in train_ymdh_set]
        test_pool = [i for i in available_indices if ymdh_arr[i] in test_ymdh_set]

        random.shuffle(train_pool)
        train_sampled = train_pool[:TARGET_SAMPLES_PER_INDIVIDUAL]

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

    train_dataset = dataset.select(all_train_indices) if all_train_indices else None
    test_dataset = dataset.select(all_test_indices) if all_test_indices else None

    dataset_info['total_train'] = len(all_train_indices)
    dataset_info['total_test'] = len(all_test_indices)

    print(f"Total: {len(all_train_indices)} train, {len(all_test_indices)} test")

    return train_dataset, test_dataset, individual_to_class, dataset_info


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_recall_simple(model, train_dataset, test_dataset, individual_to_class,
                           transform, device, batch_size=32):
    """
    Compute macro-averaged Recall@1 using cosine similarity.
    No quality filtering, no open-set evaluation.

    Returns: recall_at_1 value
    """
    model.eval()

    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    test_torch = ArcFaceDataset(test_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    # Gallery embeddings (train)
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad():
        for batch in train_loader:
            images = batch[0].to(device)
            labels = batch[1]
            emb = model(images)
            gallery_embeddings.append(emb)
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query embeddings (test)
    query_embeddings = []
    query_labels = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            labels = batch[1]
            emb = model(images)
            query_embeddings.append(emb)
            query_labels.extend(labels.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels_np = np.array(query_labels)

    # Compute cosine similarity
    query_emb_norm = F.normalize(query_embeddings, p=2, dim=1)
    gallery_emb_norm = F.normalize(gallery_embeddings, p=2, dim=1)
    similarity = torch.mm(query_emb_norm, gallery_emb_norm.t())

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


# ============================================================================
# Hyperparameter loading
# ============================================================================

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

    # Load best LR from lr_results
    lr_results_dir = os.path.join(experiment_dir, 'results/opt/lr')
    lr_files = glob.glob(os.path.join(lr_results_dir, 'lr=*.json'))
    if lr_files:
        best_lr_recall = 0.0
        for f in lr_files:
            with open(f, 'r') as fp:
                result = json.load(fp)
            recall = result['results']['best_test_recall_at_1']
            if recall > best_lr_recall:
                best_lr_recall = recall
                best_lr = result['config']['learning_rate']
        print(f"Loaded best LR from sweep: {best_lr} (R@1={best_lr_recall:.4f})")
    else:
        print(f"No LR sweep results found, using default: {best_lr}")

    print(f"Image size (native): {best_size}")

    # Load best embedding dim from embedding_dim_results
    emb_results_dir = os.path.join(experiment_dir, 'results/opt/embedding_dim')
    emb_files = glob.glob(os.path.join(emb_results_dir, 'embedding_dim=*.json'))
    if emb_files:
        best_emb_recall = 0.0
        for f in emb_files:
            with open(f, 'r') as fp:
                result = json.load(fp)
            recall = result['results']['best_test_recall_at_1']
            if recall > best_emb_recall:
                best_emb_recall = recall
                best_embedding_dim = result['config']['embedding_dim']
        print(f"Loaded best embedding dim from sweep: {best_embedding_dim} (R@1={best_emb_recall:.4f})")
    else:
        print(f"No embedding dim sweep results found, using default: {best_embedding_dim}")

    return best_lr, best_size, best_embedding_dim


# ============================================================================
# Hygiene sweep: quality filtering and gallery creation
# ============================================================================

def filter_training_pool_by_quality(metadata_cache, indices, threshold):
    """Use vectorized filtering with cached quality scores."""
    from utils.optimized_filters import filter_training_pool_by_quality_vectorized
    return filter_training_pool_by_quality_vectorized(
        metadata_cache['quality_scores'],
        indices,
        threshold
    )


def get_rare_individual_indices(metadata_cache, valid_individuals, quality_threshold=0.0,
                                promoted_individuals=None, excluded_individuals=None):
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
        train_dataset, val_dataset, individual_to_class, dataset_info
    """
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
        val_indices = config['validation_indices'][ind_id]['indices']
        all_val_indices.extend(val_indices)
        dataset_info['query_samples_per_individual'][ind_id] = len(val_indices)

        all_ind_indices = set(id_to_indices.get(ind_id, []))
        train_candidates = list(all_ind_indices - set(val_indices))

        eligible_pool = filter_training_pool_by_quality(metadata_cache, train_candidates, threshold)
        dataset_info['eligible_pool_per_individual'][ind_id] = len(eligible_pool)

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

    train_dataset = dataset.select(all_train_indices) if all_train_indices else None
    val_dataset = dataset.select(all_val_indices)

    print(f"Total: {len(all_train_indices)} gallery, {len(all_val_indices)} query")

    return train_dataset, val_dataset, individual_to_class, dataset_info


# ============================================================================
# Hygiene sweep: open-set evaluation
# ============================================================================

def compute_rare_embeddings(model, dataset, rare_indices, rare_quality, rare_labels,
                            transform, device, embedding_dim=128, batch_size=32):
    """Compute embeddings for rare/unknown individuals."""
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
        embeddings = torch.empty(0, embedding_dim)

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
    """Compute L2-normalized cosine similarity matrix."""
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
    return torch.mm(query_emb, gallery_emb.t())


def compute_open_set_metrics_cosine(known_query_labels, known_query_quality, known_scores,
                                     unknown_query_labels, unknown_query_quality, unknown_scores,
                                     score_threshold, quality_thresholds, gallery_labels):
    """
    Compute macro-averaged open-set metrics for raw cosine similarity scores.

    Uses per-individual averaging for both known accept rate and unknown reject rate.
    """
    if isinstance(known_query_labels, torch.Tensor):
        known_labels_np = known_query_labels.cpu().numpy()
    else:
        known_labels_np = np.array(known_query_labels)

    results = {}

    for q_thresh in quality_thresholds:
        # Known accept rate (macro-averaged)
        k_mask = known_query_quality >= q_thresh
        per_ind_accept = []
        n_known_individuals = 0

        if k_mask.sum() > 0:
            known_indices = np.where(k_mask)[0]
            unique_known_labels = np.unique(known_labels_np[known_indices])

            for ind_label in unique_known_labels:
                ind_mask = (known_labels_np == ind_label) & k_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                accepted = 0
                for i in ind_indices:
                    max_score = known_scores[i].max().item()
                    if max_score >= score_threshold:
                        accepted += 1

                per_ind_accept.append(accepted / len(ind_indices))

            n_known_individuals = len(per_ind_accept)

        known_accept_rate = np.mean(per_ind_accept) if per_ind_accept else 0.0

        # Unknown reject rate (macro-averaged)
        u_mask = unknown_query_quality >= q_thresh
        per_ind_reject = []
        n_unknown_individuals = 0

        if len(unknown_query_labels) > 0 and u_mask.sum() > 0:
            unknown_indices = np.where(u_mask)[0]
            unique_unknown_labels = np.unique(unknown_query_labels[unknown_indices])

            for ind_id in unique_unknown_labels:
                ind_mask = (unknown_query_labels == ind_id) & u_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                rejected = 0
                for i in ind_indices:
                    max_score = unknown_scores[i].max().item()
                    if max_score < score_threshold:
                        rejected += 1

                per_ind_reject.append(rejected / len(ind_indices))

            n_unknown_individuals = len(per_ind_reject)

        unknown_reject_rate = np.mean(per_ind_reject) if per_ind_reject else 1.0

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
                                  gallery_threshold, criterion, embedding_dim=128, batch_size=32,
                                  promoted_individuals=None, excluded_individuals=None):
    """
    Evaluate model computing Recall@1 and open-set metrics with raw cosine similarity.

    Returns:
        query_quality_metrics, open_set_metrics, val_loss
    """
    model.eval()

    # 1. Extract Embeddings
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

    # Validation loss
    val_loss = compute_validation_loss(query_embeddings, query_labels, criterion, device)

    # Rare/Unknown
    rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
        metadata_cache, valid_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
    )
    if rare_indices:
        rare_emb, rare_quality_arr, rare_labels_arr = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels_str,
            transform, device, embedding_dim=embedding_dim
        )
        rare_emb = rare_emb.to(device)
    else:
        rare_emb = torch.empty(0, embedding_dim).to(device)
        rare_quality_arr = np.array([])
        rare_labels_arr = np.array([])

    # 2. Compute Cosine Similarity Matrices
    scores_known = compute_cosine_similarity(query_embeddings, gallery_embeddings)

    if len(rare_emb) > 0:
        scores_unknown = compute_cosine_similarity(rare_emb, gallery_embeddings)
    else:
        scores_unknown = torch.empty(0, len(gallery_embeddings)).to(device)

    scores_gal_gal = compute_cosine_similarity(gallery_embeddings, gallery_embeddings)

    # 3. Compute Metrics

    # A. Recall@1 (Closed Set) - macro-averaged
    query_quality_metrics = {}
    query_labels_np = query_labels.cpu().numpy()

    for thresh in QUERY_QUALITY_THRESHOLDS:
        mask = query_quality >= thresh
        count = int(mask.sum())
        if count == 0:
            query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": 0.0, "count": 0}
            continue

        mask_indices = np.where(mask)[0]
        individual_correct = {}
        individual_total = {}

        for idx in mask_indices:
            true_label = query_labels_np[idx]
            pred_idx = scores_known[idx].argmax().item()
            pred_label = gallery_labels[pred_idx].item()

            individual_total[true_label] = individual_total.get(true_label, 0) + 1
            if pred_label == true_label:
                individual_correct[true_label] = individual_correct.get(true_label, 0) + 1

        per_ind_recall = [
            individual_correct.get(label, 0) / individual_total[label]
            for label in individual_total
        ]
        recall = np.mean(per_ind_recall) if per_ind_recall else 0.0

        query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    # B. Threshold Calibration (LOO on Gallery Scores)
    scores_gal_gal.fill_diagonal_(-1.0)

    unique_individuals = gallery_labels.unique().tolist()
    fold_results = []
    thresholds = np.linspace(0.0, 1.0, 101)

    for held_out_ind in unique_individuals:
        gallery_mask = (gallery_labels != held_out_ind)
        unknown_mask = (gallery_labels == held_out_ind)

        unknown_max_scores = scores_gal_gal[unknown_mask][:, gallery_mask].max(dim=1).values

        known_results_by_ind = {}
        gallery_indices = torch.where(gallery_mask)[0]

        for idx in gallery_indices:
            label_i = gallery_labels[idx].item()
            row = scores_gal_gal[idx]

            valid_mask = gallery_mask.clone()
            valid_mask[idx] = False

            if valid_mask.sum() > 0:
                scores_to_valid = row[valid_mask]
                max_score, local_idx = scores_to_valid.max(dim=0)

                valid_indices = torch.where(valid_mask)[0]
                pred_idx = valid_indices[local_idx]
                pred_label = gallery_labels[pred_idx].item()
                is_correct = (pred_label == label_i)
                known_results_by_ind.setdefault(label_i, []).append((max_score.item(), is_correct))

        fold_best_ba, fold_best_thresh = 0.0, 0.5
        fold_best_known_accept, fold_best_unknown_reject = 0.0, 0.0

        for thresh in thresholds:
            per_ind_accept = []
            for ind_label, results in known_results_by_ind.items():
                accepted = sum(1 for s, c in results if s >= thresh)
                per_ind_accept.append(accepted / len(results))
            known_accept = np.mean(per_ind_accept) if per_ind_accept else 0.0

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

    optimal_thresh = np.mean([f['threshold'] for f in fold_results])
    calibration_ba = np.mean([f['ba'] for f in fold_results])
    calibration_known_accept = np.mean([f['known_accept'] for f in fold_results])
    calibration_unknown_reject = np.mean([f['unknown_reject'] for f in fold_results])
    n_folds = len(fold_results)

    # C. Open Set Metrics (Balanced Accuracy)
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


# ============================================================================
# Hygiene sweep: job distribution and training
# ============================================================================

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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
                         learning_rate, image_size, embedding_dim, epochs=50):
    """Train one hygiene sweep configuration and return results."""
    from utils.arcface import ArcFaceLoss, PKBatchSampler
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
    print(f"{'='*60}")

    # Get valid individuals and diversity-based classification from config
    valid_individuals = config.get('valid_individuals', [])
    promoted_individuals = config.get('promoted_to_rare', [])
    excluded_individuals = config.get('excluded_entirely', [])

    # Check feasibility for this threshold and gallery_size
    feasible_individuals = []
    for ind_id in valid_individuals:
        training_compat = config['training_compatibility'].get(ind_id, {})
        threshold_key = f"threshold_{threshold:.2f}"
        if threshold_key not in training_compat.get('threshold_compatibility', {}):
            threshold_key = "threshold_0.00"
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

    effective_k = min(BATCH_K, gallery_size)
    if len(train_dataset) < MIN_P * effective_k:
        print(f"Not enough training samples ({len(train_dataset)}) for PK batching")
        return None

    # Create model
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    model, emb_dim = create_arcface_model(model_name, embedding_dim=embedding_dim,
                                           image_size=image_size, device=device)
    print(f"Using device: {device}")

    # Create transforms and dataset
    transform = create_transform_for_model(model_name, size=image_size)
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
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

        query_quality_metrics, open_set_metrics, val_loss = evaluate_recall_with_openset(
            model, train_dataset, val_dataset, individual_to_class, transform, device,
            dataset, feasible_individuals, metadata_cache, gallery_threshold=threshold,
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

        thresh_mean = open_set_metrics.get('threshold_calibration', {}).get('threshold', 0.0)

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
            'epochs': epochs,
            'batch_size_k': BATCH_K,
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
    Main entry point for running a hygiene sweep experiment.

    Loads dataset, config, best hyperparams, distributes work across SLURM jobs,
    and trains each configuration.
    """
    config = MODEL_CONFIGS[model_name]

    print("=" * 80)
    print(f"Open-Set {config['backbone_label']} + ArcFace + Raw Cosine Gallery Hygiene Sweep - Job {args.idx}")
    print("=" * 80)

    # Load best hyperparameters
    best_lr, best_size, best_embedding_dim = load_best_hyperparams(model_name)

    # Load dataset and config
    print("\nLoading dataset...")
    dataset = load_reidentification_dataset()
    metadata_cache = build_metadata_cache(dataset)
    feasibility_config = load_feasibility_config()

    if not feasibility_config:
        print("Failed to load feasibility config")
        return

    total_combinations = len(THRESHOLDS) * len(GALLERY_SIZES) * len(SEEDS)
    print(f"\nExperiment parameters:")
    print(f"  Model: {config['backbone_label']} + Projection Head ({best_embedding_dim}-d) + ArcFace")
    print(f"  Loss: ArcFace (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    print(f"  Score Normalization: Raw Cosine Similarity (L2-normalized)")
    print(f"  Optimizer: AdamW (lr={best_lr}, default settings)")
    print(f"  Embedding: {best_embedding_dim}-d (trainable projection)")
    print(f"  Image size: {best_size}")
    print(f"  Thresholds: {THRESHOLDS}")
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

def get_valid_individuals(config, min_p=MIN_P):
    """Get individuals compatible with gallery_size=64 at threshold=0.0."""
    valid_individuals = []
    for ind_id in config.get('valid_individuals', []):
        training_compat = config['training_compatibility'].get(ind_id, {})
        threshold_compat = training_compat.get('threshold_compatibility', {}).get('threshold_0.00', {})
        compatible_sizes = threshold_compat.get('compatible_training_sizes', [])
        if 64 in compatible_sizes:
            valid_individuals.append(ind_id)

    if len(valid_individuals) < min_p:
        print(f"Not enough individuals ({len(valid_individuals)}) for PK sampling (need {min_p})")
        return None

    return valid_individuals


def run_opt_training(model_name, sweep_param_name, sweep_param_value, args,
                     dataset, config, metadata_cache,
                     learning_rate=None, image_size=None, embedding_dim=128,
                     epochs=50, seed=0):
    """
    Shared training loop for all opt sweep scripts (LR, resize, embedding_dim).

    Args:
        model_name: Key in MODEL_CONFIGS
        sweep_param_name: Name of swept param ('lr', 'resize', 'embedding_dim')
        sweep_param_value: Value of swept param
        args: Argparse namespace (needs .output_dir, .overwrite, .device, .idx)
        dataset: HuggingFace dataset
        config: Feasibility config
        metadata_cache: Metadata cache dict
        learning_rate: LR to use (overrides default)
        image_size: Image size to use (overrides default)
        embedding_dim: Embedding dimension
        epochs: Number of training epochs
        seed: Random seed

    Returns:
        filename if saved, None if skipped
    """
    from utils.arcface import ArcFaceLoss, PKBatchSampler

    model_config = MODEL_CONFIGS[model_name]
    if learning_rate is None:
        learning_rate = model_config["default_lr"]
    if image_size is None:
        image_size = model_config["native_size"]

    set_all_seeds(seed)

    # Output path
    if sweep_param_name == 'lr':
        filename = f"lr={sweep_param_value:.6f}.json"
    elif sweep_param_name == 'embedding_dim':
        filename = f"embedding_dim={sweep_param_value}.json"
    else:
        filename = f"{sweep_param_name}={sweep_param_value}.json"

    output_path = os.path.join(args.output_dir, filename)

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skipping existing result: {filename}")
        return None

    print(f"\n{'='*60}")
    print(f"Training: {sweep_param_name}={sweep_param_value}")
    print(f"{'='*60}")

    # Get valid individuals
    valid_individuals = get_valid_individuals(config)
    if valid_individuals is None:
        return None
    print(f"Using {len(valid_individuals)} individuals: {', '.join(valid_individuals)}")

    # Create temporal split dataset
    train_dataset, test_dataset, individual_to_class, dataset_info = create_ymdh_split_dataset(
        dataset, valid_individuals, metadata_cache, config, seed=seed
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

    # Training loop
    start_time = time.time()
    epoch_history = []
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, criterion, device)

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
            'target_samples_per_individual': TARGET_SAMPLES_PER_INDIVIDUAL,
            'image_size': image_size,
            'embedding_dim': embedding_dim,
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



# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/data/hf_cache"

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
    sub_lr.add_argument("--epochs", type=int, default=50)
    sub_lr.add_argument("--seed", type=int, default=0)

    # opt_embedding_dim subcommand
    sub_emb = subparsers.add_parser("opt_embedding_dim", help="Embedding dimension sweep")
    add_common_args(sub_emb)
    sub_emb.add_argument("--values", type=int, nargs="+", required=True, help="Embedding dims to sweep")
    sub_emb.add_argument("--lr", type=float, required=True, help="Fixed learning rate")
    sub_emb.add_argument("--image-size", type=int, required=True, help="Fixed image size")
    sub_emb.add_argument("--epochs", type=int, default=50)
    sub_emb.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    model_name = args.model
    config = MODEL_CONFIGS[model_name]

    if args.command == "hygiene":
        run_hygiene_sweep(model_name, args)

    elif args.command == "opt_lr":
        lrs = args.values

        print("=" * 80)
        print(f"LR Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
        print("=" * 80)

        if args.idx >= len(lrs):
            print(f"Job {args.idx} has no work (only {len(lrs)} LRs)")
            sys.exit(0)

        dataset = load_reidentification_dataset()
        metadata_cache = build_metadata_cache(dataset)
        feasibility_config = load_feasibility_config()
        if not feasibility_config:
            sys.exit(1)

        lr = lrs[args.idx]
        print(f"\nLearning rate: {lr}, Image size: {args.image_size}, Epochs: {args.epochs}")

        try:
            run_opt_training(
                model_name, "lr", lr, args, dataset, feasibility_config, metadata_cache,
                learning_rate=lr, image_size=args.image_size,
                embedding_dim=args.embedding_dim,
                epochs=args.epochs, seed=args.seed,
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} completed!")

    elif args.command == "opt_embedding_dim":
        dims = args.values

        print("=" * 80)
        print(f"Embedding Dim Sweep - {config['backbone_label']} Re-ID - Job {args.idx}")
        print("=" * 80)

        if args.idx >= len(dims):
            print(f"Job {args.idx} has no work (only {len(dims)} dims)")
            sys.exit(0)

        dataset = load_reidentification_dataset()
        metadata_cache = build_metadata_cache(dataset)
        feasibility_config = load_feasibility_config()
        if not feasibility_config:
            sys.exit(1)

        emb_dim = dims[args.idx]
        print(f"\nEmbedding dim: {emb_dim}, LR: {args.lr}, Size: {args.image_size}, Epochs: {args.epochs}")

        try:
            run_opt_training(
                model_name, "embedding_dim", emb_dim, args, dataset, feasibility_config, metadata_cache,
                learning_rate=args.lr, image_size=args.image_size, embedding_dim=emb_dim,
                epochs=args.epochs, seed=args.seed,
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nJob {args.idx} completed!")

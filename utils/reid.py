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
        "native_size": 224,
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
QUERY_QUALITY_THRESHOLDS = [0.0, 0.25, 0.5]
EVAL_BATCH_SIZE = 128
EVAL_NUM_WORKERS = 4


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


def load_reidentification_test_dataset():
    """Load the wolverines test split for reidentification."""
    print("Loading wolverines dataset (reidentification test split)...")
    dataset = load_dataset("kdoherty/wolverines", "reidentification", split="test")
    print(f"Loaded {len(dataset)} test samples")
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


def create_ymdh_split_dataset(dataset, individuals, metadata_cache, seed=0):
    """
    Create train/test split based on capture events (ymdh).

    For each individual:
    1. Get all sample indices from the training set
    2. Get unique ymdh values and sort chronologically
    3. Earlier 50% of ymdh -> train pool
    4. Later 50% of ymdh -> test pool
    5. Train gets priority: sample TARGET_SAMPLES_PER_INDIVIDUAL (or all available if <32)
    6. Test gets remainder: sample TARGET_SAMPLES_PER_INDIVIDUAL (or all available if <32)

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
        available_indices = list(id_to_indices.get(ind_id, []))

        if len(available_indices) == 0:
            print(f"  WARNING: {ind_id} has NO samples")
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
                           transform, device, batch_size=EVAL_BATCH_SIZE):
    """
    Compute macro-averaged Recall@1 using cosine similarity.
    No quality filtering, no open-set evaluation.

    Returns: recall_at_1 value
    """
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    test_torch = ArcFaceDataset(test_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)
    test_loader = DataLoader(test_torch, batch_size=batch_size, shuffle=False,
                             num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    # Gallery embeddings (train)
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]
            emb = model(images)
            gallery_embeddings.append(emb)
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).float()
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query embeddings (test)
    query_embeddings = []
    query_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]
            emb = model(images)
            query_embeddings.append(emb)
            query_labels.extend(labels.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0).float()
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


def get_rare_individual_indices(metadata_cache, qualified_individuals, quality_threshold=0.0,
                                promoted_individuals=None, excluded_individuals=None):
    """Use vectorized operations with metadata cache."""
    from utils.optimized_filters import get_rare_individual_indices_vectorized
    return get_rare_individual_indices_vectorized(
        metadata_cache,
        qualified_individuals,
        quality_threshold,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
    )


def create_filtered_gallery_dataset(dataset, individuals, gallery_size, threshold, seed,
                                     metadata_cache, config):
    """
    Create gallery/query split with filtered gallery using train-split validation.

    Gallery: Training samples (excluding validation indices) filtered by pelage_score >= threshold, then sampled
    Query: Validation indices from greedy temporal split (from train dataset)

    Returns:
        train_dataset, val_dataset, individual_to_class, dataset_info
    """
    set_all_seeds(seed)

    id_to_indices = metadata_cache['id_to_indices']
    validation_indices = config.get('validation_indices', {})

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
        # Query: validation indices from greedy temporal split (train dataset)
        val_idx = validation_indices.get(ind_id, {}).get('indices', [])
        val_idx_set = set(val_idx)
        all_val_indices.extend(val_idx)
        dataset_info['query_samples_per_individual'][ind_id] = len(val_idx)

        # Gallery: quality-filtered train images MINUS validation indices
        all_ind_indices = list(id_to_indices.get(ind_id, []))
        train_only_indices = [i for i in all_ind_indices if i not in val_idx_set]

        eligible_pool = filter_training_pool_by_quality(metadata_cache, train_only_indices, threshold)
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

        print(f"  {ind_id}: {len(train_sampled)}/{len(eligible_pool)} gallery (threshold>={threshold}), {len(val_idx)} query")

    train_dataset_out = dataset.select(all_train_indices) if all_train_indices else None
    val_dataset = dataset.select(all_val_indices) if all_val_indices else None

    print(f"Total: {len(all_train_indices)} gallery, {len(all_val_indices)} query")

    return train_dataset_out, val_dataset, individual_to_class, dataset_info


# ============================================================================
# Hygiene sweep: open-set evaluation
# ============================================================================

def compute_rare_embeddings(model, dataset, rare_indices, rare_quality, rare_labels,
                            transform, device, embedding_dim=128, batch_size=EVAL_BATCH_SIZE):
    """Compute embeddings for rare/unknown individuals."""
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    rare_dataset = RareIndividualsDataset(dataset, rare_indices, rare_quality, transform)
    rare_loader = DataLoader(rare_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    embeddings = []
    qualities = []

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, quality in rare_loader:
            images = images.to(device, non_blocking=True)
            emb = model(images)
            embeddings.append(emb.cpu())
            qualities.extend(quality.tolist())

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0).float()
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

                max_scores = known_scores[ind_indices].max(dim=1).values
                accepted = (max_scores >= score_threshold).sum().item()

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

                max_scores = unknown_scores[ind_indices].max(dim=1).values
                rejected = (max_scores < score_threshold).sum().item()

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
                                  transform, device, dataset, qualified_individuals, metadata_cache,
                                  gallery_threshold, criterion, embedding_dim=128,
                                  batch_size=EVAL_BATCH_SIZE,
                                  promoted_individuals=None, excluded_individuals=None):
    """
    Evaluate model computing Recall@1 and open-set metrics with raw cosine similarity.

    Rare/unknown individuals are pooled from the train dataset only.

    Returns:
        query_quality_metrics, open_set_metrics, val_loss
    """
    model.eval()
    use_amp = (device.type == 'cuda') if isinstance(device, torch.device) else (device == 'cuda')

    # 1. Extract Embeddings
    train_torch = ArcFaceDataset(train_dataset, transform, individual_to_class)
    val_torch = ArcFaceDataset(val_dataset, transform, individual_to_class)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False,
                              num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)
    val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False,
                            num_workers=EVAL_NUM_WORKERS, pin_memory=use_amp)

    # Gallery
    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gallery_embeddings.append(model(images))
            gallery_labels.extend(labels.tolist())
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).float()
    gallery_labels = torch.tensor(gallery_labels).to(device)

    # Query (Known)
    query_embeddings = []
    query_labels = []
    query_quality = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels, quality in val_loader:
            images = images.to(device, non_blocking=True)
            query_embeddings.append(model(images))
            query_labels.extend(labels.tolist())
            query_quality.extend(quality.tolist())
    query_embeddings = torch.cat(query_embeddings, dim=0).float()
    query_labels = torch.tensor(query_labels).to(device)
    query_quality = np.array(query_quality)

    # Validation loss
    val_loss = compute_validation_loss(query_embeddings, query_labels, criterion, device)

    # Rare/Unknown — pool from train dataset
    rare_indices, rare_quality, rare_labels_str = get_rare_individual_indices(
        metadata_cache, qualified_individuals, quality_threshold=0.0,
        promoted_individuals=promoted_individuals,
        excluded_individuals=excluded_individuals
    )

    all_rare_emb = []
    all_rare_quality = []
    all_rare_labels = []

    if rare_indices:
        emb, qual, lab = compute_rare_embeddings(
            model, dataset, rare_indices, rare_quality, rare_labels_str,
            transform, device, embedding_dim=embedding_dim
        )
        all_rare_emb.append(emb)
        all_rare_quality.append(qual)
        all_rare_labels.append(lab)

    if all_rare_emb:
        rare_emb = torch.cat(all_rare_emb, dim=0).to(device)
        rare_quality_arr = np.concatenate(all_rare_quality)
        rare_labels_arr = np.concatenate(all_rare_labels)
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

        pred_indices = scores_known[mask_indices].argmax(dim=1)
        pred_labels = gallery_labels[pred_indices].cpu().numpy()
        true_labels = query_labels_np[mask_indices]
        correct = (pred_labels == true_labels)

        for label in np.unique(true_labels):
            label_mask = (true_labels == label)
            individual_total[label] = int(label_mask.sum())
            individual_correct[label] = int(correct[label_mask].sum())

        per_ind_recall = [
            individual_correct.get(label, 0) / individual_total[label]
            for label in individual_total
        ]
        recall = np.mean(per_ind_recall) if per_ind_recall else 0.0

        query_quality_metrics[f"q>={thresh}"] = {"recall_at_1": recall, "count": count}

    # B. Cosine Threshold Selection (sweep on validation scores)
    #
    # Per quality level, sweep 101 thresholds on query-gallery and
    # unknown-gallery scores to find the cosine threshold maximizing BA.
    # Uses macro-averaged per-individual accept/reject rates via matmul.
    thresholds = np.linspace(0.0, 1.0, 101)
    T = len(thresholds)
    thresholds_t = torch.tensor(thresholds, dtype=torch.float32, device=gallery_labels.device)

    balanced_metrics_by_quality = {}

    for q_thresh in QUERY_QUALITY_THRESHOLDS:
        q_key = f"q>={q_thresh}"

        # Known queries at this quality level
        k_mask = query_quality >= q_thresh
        k_indices = np.where(k_mask)[0]

        if len(k_indices) > 0:
            known_max_scores = scores_known[k_indices].max(dim=1).values
            known_labels_masked = query_labels_np[k_indices]
            unique_known = np.unique(known_labels_masked)

            # Indicator matrix for known individuals: (N_known, K_known)
            k_label_to_col = {l: j for j, l in enumerate(unique_known)}
            k_indicator = torch.zeros(len(k_indices), len(unique_known),
                                       device=gallery_labels.device)
            for s, l in enumerate(known_labels_masked):
                k_indicator[s, k_label_to_col[l]] = 1.0
            k_counts = k_indicator.sum(dim=0)

            # (T, N_known) @ (N_known, K_known) / counts -> (T, K_known) -> mean -> (T,)
            accepted = (known_max_scores.unsqueeze(0) >= thresholds_t.unsqueeze(1)).float()
            known_accept_curve = (accepted @ k_indicator / k_counts.unsqueeze(0)).mean(dim=1)
        else:
            known_accept_curve = torch.zeros(T, device=gallery_labels.device)

        # Unknown queries at this quality level
        u_mask = rare_quality_arr >= q_thresh if len(rare_quality_arr) > 0 else np.array([], dtype=bool)

        if len(rare_labels_arr) > 0 and u_mask.sum() > 0:
            u_indices = np.where(u_mask)[0]
            unknown_max_scores = scores_unknown[u_indices].max(dim=1).values
            unknown_labels_masked = rare_labels_arr[u_indices]
            unique_unknown = np.unique(unknown_labels_masked)

            u_label_to_col = {l: j for j, l in enumerate(unique_unknown)}
            u_indicator = torch.zeros(len(u_indices), len(unique_unknown),
                                       device=gallery_labels.device)
            for s, l in enumerate(unknown_labels_masked):
                u_indicator[s, u_label_to_col[l]] = 1.0
            u_counts = u_indicator.sum(dim=0)

            rejected = (unknown_max_scores.unsqueeze(0) < thresholds_t.unsqueeze(1)).float()
            unknown_reject_curve = (rejected @ u_indicator / u_counts.unsqueeze(0)).mean(dim=1)
        else:
            unknown_reject_curve = torch.ones(T, device=gallery_labels.device)

        ba_curve = (known_accept_curve + unknown_reject_curve) / 2
        best_idx = ba_curve.argmax().item()
        optimal_thresh_q = float(thresholds[best_idx])

        # Evaluate at optimal threshold for full metrics
        q_metrics = compute_open_set_metrics_cosine(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_arr,
            unknown_scores=scores_unknown,
            score_threshold=optimal_thresh_q,
            quality_thresholds=[q_thresh],
            gallery_labels=gallery_labels
        )
        balanced_metrics_by_quality[q_key] = q_metrics[q_key]
        balanced_metrics_by_quality[q_key]['cosine_threshold'] = optimal_thresh_q

    # q>=0.0 threshold stored at top level for backward compatibility
    optimal_thresh = balanced_metrics_by_quality.get('q>=0.0', {}).get('cosine_threshold', 0.0)

    open_set_metrics = {
        'threshold_calibration': {
            'method': 'validation_sweep',
            'threshold': optimal_thresh,
        },
        'by_quality': balanced_metrics_by_quality
    }

    return query_quality_metrics, open_set_metrics, val_loss


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
                         learning_rate, image_size, embedding_dim, epochs=100):
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

    # Get qualified individuals from config (preprocessing already enforces criteria)
    feasible_individuals = config.get('qualified_individuals', [])
    promoted_individuals = config.get('promoted_to_rare', [])
    excluded_individuals = config.get('excluded_entirely', [])

    if len(feasible_individuals) < MIN_P:
        print(f"Not enough individuals ({len(feasible_individuals)}) for PK sampling (need {MIN_P})")
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

    # Load datasets and config
    print("\nLoading datasets...")
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

def get_qualified_individuals(config, min_p=MIN_P):
    """Get qualified individuals from config (preprocessing already enforces criteria)."""
    qualified_individuals = config.get('qualified_individuals', [])

    if len(qualified_individuals) < min_p:
        print(f"Not enough individuals ({len(qualified_individuals)}) for PK sampling (need {min_p})")
        return None

    return qualified_individuals


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

    # Create PK batch sampler
    effective_k = min(BATCH_K, TARGET_SAMPLES_PER_INDIVIDUAL)
    try:
        pk_sampler = PKBatchSampler(
            labels=train_torch_dataset.get_labels(),
            p=min(MIN_P, len(qualified_individuals)),
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

def load_best_hygiene_config(model_name, criterion='harmonic_mean', threshold_filter=None):
    """
    Read all hygiene sweep results for a backbone and select the best operating point.

    Groups by (threshold, gallery_size), extracts best epoch per seed by criterion,
    averages across seeds, returns the (threshold, gallery_size, epoch) with highest mean.

    Args:
        model_name: Key in MODEL_CONFIGS
        criterion: 'recall', 'balanced_accuracy', or 'harmonic_mean'
        threshold_filter: If set, only consider groups with this threshold value

    Returns:
        dict with {threshold, gallery_size, best_epoch, score}
    """
    config = MODEL_CONFIGS[model_name]
    results_dir = os.path.join(config['experiment_dir'], 'results')

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

            # Build per-seed lookup: epoch_num -> score
            # Only include epochs that have evaluation data.
            all_histories = [r['epoch_history'] for r in group_results]

            eval_epochs = []
            for entry in all_histories[0]:
                if 'query_quality_metrics' in entry:
                    eval_epochs.append(entry['epoch'])

            # For each eval epoch, average the criterion across seeds
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

            # Extract cosine threshold at best epoch (for open-set tasks)
            # Prefer per-quality threshold; fall back to top-level for old results
            cosine_thresholds = []
            for history in all_histories:
                for entry in history:
                    if entry['epoch'] == best_epoch_num:
                        open_set = entry.get('open_set', {})
                        ct = open_set.get('by_quality', {}).get(
                            q_key, {}).get('cosine_threshold')
                        if ct is None:
                            ct = open_set.get(
                                'threshold_calibration', {}).get('threshold')
                        if ct is not None:
                            cosine_thresholds.append(ct)
                        break
            mean_cosine_thresh = float(np.mean(cosine_thresholds)) if cosine_thresholds else None

            if best_mean > best_overall_score:
                best_overall_score = best_mean
                best_config = {
                    'threshold': threshold,
                    'gallery_size': gallery_size,
                    'query_quality_threshold': q_thresh,
                    'best_epoch': best_epoch_num,
                    'score': float(best_mean),
                    'criterion': criterion,
                    'n_seeds': len(group_results),
                    'cosine_threshold': mean_cosine_thresh,
                }

    print(f"Best hygiene config ({criterion}): threshold={best_config['threshold']}, "
          f"gallery_size={best_config['gallery_size']}, "
          f"query_q>={best_config['query_quality_threshold']}, "
          f"epoch={best_config['best_epoch']}, "
          f"score={best_config['score']:.4f} (n={best_config['n_seeds']} seeds)")

    return best_config


# Each task selects its own optimal (threshold, gallery_size, epoch) from the
# hygiene sweep using the criterion and threshold constraint that match its goal.
TEST_TASKS = {
    'closed_unfiltered': {'criterion': 'recall',             'threshold_filter': 0.0},
    'closed_filtered':   {'criterion': 'recall',             'threshold_filter': None},
    'open_unfiltered':   {'criterion': 'balanced_accuracy',  'threshold_filter': 0.0},
    'open_filtered':     {'criterion': 'balanced_accuracy',  'threshold_filter': None},
}


def run_final_test(model_name, args, seed, task_name):
    """
    Final test evaluation using the best operating point from hygiene sweep.

    Each task selects its own optimal hyperparameters:
      - closed_unfiltered: best R@1 among threshold=0.0 configs
      - closed_filtered:   best R@1 from the full sweep
      - open_unfiltered:   best BA among threshold=0.0 configs
      - open_filtered:     best BA from the full sweep

    Args:
        model_name: Key in MODEL_CONFIGS
        args: Argparse namespace
        seed: Random seed for gallery sampling and training
        task_name: Key in TEST_TASKS
    """
    from utils.arcface import ArcFaceLoss, PKBatchSampler
    from utils.training import check_result_exists

    task_cfg = TEST_TASKS[task_name]
    criterion = task_cfg['criterion']
    threshold_filter = task_cfg['threshold_filter']

    model_config = MODEL_CONFIGS[model_name]
    experiment_dir = model_config['experiment_dir']

    # Load best operating point for this task
    best_config = load_best_hygiene_config(model_name, criterion=criterion,
                                           threshold_filter=threshold_filter)
    threshold = best_config['threshold']
    gallery_size = best_config['gallery_size']
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
    print(f"  criterion={criterion}, threshold={threshold}, gallery_size={gallery_size}, "
          f"query_q>={query_quality_threshold}, epochs={best_epoch}")
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

    if len(feasible_individuals) < MIN_P:
        print(f"Not enough individuals ({len(feasible_individuals)}) for PK sampling (need {MIN_P})")
        return None

    print(f"Using {len(feasible_individuals)} individuals: {', '.join(feasible_individuals)}")

    # Create gallery from TRAIN split (quality-filtered, sampled by seed)
    id_to_indices = train_metadata_cache['id_to_indices']
    individual_to_class = {ind: i for i, ind in enumerate(sorted(feasible_individuals))}

    all_gallery_indices = []
    gallery_info = {}

    for ind_id in feasible_individuals:
        all_ind_indices = list(id_to_indices.get(ind_id, []))
        eligible_pool = filter_training_pool_by_quality(train_metadata_cache, all_ind_indices, threshold)

        if len(eligible_pool) == 0:
            print(f"  WARNING: {ind_id} has NO samples above threshold {threshold}")
            sampled = []
        elif len(eligible_pool) < gallery_size:
            print(f"  WARNING: {ind_id} has only {len(eligible_pool)} eligible (need {gallery_size}), using all")
            sampled = eligible_pool
        else:
            random.shuffle(eligible_pool)
            sampled = eligible_pool[:gallery_size]

        all_gallery_indices.extend(sampled)
        gallery_info[ind_id] = {'sampled': len(sampled), 'eligible': len(eligible_pool)}
        print(f"  {ind_id}: {len(sampled)}/{len(eligible_pool)} gallery")

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

    effective_k = min(BATCH_K, gallery_size)
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

    # Train for exactly best_epoch epochs
    print(f"\nTraining for {best_epoch} epochs...")
    start_time = time.time()
    epoch_history = []

    for epoch in range(best_epoch):
        train_loss = train_epoch_arcface(model, train_loader, optimizer, arcface_loss, device)
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
        'threshold_filter': threshold_filter,
        'threshold': threshold,
        'gallery_size': gallery_size,
        'query_quality_threshold': query_quality_threshold,
        'best_epoch': best_epoch,
        'hygiene_score': best_config['score'],
        'learning_rate': best_lr,
        'image_size': best_size,
        'embedding_dim': best_embedding_dim,
        'loss': 'ArcFace',
        'arcface_margin': ARCFACE_MARGIN,
        'arcface_scale': ARCFACE_SCALE,
        'backbone': model_config['backbone_label'],
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
            'epoch_history': epoch_history,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'job_idx': args.idx,
            }
        }

    else:
        # ---- OPEN-SET: BA only, cosine threshold from hygiene ----
        cosine_threshold = best_config['cosine_threshold']
        if cosine_threshold is None:
            raise ValueError("No cosine threshold found in hygiene results for open-set task")

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

        # BA using hygiene-calibrated cosine threshold at the selected query quality threshold
        ba_metrics = compute_open_set_metrics_cosine(
            known_query_labels=query_labels,
            known_query_quality=query_quality,
            known_scores=scores_known,
            unknown_query_labels=rare_labels_arr,
            unknown_query_quality=rare_quality_vals,
            unknown_scores=scores_unknown,
            score_threshold=cosine_threshold,
            quality_thresholds=[query_quality_threshold],
            gallery_labels=gallery_labels
        )

        q_key = f"q>={query_quality_threshold}"
        ba_data = ba_metrics.get(q_key, {})
        ba = ba_data.get('balanced_accuracy', 0.0)
        kar = ba_data.get('known_accept_rate', 0.0)
        urr = ba_data.get('unknown_reject_rate', 0.0)
        n_k = ba_data.get('n_known_individuals', 0)
        n_u = ba_data.get('n_unknown_individuals', 0)

        print(f"\nTest Results (seed={seed}, task={task_name}):")
        print(f"  Cosine threshold (from hygiene): {cosine_threshold:.4f}")
        print(f"  {q_key}: BA={ba:.4f} (K={kar:.2f}[{n_k}], U={urr:.2f}[{n_u}])")

        dataset_info.update({
            'total_query_unknown': len(rare_indices) if rare_indices else 0,
            'n_unknown_individuals': len(set(rare_labels_str)) if rare_labels_str else 0,
            'unknown_source': 'hf_test_split',
        })

        result = {
            'config': {**result_config, 'cosine_threshold': cosine_threshold},
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
    sub_lr.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # opt_embedding_dim subcommand
    sub_emb = subparsers.add_parser("opt_embedding_dim", help="Embedding dimension sweep")
    add_common_args(sub_emb)
    sub_emb.add_argument("--values", type=int, nargs="+", required=True, help="Embedding dims to sweep")
    sub_emb.add_argument("--lr", type=float, required=True, help="Fixed learning rate")
    sub_emb.add_argument("--image-size", type=int, required=True, help="Fixed image size")
    sub_emb.add_argument("--epochs", type=int, default=50)
    sub_emb.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # test_eval subcommand
    sub_test = subparsers.add_parser("test_eval", help="Final test evaluation")
    add_common_args(sub_test)
    sub_test.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])

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
        seeds = args.seeds
        task_names = list(TEST_TASKS.keys())

        print("=" * 80)
        print(f"Final Test Evaluation - {config['backbone_label']} Re-ID - Job {args.idx}")
        print(f"  {len(seeds)} seeds x {len(task_names)} tasks = {len(seeds) * len(task_names)} runs")
        print(f"  Tasks: {', '.join(task_names)}")
        print("=" * 80)

        if args.idx >= len(seeds):
            print(f"Job {args.idx} has no work (only {len(seeds)} seeds)")
            sys.exit(0)

        seed = seeds[args.idx]
        print(f"\nSeed: {seed}")

        for task_name in task_names:
            try:
                run_final_test(model_name, args, seed, task_name=task_name)
            except Exception as e:
                print(f"Error ({task_name}): {e}")
                import traceback
                traceback.print_exc()

        print(f"\nJob {args.idx} (test_eval) completed!")

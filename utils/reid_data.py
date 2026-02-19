"""
Dataset classes, loading, and splitting for reid experiments.
"""

import os
import json
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset

from utils.dataset import set_all_seeds
from utils.reid_config import TARGET_SAMPLES_PER_INDIVIDUAL, MIN_P

import datasets
datasets.config.NUM_PROC = 1


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

    # Filenames for temporal ordering within events
    filenames = None
    if 'filename' in dataset.column_names:
        filenames = np.array(dataset['filename'], dtype=object)

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
    if filenames is not None:
        cache['filenames'] = filenames

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


def get_event_index_map(metadata_cache, indices):
    """Group indices by YMDH event."""
    ymdh = metadata_cache['ymdh']
    event_map = defaultdict(list)
    for idx in indices:
        event_map[int(ymdh[idx])].append(idx)
    return dict(event_map)


def subsample_to_match_filtered(metadata_cache, all_indices, filtered_indices):
    """
    Deterministic event-matched subsampling by temporal spacing.

    For each YMDH event, sort all images by filename (which encodes capture
    order), then pick n_target images evenly spaced from first to last using
    linspace indices. This is deterministic and preserves temporal coverage
    across each burst.
    """
    filenames = metadata_cache['filenames']

    all_event_map = get_event_index_map(metadata_cache, all_indices)
    filtered_event_map = get_event_index_map(metadata_cache, filtered_indices)

    sampled = []
    for event, filtered_event_indices in filtered_event_map.items():
        n_target = len(filtered_event_indices)
        available = all_event_map.get(event, [])
        if not available:
            continue

        # Sort by filename to get capture order
        available_sorted = sorted(available, key=lambda idx: filenames[idx])

        n_available = len(available_sorted)
        if n_target >= n_available:
            sampled.extend(available_sorted)
        else:
            # Evenly spaced indices from first to last
            pick_indices = np.round(np.linspace(0, n_available - 1, n_target)).astype(int)
            sampled.extend(available_sorted[i] for i in pick_indices)

    return sampled


def get_qualified_individuals(config, min_p=MIN_P):
    """Get qualified individuals from config (preprocessing already enforces criteria)."""
    qualified_individuals = config.get('qualified_individuals', [])

    if len(qualified_individuals) < min_p:
        print(f"Not enough individuals ({len(qualified_individuals)}) for PK sampling (need {min_p})")
        return None

    return qualified_individuals

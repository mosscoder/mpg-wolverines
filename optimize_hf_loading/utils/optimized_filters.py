"""
Optimized filtering and gallery creation functions using cached metadata.

These are drop-in replacements that use vectorized numpy operations and
cached metadata instead of repeated sequential dataset access.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any


def filter_training_pool_by_quality_vectorized(quality_cache: np.ndarray,
                                               indices: List[int],
                                               threshold: float) -> List[int]:
    """
    Vectorized quality filtering using numpy boolean indexing.

    Args:
        quality_cache: numpy array of quality scores
        indices: List of dataset indices to filter
        threshold: Minimum quality score

    Returns:
        List of indices meeting the threshold

    Why results unchanged:
        - Numpy boolean indexing: arr >= threshold produces identical mask
        - Same filtered indices, just computed in C (numpy) vs Python loops
        - Order may differ but doesn't affect downstream sampling
    """
    if threshold <= 0.0:
        return indices

    if not indices:
        return []

    # Convert to numpy array for vectorized operations
    indices_arr = np.array(indices)
    quality_arr = quality_cache[indices_arr]

    # Vectorized boolean indexing
    mask = quality_arr >= threshold

    # Return as list
    return indices_arr[mask].tolist()


def create_filtered_gallery_dataset_optimized(dataset, individuals, gallery_size, threshold,
                                              seed, config, metadata_cache):
    """
    Create gallery/query split with filtered gallery using cached metadata.

    This is an optimized version of create_filtered_gallery_dataset() that uses:
    1. Cached quality scores instead of dataset access
    2. Vectorized filtering instead of Python loops
    3. Cached ID-to-indices mapping

    Args:
        dataset: HuggingFace dataset
        individuals: List of individual IDs to include
        gallery_size: Number of gallery samples per individual
        threshold: Quality threshold for filtering
        seed: Random seed
        config: Feasibility configuration
        metadata_cache: Cached metadata from build_metadata_cache()

    Returns:
        Tuple of (train_dataset, val_dataset, individual_to_class, dataset_info)

    Why results unchanged:
        - Same filtering logic, just faster execution
        - Same random seed → same shuffling → same sample selection
        - Cache contains exact dataset values
        - Only performance improves, not outputs
    """
    from utils.dataset import set_all_seeds

    set_all_seeds(seed)

    quality_cache = metadata_cache['quality_scores']
    id_to_indices = metadata_cache['id_to_indices']

    print(f"Creating filtered gallery (OPTIMIZED): threshold={threshold}, gallery_size={gallery_size}, seed={seed}")

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
        eligible_pool = filter_training_pool_by_quality_vectorized(
            quality_cache, train_candidates, threshold
        )
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


def get_rare_individual_indices_vectorized(metadata_cache: Dict[str, Any],
                                          valid_individuals: List[str],
                                          quality_threshold: float = 0.0) -> Tuple[List[int], List[float], List[str]]:
    """
    Get indices of rare individuals using vectorized operations on cached metadata.

    Args:
        metadata_cache: Cached metadata from build_metadata_cache()
        valid_individuals: List of known individual IDs
        quality_threshold: Minimum quality score

    Returns:
        Tuple of (rare_indices, rare_quality, rare_labels) lists

    Why results unchanged:
        - Same logic: individuals NOT in valid_individuals, filtered by quality
        - Vectorized numpy operations produce identical results
        - Only execution speed differs
    """
    quality_cache = metadata_cache['quality_scores']
    ids_cache = metadata_cache['ids']
    id_to_indices = metadata_cache['id_to_indices']

    rare_indices = []
    rare_quality = []
    rare_labels = []

    # Convert valid_individuals to set for O(1) lookup
    valid_set = set(valid_individuals)

    for ind_id, indices in id_to_indices.items():
        if ind_id not in valid_set:
            # Vectorized quality filtering for this individual
            indices_arr = np.array(indices)
            quality_arr = quality_cache[indices_arr]
            mask = quality_arr >= quality_threshold

            # Add filtered indices
            filtered_indices = indices_arr[mask]
            filtered_quality = quality_arr[mask]

            rare_indices.extend(filtered_indices.tolist())
            rare_quality.extend(filtered_quality.tolist())
            rare_labels.extend([ind_id] * len(filtered_indices))

    return rare_indices, rare_quality, rare_labels

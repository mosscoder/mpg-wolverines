"""
Metadata caching utilities for HuggingFace datasets.

Provides functions to extract and cache metadata (quality scores, IDs, mappings)
from HuggingFace datasets to avoid repeated sequential access.
"""

import numpy as np
from typing import Dict, List, Any


def build_metadata_cache(dataset) -> Dict[str, Any]:
    """
    Extract all metadata from dataset in a single pass.

    This replaces repeated sequential dataset access with a one-time scan,
    storing metadata in numpy arrays for fast vectorized operations.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Dictionary containing:
        - 'quality_scores': np.array of pelage_score values (float32)
        - 'ids': np.array of individual IDs (object/str)
        - 'id_to_indices': dict mapping ID -> list of indices

    Why results unchanged:
        - Caches exact values from dataset (read-only)
        - Same data, just faster access: dataset[i]['key'] vs cache[i]
        - No transformation applied, pure extraction
    """
    n = len(dataset)

    # Pre-allocate arrays
    quality_scores = np.zeros(n, dtype=np.float32)
    ids = []

    print(f"Building metadata cache for {n} samples...")

    # Single sequential pass through dataset
    for idx in range(n):
        sample = dataset[idx]
        quality_scores[idx] = sample['pelage_score']
        ids.append(sample['id'])

    # Convert IDs to numpy array for consistency
    ids = np.array(ids, dtype=object)

    # Build ID-to-indices mapping from array (much faster than dataset iteration)
    id_to_indices = _build_id_to_indices_from_array(ids)

    print(f"  Cached {n} quality scores")
    print(f"  Cached {n} IDs ({len(id_to_indices)} unique individuals)")

    return {
        'quality_scores': quality_scores,
        'ids': ids,
        'id_to_indices': id_to_indices,
        'dataset_size': n
    }


def _build_id_to_indices_from_array(ids: np.ndarray) -> Dict[str, List[int]]:
    """
    Build ID-to-indices mapping from numpy array.

    Much faster than iterating over dataset because we're working with
    in-memory numpy array instead of HuggingFace dataset access.

    Args:
        ids: numpy array of individual IDs

    Returns:
        Dictionary mapping individual ID -> list of dataset indices
    """
    id_to_indices = {}

    for idx, ind_id in enumerate(ids):
        if ind_id not in id_to_indices:
            id_to_indices[ind_id] = []
        id_to_indices[ind_id].append(idx)

    return id_to_indices


def filter_training_pool_by_quality_cached(quality_cache: np.ndarray,
                                          indices: List[int],
                                          threshold: float) -> List[int]:
    """
    Filter indices by quality threshold using cached quality scores.

    This is a drop-in replacement for filter_training_pool_by_quality()
    that uses cached quality scores instead of dataset access.

    Args:
        quality_cache: numpy array of quality scores (from build_metadata_cache)
        indices: List of dataset indices to filter
        threshold: Minimum quality score to include

    Returns:
        List of indices meeting the threshold

    Why results unchanged:
        - Same filtering logic: score >= threshold
        - Cache contains exact dataset values
        - Only difference is numpy lookup vs dataset lookup
    """
    if threshold <= 0.0:
        return indices  # No filtering needed

    filtered = []
    for idx in indices:
        if quality_cache[idx] >= threshold:
            filtered.append(idx)

    return filtered


def get_rare_individual_indices_cached(quality_cache: np.ndarray,
                                      ids_cache: np.ndarray,
                                      valid_individuals: List[str],
                                      id_to_indices: Dict[str, List[int]],
                                      quality_threshold: float = 0.0):
    """
    Get indices of rare individuals using cached metadata.

    This is a drop-in replacement for get_rare_individual_indices()
    that uses cached metadata instead of dataset access.

    Args:
        quality_cache: numpy array of quality scores
        ids_cache: numpy array of individual IDs
        valid_individuals: List of known individual IDs
        id_to_indices: ID-to-indices mapping
        quality_threshold: Minimum quality score

    Returns:
        Tuple of (rare_indices, rare_quality, rare_labels) lists

    Why results unchanged:
        - Same logic: individuals NOT in valid_individuals, filtered by quality
        - Cache contains exact dataset values
        - Only difference is array lookup vs dataset access
    """
    rare_indices = []
    rare_quality = []
    rare_labels = []

    for ind_id, indices in id_to_indices.items():
        if ind_id not in valid_individuals:
            for idx in indices:
                q = quality_cache[idx]
                if q >= quality_threshold:
                    rare_indices.append(idx)
                    rare_quality.append(q)
                    rare_labels.append(ind_id)

    return rare_indices, rare_quality, rare_labels

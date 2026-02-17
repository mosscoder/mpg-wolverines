"""
Optimized filtering and gallery creation functions using cached metadata.

These are drop-in replacements that use vectorized numpy operations and
cached metadata instead of repeated sequential dataset access.
"""

import numpy as np
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


def get_rare_individual_indices_vectorized(metadata_cache: Dict[str, Any],
                                          qualified_individuals: List[str],
                                          quality_threshold: float = 0.0,
                                          promoted_individuals: List[str] = None,
                                          excluded_individuals: List[str] = None) -> Tuple[List[int], List[float], List[str]]:
    """
    Get indices of rare/novel individuals using vectorized operations on cached metadata.

    Includes:
    - Individuals NOT in qualified_individuals (existing rare)
    - promoted_individuals (demoted from closed-set but pass diversity)

    Excludes:
    - qualified_individuals (used for closed-set)
    - excluded_individuals (fail diversity requirements)

    Args:
        metadata_cache: Cached metadata from build_metadata_cache()
        qualified_individuals: List of known individual IDs (closed-set)
        quality_threshold: Minimum quality score
        promoted_individuals: List of individuals promoted to rare/novel set
        excluded_individuals: List of individuals to exclude entirely

    Returns:
        Tuple of (rare_indices, rare_quality, rare_labels) lists

    Why results unchanged:
        - Same logic: individuals NOT in qualified_individuals, filtered by quality
        - Vectorized numpy operations produce identical results
        - Only execution speed differs
    """
    quality_cache = metadata_cache['quality_scores']
    ids_cache = metadata_cache['ids']
    id_to_indices = metadata_cache['id_to_indices']

    rare_indices = []
    rare_quality = []
    rare_labels = []

    # Convert to sets for O(1) lookup
    qualified_set = set(qualified_individuals)
    promoted_set = set(promoted_individuals or [])
    excluded_set = set(excluded_individuals or [])

    for ind_id, indices in id_to_indices.items():
        # Include if: (not qualified AND not excluded) OR promoted
        if (ind_id not in qualified_set and ind_id not in excluded_set) or ind_id in promoted_set:
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

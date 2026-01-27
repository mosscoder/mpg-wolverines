"""
Arrow-based metadata caching for HuggingFace datasets.

Uses columnar access via Arrow tables instead of row-by-row iteration
for significantly faster metadata extraction.
"""

import numpy as np
from typing import Dict, List, Any


def build_metadata_cache_arrow(dataset) -> Dict[str, Any]:
    """
    Extract all metadata using Arrow columnar access.

    HuggingFace datasets are backed by Apache Arrow tables, which support
    efficient columnar operations. Instead of iterating row-by-row, we
    access entire columns at once.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Dictionary containing:
        - 'quality_scores': np.array of pelage_score values (float32)
        - 'ids': np.array of individual IDs (object/str)
        - 'id_to_indices': dict mapping ID -> list of indices

    Why results unchanged:
        - Same exact data, just accessed via columnar API
        - Arrow table is the underlying storage - we're just reading it differently
        - No transformation applied, pure extraction

    Performance:
        - Row-by-row: O(n) random access operations
        - Columnar: O(1) column access + O(n) array conversion
        - Expected speedup: 10-100x for large datasets
    """
    n = len(dataset)

    print(f"Building metadata cache (Arrow columnar access) for {n} samples...")

    # Columnar access - single operation instead of n lookups!
    # This accesses the underlying Arrow table directly
    quality_scores = np.array(dataset['pelage_score'], dtype=np.float32)
    ids = np.array(dataset['id'], dtype=object)

    # Build ID-to-indices mapping from array (same as before)
    id_to_indices = _build_id_to_indices_from_array(ids)

    print(f"  Cached {n} quality scores (columnar)")
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

    This can potentially be optimized further using numpy operations,
    but for ~10 unique individuals it's already very fast.

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


def build_id_to_indices_vectorized(ids: np.ndarray) -> Dict[str, List[int]]:
    """
    Build ID-to-indices mapping using numpy vectorized operations.

    Alternative implementation that uses numpy's unique and where operations
    for potentially better performance on large datasets.

    Args:
        ids: numpy array of individual IDs

    Returns:
        Dictionary mapping individual ID -> list of dataset indices
    """
    id_to_indices = {}

    # Get unique IDs
    unique_ids = np.unique(ids)

    # For each unique ID, find all indices where it appears
    for ind_id in unique_ids:
        indices = np.where(ids == ind_id)[0].tolist()
        id_to_indices[ind_id] = indices

    return id_to_indices

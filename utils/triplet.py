"""
Triplet loss utilities for wolverine re-identification experiments.
Includes quality-weighted triplet loss, PK batch sampling, and embedding model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import AutoModel
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class EmbeddingHead(nn.Module):
    """Linear projection head that outputs L2-normalized embeddings."""

    def __init__(self, input_dim: int = 768, embedding_dim: int = 128):
        """
        Args:
            input_dim: Input feature dimension from backbone (768 for DINOv3-ViT-B)
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            L2-normalized embeddings of shape (batch_size, embedding_dim)
        """
        return F.normalize(self.fc(x), p=2, dim=1)


class QualityWeightedTripletLoss(nn.Module):
    """
    Triplet loss with confidence weighting (down-weighting low-quality pairs).

    L = weight * max(0, d(a,p) - d(a,n) + margin)

    where weight = min_weight + (1 - min_weight) * q_anchor * q_positive

    This down-weights noisy gradients from low-quality pairs rather than trying
    to up-weight good pairs (which often have zero loss anyway due to ReLU).

    - High quality pair (q=1.0): weight = 1.0 (full contribution)
    - Low quality pair (q=0.0): weight = min_weight (suppressed)
    """

    def __init__(self, margin: float = 0.3, min_weight: float = 1.0):
        """
        Args:
            margin: Triplet loss margin
            min_weight: Minimum weight for lowest quality pairs.
                        1.0 = no weighting (baseline), 0.1 = 10x suppression.
        """
        super().__init__()
        self.margin = margin
        self.min_weight = min_weight

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor,
                q_anchor: torch.Tensor,
                q_positive: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
            q_anchor: Quality scores for anchors (batch_size,) in [0, 1]
            q_positive: Quality scores for positives (batch_size,) in [0, 1]

        Returns:
            Scalar loss value
        """
        # Compute distances
        d_ap = torch.norm(anchor - positive, p=2, dim=1)
        d_an = torch.norm(anchor - negative, p=2, dim=1)

        # Standard triplet loss
        base_loss = torch.clamp(d_ap - d_an + self.margin, min=0)

        # Confidence weighting: down-weight low-quality pairs
        # High quality (q=1): weight = 1.0
        # Low quality (q=0): weight = min_weight
        quality_product = q_anchor * q_positive
        weight = self.min_weight + (1.0 - self.min_weight) * quality_product

        return (weight * base_loss).mean()


class PKBatchSampler(Sampler):
    """
    Sampler for P-K batch sampling in metric learning.
    Each batch contains P identities with K samples each.
    """

    def __init__(self,
                 labels: List[int],
                 p: int = 5,
                 k: int = 8,
                 drop_last: bool = True):
        """
        Args:
            labels: List of class labels for each sample
            p: Number of identities per batch
            k: Number of samples per identity
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__(None)
        self.labels = labels
        self.p = p
        self.k = k
        self.drop_last = drop_last

        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Filter to labels with at least k samples
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= k
        ]

        if len(self.valid_labels) < p:
            raise ValueError(
                f"Not enough identities with >= {k} samples. "
                f"Need {p}, have {len(self.valid_labels)}"
            )

        self.batch_size = p * k

    def __iter__(self):
        """Generate batches of P identities × K samples."""
        # Shuffle valid labels
        labels = self.valid_labels.copy()
        random.shuffle(labels)

        # Shuffle indices within each label
        label_indices = {}
        for label in labels:
            indices = self.label_to_indices[label].copy()
            random.shuffle(indices)
            label_indices[label] = indices

        # Generate batches
        batch = []
        label_idx = 0

        while label_idx + self.p <= len(labels):
            # Select P labels for this batch
            batch_labels = labels[label_idx:label_idx + self.p]
            label_idx += self.p

            # Get K samples from each label
            for label in batch_labels:
                indices = label_indices[label]

                # If not enough samples, reshuffle and restart
                if len(indices) < self.k:
                    indices = self.label_to_indices[label].copy()
                    random.shuffle(indices)
                    label_indices[label] = indices

                # Take K samples
                batch.extend(indices[:self.k])
                label_indices[label] = indices[self.k:]

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle remaining labels by cycling back
        if not self.drop_last and batch:
            yield batch

    def __len__(self):
        """Number of batches per epoch."""
        num_batches = len(self.valid_labels) // self.p
        return num_batches


def create_embedding_model(model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                           embedding_dim: int = 128,
                           device: str = "cuda") -> nn.Module:
    """
    Create DINOv3 backbone with embedding head for re-identification.

    Args:
        model_name: HuggingFace model name for backbone
        embedding_dim: Output embedding dimension
        device: Device to place model on

    Returns:
        Model with frozen backbone and trainable embedding head
    """
    # Load backbone
    backbone = AutoModel.from_pretrained(model_name)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Get hidden size
    hidden_size = backbone.config.hidden_size

    # Create embedding head
    embedding_head = EmbeddingHead(input_dim=hidden_size, embedding_dim=embedding_dim)

    class EmbeddingModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                outputs = self.backbone(x)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            return self.head(features)

        def get_trainable_parameters(self):
            return self.head.parameters()

    model = EmbeddingModel(backbone, embedding_head)

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

    return model.to(device)


def mine_random_triplets(embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         quality_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine random triplets from a batch.

    For each sample as anchor:
    - Positive: Random sample with same label (excluding self)
    - Negative: Random sample with different label

    Args:
        embeddings: Batch embeddings (batch_size, embedding_dim)
        labels: Batch labels (batch_size,)
        quality_scores: Batch quality scores (batch_size,)

    Returns:
        Tuple of (anchor_emb, positive_emb, negative_emb, anchor_quality, positive_quality)
    """
    batch_size = embeddings.size(0)
    device = embeddings.device

    anchors = []
    positives = []
    negatives = []
    anchor_qualities = []
    positive_qualities = []

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx in range(batch_size):
        label_to_indices[labels[idx].item()].append(idx)

    unique_labels = list(label_to_indices.keys())

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx].item()
        same_label_indices = [i for i in label_to_indices[anchor_label] if i != anchor_idx]
        diff_label_indices = [i for label in unique_labels if label != anchor_label
                              for i in label_to_indices[label]]

        if not same_label_indices or not diff_label_indices:
            continue

        # Random positive and negative
        pos_idx = random.choice(same_label_indices)
        neg_idx = random.choice(diff_label_indices)

        anchors.append(anchor_idx)
        positives.append(pos_idx)
        negatives.append(neg_idx)
        anchor_qualities.append(quality_scores[anchor_idx])
        positive_qualities.append(quality_scores[pos_idx])

    if not anchors:
        # Return empty tensors if no valid triplets
        return (torch.empty(0, embeddings.size(1), device=device),
                torch.empty(0, embeddings.size(1), device=device),
                torch.empty(0, embeddings.size(1), device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device))

    anchor_emb = embeddings[anchors]
    positive_emb = embeddings[positives]
    negative_emb = embeddings[negatives]
    anchor_quality = torch.stack(anchor_qualities)
    positive_quality = torch.stack(positive_qualities)

    return anchor_emb, positive_emb, negative_emb, anchor_quality, positive_quality


def compute_recall_at_k(query_embeddings: torch.Tensor,
                        gallery_embeddings: torch.Tensor,
                        query_labels: torch.Tensor,
                        gallery_labels: torch.Tensor,
                        k: int = 1) -> float:
    """
    Compute macro-averaged Rank-K (mean of per-individual Rank-K accuracy).
    Each individual contributes equally regardless of sample count.

    Rank-1: fraction of queries where top-1 match is correct

    Args:
        query_embeddings: Query embeddings (n_query, embedding_dim)
        gallery_embeddings: Gallery embeddings (n_gallery, embedding_dim)
        query_labels: Query labels (n_query,)
        gallery_labels: Gallery labels (n_gallery,)
        k: Number of nearest neighbors to consider

    Returns:
        Macro-averaged Rank-K score in [0, 1]
    """
    if query_embeddings.size(0) == 0 or gallery_embeddings.size(0) == 0:
        return 0.0

    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)
    _, top_k_indices = distances.topk(k, dim=1, largest=False)

    # Group by individual
    individual_correct = defaultdict(int)
    individual_total = defaultdict(int)

    for i in range(query_embeddings.size(0)):
        query_label = query_labels[i].item()
        individual_total[query_label] += 1

        top_k_labels = gallery_labels[top_k_indices[i]].tolist()
        if query_label in top_k_labels:
            individual_correct[query_label] += 1

    # Per-individual Rank-K, then mean
    per_individual_rank_k = [
        individual_correct[label] / individual_total[label]
        for label in individual_total
    ]

    return np.mean(per_individual_rank_k)


def compute_recall_at_k_by_quality_bin(query_embeddings: torch.Tensor,
                                       gallery_embeddings: torch.Tensor,
                                       query_labels: torch.Tensor,
                                       gallery_labels: torch.Tensor,
                                       query_quality_scores: np.ndarray,
                                       k: int = 1) -> Dict[str, Dict[str, float]]:
    """
    Compute macro-averaged Rank-K by query quality bin.
    Within each bin, compute per-individual Rank-K then take the mean.

    Args:
        query_embeddings: Query embeddings (n_query, embedding_dim)
        gallery_embeddings: Gallery embeddings (n_gallery, embedding_dim)
        query_labels: Query labels (n_query,)
        gallery_labels: Gallery labels (n_gallery,)
        query_quality_scores: Quality scores for queries (n_query,)
        k: Number of nearest neighbors

    Returns:
        Dict mapping bin name to {'recall_at_1': float, 'recall_at_5': float, 'count': int}
    """
    if query_embeddings.size(0) == 0 or gallery_embeddings.size(0) == 0:
        return {}

    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)

    # Get top-K nearest indices (get enough for recall@5)
    k_max = min(5, gallery_embeddings.size(0))
    _, top_k_indices = distances.topk(k_max, dim=1, largest=False)

    # Define quality bins
    quality_bins = [
        (0.75, 1.0, '[0.75,1.0]'),
        (0.50, 0.75, '[0.5,0.75)'),
        (0.25, 0.50, '[0.25,0.5)'),
        (0.00, 0.25, '[0,0.25)')
    ]

    bin_metrics = {}

    for min_q, max_q, bin_name in quality_bins:
        if bin_name == '[0.75,1.0]':
            mask = (query_quality_scores >= min_q) & (query_quality_scores <= max_q)
        else:
            mask = (query_quality_scores >= min_q) & (query_quality_scores < max_q)

        bin_indices = np.where(mask)[0]

        if len(bin_indices) == 0:
            bin_metrics[bin_name] = {'recall_at_1': 0.0, 'recall_at_5': 0.0, 'count': 0}
            continue

        # Group by individual within this bin
        individual_correct_at_1 = defaultdict(int)
        individual_correct_at_5 = defaultdict(int)
        individual_total = defaultdict(int)

        for idx in bin_indices:
            query_label = query_labels[idx].item()
            individual_total[query_label] += 1

            # Recall@1
            if gallery_labels[top_k_indices[idx, 0]].item() == query_label:
                individual_correct_at_1[query_label] += 1

            # Recall@5
            top_5_labels = gallery_labels[top_k_indices[idx, :k_max]].tolist()
            if query_label in top_5_labels:
                individual_correct_at_5[query_label] += 1

        # Per-individual Rank-K, then mean
        per_individual_rank_1 = [
            individual_correct_at_1[label] / individual_total[label]
            for label in individual_total
        ]
        per_individual_rank_5 = [
            individual_correct_at_5[label] / individual_total[label]
            for label in individual_total
        ]

        bin_metrics[bin_name] = {
            'recall_at_1': np.mean(per_individual_rank_1) if per_individual_rank_1 else 0.0,
            'recall_at_5': np.mean(per_individual_rank_5) if per_individual_rank_5 else 0.0,
            'count': len(bin_indices)
        }

    return bin_metrics


def compute_recall_by_query_gallery_quality(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_quality: np.ndarray,
    gallery_quality: np.ndarray,
    threshold: float = 0.5,
    k: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute Recall@K for 4 query×gallery quality combinations.

    Filters both queries and gallery by quality, then runs 4 separate retrievals:
    - HQ_HG: high-quality queries vs high-quality gallery
    - HQ_LG: high-quality queries vs low-quality gallery
    - LQ_HG: low-quality queries vs high-quality gallery
    - LQ_LG: low-quality queries vs low-quality gallery

    Args:
        query_embeddings: Query embeddings (n_query, embedding_dim)
        gallery_embeddings: Gallery embeddings (n_gallery, embedding_dim)
        query_labels: Query labels (n_query,)
        gallery_labels: Gallery labels (n_gallery,)
        query_quality: Quality scores for queries (n_query,)
        gallery_quality: Quality scores for gallery (n_gallery,)
        threshold: Quality threshold (>= is high, < is low)
        k: Number of nearest neighbors

    Returns:
        Dict mapping combination name to {'recall_at_1': float, 'count': int, 'gallery_size': int}
    """
    query_quality = np.array(query_quality)
    gallery_quality = np.array(gallery_quality)

    # Split indices by quality
    hq_query_mask = query_quality >= threshold
    lq_query_mask = ~hq_query_mask
    hg_gallery_mask = gallery_quality >= threshold
    lg_gallery_mask = ~hg_gallery_mask

    combinations = [
        ('HQ_HG', hq_query_mask, hg_gallery_mask),
        ('HQ_LG', hq_query_mask, lg_gallery_mask),
        ('LQ_HG', lq_query_mask, hg_gallery_mask),
        ('LQ_LG', lq_query_mask, lg_gallery_mask),
    ]

    results = {}
    for name, q_mask, g_mask in combinations:
        # Filter embeddings and labels
        q_emb = query_embeddings[q_mask]
        g_emb = gallery_embeddings[g_mask]

        # Handle tensor vs ndarray for labels
        if isinstance(query_labels, np.ndarray):
            q_lab = query_labels[q_mask]
        else:
            q_lab = query_labels[torch.tensor(q_mask)]

        if isinstance(gallery_labels, np.ndarray):
            g_lab = gallery_labels[g_mask]
        else:
            g_lab = gallery_labels[torch.tensor(g_mask)]

        if len(q_emb) == 0 or len(g_emb) == 0:
            results[name] = {'recall_at_1': 0.0, 'count': int(q_mask.sum()), 'gallery_size': int(g_mask.sum())}
            continue

        # Compute distances and get top-k
        distances = torch.cdist(q_emb, g_emb, p=2)
        _, top_idx = distances.topk(min(k, len(g_emb)), dim=1, largest=False)

        # Count correct matches
        correct = 0
        for i in range(len(q_lab)):
            q_label = q_lab[i].item() if hasattr(q_lab[i], 'item') else q_lab[i]
            matched_label = g_lab[top_idx[i, 0]].item() if hasattr(g_lab[top_idx[i, 0]], 'item') else g_lab[top_idx[i, 0]]
            if matched_label == q_label:
                correct += 1

        results[name] = {
            'recall_at_1': correct / len(q_lab),
            'count': len(q_lab),
            'gallery_size': len(g_lab)
        }

    return results


def compute_mean_average_precision(query_embeddings: torch.Tensor,
                                   gallery_embeddings: torch.Tensor,
                                   query_labels: torch.Tensor,
                                   gallery_labels: torch.Tensor) -> float:
    """
    Compute macro-averaged Mean Average Precision (mAP) for retrieval evaluation.
    Each individual contributes equally regardless of sample count.

    Args:
        query_embeddings: Query embeddings (n_query, embedding_dim)
        gallery_embeddings: Gallery embeddings (n_gallery, embedding_dim)
        query_labels: Query labels (n_query,)
        gallery_labels: Gallery labels (n_gallery,)

    Returns:
        Macro-averaged mAP score in [0, 1]
    """
    if query_embeddings.size(0) == 0 or gallery_embeddings.size(0) == 0:
        return 0.0

    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)

    # Sort gallery by distance for each query
    sorted_indices = distances.argsort(dim=1)

    # Group APs by individual
    individual_aps = defaultdict(list)

    for i in range(query_embeddings.size(0)):
        query_label = query_labels[i].item()
        sorted_gallery_labels = gallery_labels[sorted_indices[i]].tolist()

        # Count relevant items
        n_relevant = sum(1 for label in gallery_labels.tolist() if label == query_label)
        if n_relevant == 0:
            continue

        # Compute AP for this query
        relevant_count = 0
        precision_sum = 0.0
        for rank, label in enumerate(sorted_gallery_labels, 1):
            if label == query_label:
                relevant_count += 1
                precision_sum += relevant_count / rank

        ap = precision_sum / n_relevant
        individual_aps[query_label].append(ap)

    # Mean AP per individual, then mean across individuals
    per_individual_map = [
        np.mean(aps) for aps in individual_aps.values()
    ]

    return np.mean(per_individual_map) if per_individual_map else 0.0

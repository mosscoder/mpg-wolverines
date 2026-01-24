"""
Triplet loss utilities for wolverine re-identification experiments.
Includes quality-weighted triplet loss, PK batch sampling, and embedding model.
"""

import math
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


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for classification-based metric learning.
    Official MegaDescriptor training uses margin=0.5, scale=64.
    """

    def __init__(self, num_classes: int, embedding_size: int,
                 margin: float = 0.5, scale: float = 64.0):
        """
        Args:
            num_classes: Number of identity classes
            embedding_size: Dimension of input embeddings
            margin: Angular margin penalty (default 0.5 radians)
            scale: Scaling factor for logits (default 64)
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: L2-normalized embeddings (batch_size, embedding_size)
            labels: Ground truth class labels (batch_size,)

        Returns:
            Scalar loss value
        """
        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weights)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))

        # ArcFace formula: cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return F.cross_entropy(output, labels)


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


def create_megadescriptor_embedding_model(
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384",
    embedding_dim: int = 128,
    device: str = "cuda"
) -> nn.Module:
    """
    Create MegaDescriptor backbone with embedding head for re-identification.

    Args:
        model_name: timm model name for MegaDescriptor
        embedding_dim: Output embedding dimension
        device: Device to place model on

    Returns:
        Model with frozen backbone and trainable embedding head
    """
    import timm

    # Load backbone
    backbone = timm.create_model(model_name, pretrained=True)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Get hidden size via dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 384)
        hidden_size = backbone(dummy).shape[1]

    # Create embedding head
    embedding_head = EmbeddingHead(input_dim=hidden_size, embedding_dim=embedding_dim)

    class MegaDescriptorEmbeddingModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                features = self.backbone(x)  # Direct output, no CLS token extraction
            return self.head(features)

        def get_trainable_parameters(self):
            return self.head.parameters()

    model = MegaDescriptorEmbeddingModel(backbone, embedding_head)

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


def create_megadescriptor_arcface_model(
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384",
    embedding_dim: int = 128,
    device: str = "cuda"
) -> Tuple[nn.Module, int]:
    """
    Create MegaDescriptor backbone with trainable projection head for ArcFace training.

    Architecture: Frozen MegaDescriptor -> ~1536-d -> Linear -> 128-d

    Args:
        model_name: timm model name for MegaDescriptor
        embedding_dim: Output embedding dimension (default 128)
        device: Device to place model on

    Returns:
        Tuple of (model, embedding_dim): Model and its output dimension
    """
    import timm

    backbone = timm.create_model(model_name, pretrained=True)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Get backbone output dimension via dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 384)
        backbone_dim = backbone(dummy).shape[1]  # ~1536

    # Trainable projection head
    head = nn.Linear(backbone_dim, embedding_dim)

    class MegaDescriptorWithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                features = self.backbone(x)
            return self.head(features)

        def get_trainable_parameters(self):
            return self.head.parameters()

    model = MegaDescriptorWithHead(backbone, head)

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


def find_optimal_distance_threshold(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, Dict]:
    """
    Find distance threshold that best separates correct vs incorrect matches.

    For each query, finds the nearest gallery sample. A match is "correct" if
    the nearest gallery sample has the same ID as the query. The threshold
    determines whether to accept (distance < threshold) or reject the match.

    Args:
        query_emb: Query embeddings (n_query, embedding_dim)
        gallery_emb: Gallery embeddings (n_gallery, embedding_dim)
        query_labels: Query labels (n_query,)
        gallery_labels: Gallery labels (n_gallery,)
        thresholds: Array of thresholds to sweep (default: 0.1 to 2.0 step 0.05)

    Returns:
        optimal_threshold: Distance threshold maximizing F1
        metrics: Dict with f1, precision, recall at optimal threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 2.0, 0.05)

    if query_emb.size(0) == 0 or gallery_emb.size(0) == 0:
        return thresholds[0], {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Normalize embeddings before computing distances
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)

    # Compute distances and find nearest gallery sample for each query
    distances = torch.cdist(query_emb, gallery_emb, p=2)
    min_distances, nearest_idx = distances.min(dim=1)

    # Get predicted labels (label of nearest gallery sample)
    predicted_labels = gallery_labels[nearest_idx]

    # Determine which matches are correct (same ID)
    is_correct = (predicted_labels == query_labels)

    # Convert to numpy for threshold sweep
    min_distances_np = min_distances.cpu().numpy()
    is_correct_np = is_correct.cpu().numpy()

    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    for thresh in thresholds:
        # Accept if distance < threshold
        accepted = min_distances_np < thresh

        # True positives: accepted AND correct
        tp = np.sum(accepted & is_correct_np)
        # False positives: accepted AND incorrect
        fp = np.sum(accepted & ~is_correct_np)
        # False negatives: rejected AND correct
        fn = np.sum(~accepted & is_correct_np)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            }

    return float(best_threshold), best_metrics


def compute_correct_flag_rate(
    unknown_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    threshold: float
) -> Dict:
    """
    Compute rate at which unknowns are correctly flagged as unknown.

    An unknown is correctly flagged when its distance to the nearest
    gallery sample is >= threshold (i.e., rejected as not matching any
    known individual).

    Args:
        unknown_emb: Embeddings of unknown individuals (n_unknown, embedding_dim)
        gallery_emb: Gallery embeddings (n_gallery, embedding_dim)
        threshold: Distance threshold (reject if distance >= threshold)

    Returns:
        Dict with correct_flag_rate and count
    """
    if unknown_emb.size(0) == 0 or gallery_emb.size(0) == 0:
        return {'correct_flag_rate': 0.0, 'count': 0}

    # Normalize embeddings before computing distances
    unknown_emb = F.normalize(unknown_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)

    # Compute distances to nearest gallery sample
    distances = torch.cdist(unknown_emb, gallery_emb, p=2)
    min_distances = distances.min(dim=1).values

    # Correctly flagged if distance >= threshold (rejected as unknown)
    correctly_flagged = (min_distances >= threshold).sum().item()
    total = len(unknown_emb)

    return {
        'correct_flag_rate': correctly_flagged / total if total > 0 else 0.0,
        'count': total
    }


def evaluate_open_set_by_quality(
    unknown_emb: torch.Tensor,
    unknown_quality: np.ndarray,
    gallery_emb: torch.Tensor,
    threshold: float,
    quality_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Compute correct flag rate for unknowns at each quality level.

    For each quality threshold q, filters unknowns with quality >= q and
    computes the correct flag rate (proportion correctly rejected).

    Args:
        unknown_emb: Embeddings of unknown individuals (n_unknown, embedding_dim)
        unknown_quality: Quality scores for unknowns (n_unknown,)
        gallery_emb: Gallery embeddings (n_gallery, embedding_dim)
        threshold: Distance threshold (reject if distance >= threshold)
        quality_thresholds: List of quality thresholds to evaluate

    Returns:
        Dict mapping "q>=X" to {correct_flag_rate, count}
    """
    if quality_thresholds is None:
        quality_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if unknown_emb.size(0) == 0 or gallery_emb.size(0) == 0:
        return {f"q>={q}": {"correct_flag_rate": 0.0, "count": 0} for q in quality_thresholds}

    # Normalize embeddings before computing distances
    unknown_emb = F.normalize(unknown_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)

    # Compute distances to nearest gallery sample
    distances = torch.cdist(unknown_emb, gallery_emb, p=2)
    min_distances = distances.min(dim=1).values.cpu().numpy()

    unknown_quality = np.array(unknown_quality)

    results = {}
    for q_thresh in quality_thresholds:
        mask = unknown_quality >= q_thresh
        count = int(mask.sum())

        if count == 0:
            results[f"q>={q_thresh}"] = {"correct_flag_rate": 0.0, "count": 0}
            continue

        filtered_distances = min_distances[mask]
        correctly_flagged = int((filtered_distances >= threshold).sum())

        results[f"q>={q_thresh}"] = {
            "correct_flag_rate": correctly_flagged / count,
            "count": count
        }

    return results


def calibrate_threshold_loo(
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict]:
    """
    Leave-one-individual-out threshold calibration within training gallery.

    For each individual, holds them out and calibrates threshold on remaining
    individuals. Returns mean threshold across all LOO folds.

    Args:
        gallery_emb: Normalized gallery embeddings (n_gallery, embed_dim)
        gallery_labels: Individual IDs (n_gallery,)
        thresholds: Threshold sweep range (default 0.1-2.0)

    Returns:
        threshold_mean: Mean threshold across LOO folds
        threshold_std: Std of thresholds (stability indicator)
        metrics: Dict with per-fold details
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 2.0, 0.05)

    if gallery_emb.size(0) == 0:
        return float(thresholds[len(thresholds) // 2]), 0.0, {'method': 'loo', 'n_folds': 0}

    # Normalize embeddings
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)

    # Get unique individuals
    if isinstance(gallery_labels, torch.Tensor):
        unique_labels = torch.unique(gallery_labels).tolist()
    else:
        unique_labels = list(set(gallery_labels))

    if len(unique_labels) < 2:
        # Not enough individuals for LOO
        return float(thresholds[len(thresholds) // 2]), 0.0, {'method': 'loo', 'n_folds': 0}

    fold_thresholds = []
    fold_details = []

    for held_out_label in unique_labels:
        # Create mask for samples NOT from held-out individual
        if isinstance(gallery_labels, torch.Tensor):
            mask = gallery_labels != held_out_label
        else:
            mask = torch.tensor([l != held_out_label for l in gallery_labels])

        remaining_emb = gallery_emb[mask]
        remaining_labels = gallery_labels[mask] if isinstance(gallery_labels, torch.Tensor) else torch.tensor(gallery_labels)[mask]

        if remaining_emb.size(0) < 2:
            continue

        # Compute pairwise distances within remaining gallery
        distances = torch.cdist(remaining_emb, remaining_emb, p=2)

        # Create same-ID and different-ID masks (exclude diagonal)
        n = remaining_emb.size(0)
        labels_expanded = remaining_labels.unsqueeze(0).expand(n, n)
        same_id_mask = (labels_expanded == remaining_labels.unsqueeze(1)) & (~torch.eye(n, dtype=torch.bool, device=remaining_emb.device))
        diff_id_mask = labels_expanded != remaining_labels.unsqueeze(1)

        # Get same-ID and different-ID distances
        same_id_distances = distances[same_id_mask].cpu().numpy()
        diff_id_distances = distances[diff_id_mask].cpu().numpy()

        if len(same_id_distances) == 0 or len(diff_id_distances) == 0:
            continue

        # Find threshold that maximizes accuracy
        # TP: same_id distances < threshold
        # TN: diff_id distances >= threshold
        best_acc = 0.0
        best_thresh = thresholds[0]

        for thresh in thresholds:
            tp = np.sum(same_id_distances < thresh)
            tn = np.sum(diff_id_distances >= thresh)
            acc = (tp + tn) / (len(same_id_distances) + len(diff_id_distances))

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        fold_thresholds.append(best_thresh)
        fold_details.append({
            'held_out': held_out_label,
            'threshold': float(best_thresh),
            'accuracy': float(best_acc),
            'n_same_pairs': len(same_id_distances),
            'n_diff_pairs': len(diff_id_distances)
        })

    if not fold_thresholds:
        return float(thresholds[len(thresholds) // 2]), 0.0, {'method': 'loo', 'n_folds': 0}

    threshold_mean = float(np.mean(fold_thresholds))
    threshold_std = float(np.std(fold_thresholds, ddof=1)) if len(fold_thresholds) > 1 else 0.0

    metrics = {
        'method': 'loo',
        'threshold_mean': threshold_mean,
        'threshold_std': threshold_std,
        'n_folds': len(fold_thresholds),
        'fold_details': fold_details
    }

    return threshold_mean, threshold_std, metrics


def evaluate_open_set_balanced(
    known_query_emb: torch.Tensor,
    known_query_labels: torch.Tensor,
    known_query_quality: np.ndarray,
    unknown_query_emb: torch.Tensor,
    unknown_query_quality: np.ndarray,
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    distance_threshold: float,
    quality_thresholds: Optional[List[float]] = None,
    unknown_query_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate open-set performance with macro-averaged balanced accuracy at each quality threshold.

    Uses macro-averaging: compute per-individual rates, then average across individuals.
    This ensures each individual contributes equally regardless of sample count.

    For each quality threshold q, filters both known and unknown queries by q,
    then computes balanced accuracy = (macro_known_accept_rate + macro_unknown_reject_rate) / 2

    Args:
        known_query_emb: Embeddings of known validation individuals
        known_query_labels: Class labels of known queries (for verifying correct match)
        known_query_quality: Quality scores for known queries
        unknown_query_emb: Embeddings of unknown individuals
        unknown_query_quality: Quality scores for unknown individuals
        gallery_emb: Training gallery embeddings
        gallery_labels: Training gallery class labels
        distance_threshold: Distance threshold (accept if < threshold)
        quality_thresholds: List of quality thresholds to evaluate (default 0.0-0.5)
        unknown_query_labels: Individual IDs for unknown queries (for macro-averaging)

    Returns:
        Dict mapping "q>=X" to metrics dict with:
        - balanced_accuracy: (known_accept_rate + unknown_reject_rate) / 2 (macro-averaged)
        - known_accept_rate: Mean of per-individual acceptance rates
        - unknown_reject_rate: Mean of per-individual rejection rates
        - f1_score: F1 for known class
        - n_known_samples: Number of known query samples at this threshold
        - n_unknown_samples: Number of unknown query samples at this threshold
        - n_known_individuals: Number of known individuals at this threshold
        - n_unknown_individuals: Number of unknown individuals at this threshold
    """
    if quality_thresholds is None:
        quality_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    known_query_quality = np.array(known_query_quality)
    unknown_query_quality = np.array(unknown_query_quality)
    if unknown_query_labels is not None:
        unknown_query_labels = np.array(unknown_query_labels)

    # Normalize embeddings once
    if gallery_emb.size(0) > 0:
        gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
    if known_query_emb.size(0) > 0:
        known_query_emb = F.normalize(known_query_emb, p=2, dim=1)
    if unknown_query_emb.size(0) > 0:
        unknown_query_emb = F.normalize(unknown_query_emb, p=2, dim=1)

    # Precompute distances for all queries
    known_min_distances = None
    known_predicted_labels = None
    if known_query_emb.size(0) > 0 and gallery_emb.size(0) > 0:
        distances = torch.cdist(known_query_emb, gallery_emb, p=2)
        known_min_distances, nearest_idx = distances.min(dim=1)
        known_predicted_labels = gallery_labels[nearest_idx]

    unknown_min_distances = None
    if unknown_query_emb.size(0) > 0 and gallery_emb.size(0) > 0:
        distances = torch.cdist(unknown_query_emb, gallery_emb, p=2)
        unknown_min_distances = distances.min(dim=1).values

    # Convert known_query_labels to numpy for grouping
    if isinstance(known_query_labels, torch.Tensor):
        known_query_labels_np = known_query_labels.cpu().numpy()
    else:
        known_query_labels_np = np.array(known_query_labels)

    results = {}
    for q_thresh in quality_thresholds:
        # Filter by quality threshold
        known_mask = known_query_quality >= q_thresh
        unknown_mask = unknown_query_quality >= q_thresh

        n_known_samples = int(known_mask.sum())
        n_unknown_samples = int(unknown_mask.sum())

        result = {
            'balanced_accuracy': 0.5,
            'known_accept_rate': 0.0,
            'unknown_reject_rate': 1.0,
            'f1_score': 0.0,
            'n_known_samples': n_known_samples,
            'n_unknown_samples': n_unknown_samples,
            'n_known_individuals': 0,
            'n_unknown_individuals': 0
        }

        if gallery_emb.size(0) == 0:
            results[f"q>={q_thresh}"] = result
            continue

        # --- MACRO-AVERAGE for known individuals ---
        # Group by individual, compute per-individual acceptance rate
        per_individual_accept = []
        known_accepted_correct_total = 0  # For F1 calculation

        if n_known_samples > 0 and known_min_distances is not None:
            known_indices = np.where(known_mask)[0]
            unique_known_individuals = np.unique(known_query_labels_np[known_indices])

            for ind_label in unique_known_individuals:
                # Get indices for this individual within the quality-filtered set
                ind_mask = (known_query_labels_np == ind_label) & known_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # Count accepted AND correctly matched for this individual
                accepted_correct = 0
                for i in ind_indices:
                    if known_min_distances[i].item() < distance_threshold:
                        if known_predicted_labels[i].item() == known_query_labels_np[i]:
                            accepted_correct += 1

                per_individual_accept.append(accepted_correct / len(ind_indices))
                known_accepted_correct_total += accepted_correct

            result['n_known_individuals'] = len(per_individual_accept)

        macro_known_accept_rate = np.mean(per_individual_accept) if per_individual_accept else 0.0

        # --- MACRO-AVERAGE for unknown individuals ---
        # Group by individual, compute per-individual rejection rate
        per_individual_reject = []
        unknown_rejected_total = 0  # For F1 calculation

        if n_unknown_samples > 0 and unknown_min_distances is not None and unknown_query_labels is not None:
            unknown_indices = np.where(unknown_mask)[0]
            unique_unknown_individuals = np.unique(unknown_query_labels[unknown_indices])

            for ind_id in unique_unknown_individuals:
                # Get indices for this individual within the quality-filtered set
                ind_mask = (unknown_query_labels == ind_id) & unknown_mask
                ind_indices = np.where(ind_mask)[0]

                if len(ind_indices) == 0:
                    continue

                # Count rejected (distance >= threshold) for this individual
                rejected = 0
                for i in ind_indices:
                    if unknown_min_distances[i].item() >= distance_threshold:
                        rejected += 1

                per_individual_reject.append(rejected / len(ind_indices))
                unknown_rejected_total += rejected

            result['n_unknown_individuals'] = len(per_individual_reject)

        elif n_unknown_samples > 0 and unknown_min_distances is not None:
            # Fallback to micro-averaging if unknown_query_labels not provided
            filtered_distances = unknown_min_distances[torch.tensor(unknown_mask)]
            unknown_rejected_total = int((filtered_distances >= distance_threshold).sum().item())
            per_individual_reject = [unknown_rejected_total / n_unknown_samples] if n_unknown_samples > 0 else []
            result['n_unknown_individuals'] = 1  # Treat as single group

        macro_unknown_reject_rate = np.mean(per_individual_reject) if per_individual_reject else 1.0

        # Compute balanced accuracy from macro-averaged rates
        balanced_accuracy = (macro_known_accept_rate + macro_unknown_reject_rate) / 2

        # Confusion matrix for F1 (using sample counts for F1 consistency)
        tp = known_accepted_correct_total
        fn = n_known_samples - known_accepted_correct_total
        tn = unknown_rejected_total
        fp = n_unknown_samples - unknown_rejected_total

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result['balanced_accuracy'] = float(balanced_accuracy)
        result['known_accept_rate'] = float(macro_known_accept_rate)
        result['unknown_reject_rate'] = float(macro_unknown_reject_rate)
        result['f1_score'] = float(f1)

        results[f"q>={q_thresh}"] = result

    return results

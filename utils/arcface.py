"""
ArcFace model factories and utilities for wolverine re-identification.
Includes embedding head, ArcFace loss, balanced batch sampling, and model creation.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import AutoModel
import random
from typing import List, Tuple
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


class EmbeddingHead(nn.Module):
    """Linear -> BatchNorm1d -> L2 Norm projection head."""

    def __init__(self, input_dim: int = 768, embedding_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bn(self.fc(x)), p=2, dim=1)


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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                sample_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: L2-normalized embeddings (batch_size, embedding_size)
            labels: Ground truth class labels (batch_size,)
            sample_weights: Optional per-sample weights (batch_size,).
                If provided, computes weighted mean of per-sample losses.

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

        if sample_weights is not None:
            return (F.cross_entropy(output, labels, reduction='none') * sample_weights).mean()
        return F.cross_entropy(output, labels)


class CoverageSampler(Sampler):
    """
    Yield balanced batches indefinitely, guaranteeing full image coverage
    per individual before repeats.

    Each individual maintains an independent shuffled queue of ALL its images.
    When an individual's queue is exhausted, reshuffle just that individual.
    Batches are balanced: k samples per class, remainder filled randomly.
    Tracks coverage statistics (resets per individual).
    """
    def __init__(self, labels, batch_size=36):
        super().__init__(None)
        self.batch_size = batch_size

        self.label_to_all_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_all_indices[label].append(idx)

        self.n_classes = len(self.label_to_all_indices)
        self.k = max(1, batch_size // self.n_classes)
        self.remainder = batch_size - (self.k * self.n_classes)

        # Per-individual shuffled queues and coverage counters
        self._queues = {}
        self._coverage_resets = defaultdict(int)  # count of full-coverage reshuffles
        for label, indices in self.label_to_all_indices.items():
            q = list(indices)
            random.shuffle(q)
            self._queues[label] = q

    def _next_for_label(self, label):
        """Pop next index for this individual; reshuffle if exhausted."""
        q = self._queues[label]
        if not q:
            q = list(self.label_to_all_indices[label])
            random.shuffle(q)
            self._queues[label] = q
            self._coverage_resets[label] += 1
        return q.pop()

    @property
    def coverage_stats(self):
        """Return per-label coverage reset counts and pool sizes."""
        return {
            label: {
                "pool_size": len(self.label_to_all_indices[label]),
                "resets": self._coverage_resets[label],
                "remaining": len(self._queues[label]),
            }
            for label in self.label_to_all_indices
        }

    def __iter__(self):
        labels = list(self.label_to_all_indices.keys())
        while True:
            batch = []
            for label in labels:
                for _ in range(self.k):
                    batch.append(self._next_for_label(label))
            if self.remainder > 0:
                extra_labels = random.sample(labels, self.remainder)
                for label in extra_labels:
                    batch.append(self._next_for_label(label))
            yield batch

    def __len__(self):
        # Approximate: one full coverage pass
        if not self.label_to_all_indices:
            return 0
        max_pool = max(len(v) for v in self.label_to_all_indices.values())
        return math.ceil(max_pool / self.k)


def create_megadescriptor_arcface_model(
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384",
    embedding_dim: int = 128,
    device: str = "cuda"
) -> Tuple[nn.Module, int]:
    """
    Create MegaDescriptor backbone with trainable projection head for ArcFace training.

    Architecture: Frozen MegaDescriptor -> ~1536-d -> EmbeddingHead -> ArcFace

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

    head = EmbeddingHead(backbone_dim, embedding_dim)

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


def create_bioclip2_arcface_model(
    embedding_dim: int = 128,
    image_size: int = 224,
    device: str = "cuda"
) -> Tuple[nn.Module, int]:
    """
    Create BioCLIP-2 backbone with trainable projection head for ArcFace training.

    Architecture: Frozen BioCLIP-2 ViT-L/14 -> 1024-d (raw ViT features) -> EmbeddingHead -> ArcFace

    Args:
        embedding_dim: Output embedding dimension (default 128)
        image_size: Input image size (default 224, must be multiple of 14).
                    Non-native sizes trigger positional embedding interpolation.
        device: Device to place model on

    Returns:
        Tuple of (model, embedding_dim): Model and its output dimension
    """
    import open_clip

    # force_image_size interpolates positional embeddings for non-224 inputs
    backbone = open_clip.create_model('hf-hub:imageomics/bioclip-2',
                                       force_image_size=image_size)

    # Disable CLIP text-alignment projection to get raw ViT-L CLS features (1024-d)
    # instead of projected features (768-d) that discard fine-grained visual info
    backbone.visual.proj = None

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Get backbone output dimension via dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        backbone_dim = backbone.encode_image(dummy).shape[1]  # 1024 (raw ViT-L features)

    head = EmbeddingHead(backbone_dim, embedding_dim)

    class BioCLIP2WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                features = self.backbone.encode_image(x)
            return self.head(features)

        def get_trainable_parameters(self):
            return self.head.parameters()

    model = BioCLIP2WithHead(backbone, head)

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


def create_dinov3_arcface_model(
    embedding_dim: int = 128,
    device: str = "cuda"
) -> Tuple[nn.Module, int]:
    """
    Create DINOv3 backbone with trainable projection head for ArcFace training.

    Architecture: Frozen DINOv3-ViT-B/16 -> 768-d CLS -> EmbeddingHead -> ArcFace

    Args:
        embedding_dim: Output embedding dimension (default 128)
        device: Device to place model on

    Returns:
        Tuple of (model, embedding_dim): Model and its output dimension
    """
    backbone = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        token=os.environ.get("HF_TOKEN"),
    )

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    head = EmbeddingHead(768, embedding_dim)

    class DINOv3WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                outputs = self.backbone(x)
                features = outputs.last_hidden_state[:, 0, :]
            return self.head(features)

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

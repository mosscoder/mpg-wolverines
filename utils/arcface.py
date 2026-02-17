"""
ArcFace model factories and utilities for wolverine re-identification.
Includes embedding head, ArcFace loss, PK batch sampling, and model creation.
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
        """Generate batches of P identities x K samples."""
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


def create_bioclip2_arcface_model(
    embedding_dim: int = 128,
    image_size: int = 224,
    device: str = "cuda"
) -> Tuple[nn.Module, int]:
    """
    Create BioCLIP-2 backbone with trainable projection head for ArcFace training.

    Architecture: Frozen BioCLIP-2 ViT-L/14 -> 1024-d (raw ViT features) -> Linear -> 128-d

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

    # Trainable projection head
    head = nn.Linear(backbone_dim, embedding_dim)

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

    Architecture: Frozen DINOv3-ViT-B/16 -> 768-d CLS -> Linear -> embedding_dim

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

    # Trainable projection head (768 -> embedding_dim)
    head = nn.Linear(768, embedding_dim)

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

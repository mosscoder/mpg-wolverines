"""
Quality-adaptive AdaFace loss for wolverine re-identification.

Extends ArcFace by scaling the angular margin per-sample based on pelage quality
scores. High-quality images get the full margin; low-quality images get reduced
or negative margin, preventing the model from overfitting to noisy samples.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityAdaFaceLoss(nn.Module):
    """
    AdaFace loss with quality-adaptive angular margin.

    For each sample, the margin is scaled by g(q) = 2*q - 1, mapping quality
    scores in [0, 1] to [-1, 1]. High-quality samples (q~1) get full +m margin,
    low-quality samples (q~0) get -m margin (relaxed).

    When use_quality_scaling=False, applies fixed margin +m to all samples
    (equivalent to standard ArcFace, saves compute).
    """

    def __init__(self, num_classes: int, embedding_size: int,
                 margin: float = 0.5, scale: float = 64.0,
                 use_quality_scaling: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.use_quality_scaling = use_quality_scaling

        # Learnable class centers (same as ArcFace)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                quality_scores: torch.Tensor = None) -> torch.Tensor:
        """
        Compute quality-adaptive AdaFace loss.

        Args:
            embeddings: L2-normalized embeddings (B, embedding_size)
            labels: Ground truth class labels (B,)
            quality_scores: Pelage quality scores in [0, 1] (B,).
                If None, defaults to all-ones (standard ArcFace margin).

        Returns:
            Scalar loss value
        """
        # L2 normalize embeddings and weights
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (B, C)
        cos_theta = F.linear(emb_norm, weight_norm)

        # Safe acos
        theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

        # Per-sample adaptive margin
        if self.use_quality_scaling and quality_scores is not None:
            # g(q) maps [0, 1] -> [-1, 1]
            g_q = (2.0 * quality_scores - 1.0).detach()  # (B,)
            margin_dynamic = g_q * self.margin  # (B,)
        else:
            # Fixed margin (standard ArcFace behavior)
            margin_dynamic = torch.full(
                (embeddings.size(0),), self.margin,
                device=embeddings.device, dtype=embeddings.dtype
            )

        # One-hot encode labels: (B, C)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Add margin to positive class angles only
        theta_m = theta + one_hot * margin_dynamic.unsqueeze(1)

        # Clamp to valid range [0, pi]
        theta_m = theta_m.clamp(0, math.pi)

        # Convert back to cosine and scale
        output = torch.cos(theta_m) * self.scale

        return F.cross_entropy(output, labels)

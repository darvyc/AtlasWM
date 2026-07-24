"""Matched anti-collapse baselines for controlled experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from atlaswm.statistics import gaussian_bhep_discrepancy


class ZeroRegularizer(nn.Module):
    """Return a differentiable zero scalar for prediction-only controls."""

    def forward(self, latent: Tensor) -> Tensor:
        return latent.sum() * 0.0


class CovarianceRegularizer(nn.Module):
    """Penalize latent mean, variance error and off-diagonal covariance."""

    def __init__(self, variance_target: float = 1.0, eps: float = 1e-4):
        super().__init__()
        if variance_target <= 0 or eps <= 0:
            raise ValueError("variance_target and eps must be positive")
        self.variance_target = variance_target
        self.eps = eps

    def forward(self, latent: Tensor) -> Tensor:
        latent = latent.reshape(-1, latent.shape[-1])
        centered = latent - latent.mean(dim=0, keepdim=True)
        denominator = max(latent.shape[0] - 1, 1)
        covariance = centered.t() @ centered / denominator
        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance - torch.diag_embed(diagonal)
        mean_penalty = latent.mean(dim=0).square().mean()
        variance_penalty = (diagonal - self.variance_target).square().mean()
        covariance_penalty = off_diagonal.square().mean()
        return mean_penalty + variance_penalty + covariance_penalty


class FullGaussianMMDRegularizer(nn.Module):
    """Full-dimensional Gaussian MMD/BHEP baseline."""

    def __init__(self, beta: float = 1.0, pair_chunk_size: int | None = 512):
        super().__init__()
        self.beta = beta
        self.pair_chunk_size = pair_chunk_size

    def forward(self, latent: Tensor) -> Tensor:
        latent = latent.reshape(-1, latent.shape[-1])
        return gaussian_bhep_discrepancy(
            latent,
            beta=self.beta,
            pair_chunk_size=self.pair_chunk_size,
        )

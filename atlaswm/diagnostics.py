"""Latent-geometry diagnostics used for collapse detection and evaluation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def effective_rank(values: Tensor, eps: float = 1e-12) -> Tensor:
    """Entropy effective rank of a centered sample matrix."""
    if values.ndim != 2:
        raise ValueError("values must have shape (N,D)")
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return entropy.exp()


def covariance_error(values: Tensor, eps: float = 1e-12) -> Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape (N,D)")
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / max(values.shape[0] - 1, 1)
    identity = torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
    return torch.linalg.matrix_norm(covariance - identity) / math.sqrt(values.shape[1] + eps)


def latent_diagnostics(values: Tensor, pair_sample_size: int = 512) -> dict[str, float]:
    """Compute independent indicators rather than declaring collapse from one scalar."""
    values = values.reshape(-1, values.shape[-1]).float()
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(centered)
    singular_fraction = singular_values.max() / singular_values.sum().clamp_min(1e-12)
    sample = values[: min(values.shape[0], pair_sample_size)]
    normalized = torch.nn.functional.normalize(sample, dim=-1)
    cosine = normalized @ normalized.t()
    if cosine.shape[0] > 1:
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
        pair_values = cosine[mask]
        cosine_mean = pair_values.mean()
        cosine_std = pair_values.std(unbiased=False)
    else:
        cosine_mean = values.new_zeros(())
        cosine_std = values.new_zeros(())
    return {
        "latent_mean_norm": float(values.mean(dim=0).norm()),
        "coordinate_variance_mean": float(variance.mean()),
        "coordinate_variance_min": float(variance.min()),
        "effective_rank": float(effective_rank(values)),
        "effective_rank_fraction": float(effective_rank(values) / values.shape[1]),
        "largest_singular_fraction": float(singular_fraction),
        "pairwise_cosine_mean": float(cosine_mean),
        "pairwise_cosine_std": float(cosine_std),
        "covariance_error": float(covariance_error(values)),
    }

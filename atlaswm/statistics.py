"""Statistical primitives used by AtlasReg and evaluation diagnostics."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

EstimatorName = Literal["biased", "unbiased"]


def henze_zirkler_beta(n_samples: int, dim: int) -> float:
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if dim < 1:
        raise ValueError("dim must be positive")
    return (2.0**-0.5) * (((2.0 * dim + 1.0) * n_samples / 4.0) ** (1.0 / (dim + 4.0)))


def gaussian_bhep_null_floor(n_samples: int, dim: int, beta: float) -> float:
    if n_samples < 1 or dim < 1 or beta <= 0:
        raise ValueError("n_samples, dim and beta must be positive")
    return (1.0 / n_samples) * (1.0 - (1.0 + 2.0 * beta * beta) ** (-dim / 2.0))


def gaussian_bhep_discrepancy(
    samples: Tensor,
    *,
    beta: float = 1.0,
    estimator: EstimatorName = "biased",
    pair_chunk_size: int | None = None,
) -> Tensor:
    """Gaussian-weighted BHEP/MMD discrepancy against ``N(0,I)``.

    ``pair_chunk_size`` bounds the temporary pairwise distance matrix. The
    result remains exact up to floating-point summation order.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"unknown estimator: {estimator!r}")
    if samples.ndim == 1:
        samples = samples.unsqueeze(-1)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (N,k)")
    n_samples, dim = samples.shape
    if n_samples < 1:
        raise ValueError("at least one sample is required")
    if estimator == "unbiased" and n_samples < 2:
        raise ValueError("unbiased estimator requires at least two samples")

    beta_sq = beta * beta
    norms = samples.square().sum(dim=-1)
    chunk = n_samples if pair_chunk_size is None else pair_chunk_size
    if chunk < 1:
        raise ValueError("pair_chunk_size must be positive")

    gram_sum = samples.new_zeros(())
    diagonal_sum = samples.new_zeros(())
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        block = samples[start:stop]
        distances = (
            block.square().sum(dim=-1, keepdim=True)
            + norms.unsqueeze(0)
            - 2.0 * block @ samples.t()
        ).clamp_min(0.0)
        gram = torch.exp(-0.5 * beta_sq * distances)
        gram_sum = gram_sum + gram.sum()
        rows = torch.arange(start, stop, device=samples.device)
        diagonal_sum = diagonal_sum + gram[
            torch.arange(stop - start, device=samples.device), rows
        ].sum()

    if estimator == "biased":
        first = gram_sum / (n_samples * n_samples)
    else:
        first = (gram_sum - diagonal_sum) / (n_samples * (n_samples - 1))

    coefficient = (1.0 + beta_sq) ** (-dim / 2.0)
    second = 2.0 * coefficient * torch.exp(
        -beta_sq * norms / (2.0 * (1.0 + beta_sq))
    ).mean()
    third = (1.0 + 2.0 * beta_sq) ** (-dim / 2.0)
    return first - second + third


def empirical_cf_discrepancy(
    samples: Tensor,
    nodes: Tensor,
    target_cf: Tensor,
    integration_weights: Tensor,
    *,
    estimator: EstimatorName = "biased",
    frequency_chunk_size: int = 16,
) -> Tensor:
    """Return one weighted ECF discrepancy per projected sample column."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape (N,M)")
    if nodes.ndim != 1 or target_cf.ndim != 1 or integration_weights.ndim != 1:
        raise ValueError("nodes, target_cf and integration_weights must be vectors")
    if not (nodes.numel() == target_cf.numel() == integration_weights.numel()):
        raise ValueError("quadrature vectors must have equal length")
    if frequency_chunk_size < 1:
        raise ValueError("frequency_chunk_size must be positive")
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"unknown estimator: {estimator!r}")
    n_samples = samples.shape[0]
    if n_samples < 1:
        raise ValueError("at least one sample is required")
    if estimator == "unbiased" and n_samples < 2:
        raise ValueError("unbiased estimator requires at least two samples")

    result = samples.new_zeros(samples.shape[1])
    for start in range(0, nodes.numel(), frequency_chunk_size):
        stop = min(start + frequency_chunk_size, nodes.numel())
        current_nodes = nodes[start:stop].reshape(-1, 1, 1)
        angles = current_nodes * samples.unsqueeze(0)
        real = angles.cos().mean(dim=1)
        imaginary = angles.sin().mean(dim=1)
        empirical_abs_sq = real.square() + imaginary.square()
        target = target_cf[start:stop].reshape(-1, 1)
        if estimator == "unbiased":
            empirical_abs_sq = (n_samples * empirical_abs_sq - 1.0) / (n_samples - 1.0)
        integrand = empirical_abs_sq - 2.0 * target * real + target.square()
        weights = integration_weights[start:stop].reshape(-1, 1)
        result = result + (weights * integrand).sum(dim=0)
    return result


def spherical_projection_even_moment_coefficient(dim: int, order: int) -> float:
    if dim < 1:
        raise ValueError("dim must be positive")
    if order < 0 or order % 2:
        raise ValueError("order must be a non-negative even integer")
    if order == 0:
        return 1.0
    half_order = order // 2
    numerator = math.prod(range(1, 2 * half_order, 2))
    denominator = math.prod(dim + 2 * index for index in range(half_order))
    return numerator / denominator


def student_t_unit_variance_scale(nu: float) -> float:
    if nu <= 2:
        raise ValueError("unit variance requires nu > 2")
    return math.sqrt((nu - 2.0) / nu)

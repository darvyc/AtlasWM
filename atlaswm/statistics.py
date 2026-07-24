"""Mathematical primitives for characteristic-function distribution matching.

The functions in this module make explicit which quantities are population
metrics, which are finite-sample estimators, and where finite-sample bias enters.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

EstimatorName = Literal["biased", "unbiased"]


def henze_zirkler_beta(n_samples: int, dim: int) -> float:
    """Return the classical Henze-Zirkler bandwidth parameter.

    beta_n = 2^(-1/2) * (((2d + 1)n) / 4)^(1 / (d + 4)).
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if dim < 1:
        raise ValueError("dim must be positive")
    return (2.0 ** -0.5) * (((2.0 * dim + 1.0) * n_samples / 4.0) ** (1.0 / (dim + 4.0)))


def gaussian_bhep_null_floor(
    n_samples: int,
    dim: int,
    beta: float,
) -> float:
    """Expected biased BHEP/MMD floor under an exact Gaussian target.

    For iid Y_i ~ N(0, I_dim), the biased empirical-measure discrepancy has

        E[D_V] = (1 / n) * (1 - (1 + 2 beta^2)^(-dim / 2)).

    The floor is sampling bias, not model misspecification.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if dim < 1:
        raise ValueError("dim must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    return (1.0 / n_samples) * (1.0 - (1.0 + 2.0 * beta * beta) ** (-dim / 2.0))


def gaussian_bhep_discrepancy(
    y: Tensor,
    beta: float = 1.0,
    estimator: EstimatorName = "biased",
) -> Tensor:
    """Exact Gaussian-weighted CF discrepancy against N(0, I).

    This is the normalized Baringhaus-Henze-Epps-Pulley functional. For a
    sample y_1, ..., y_n in R^k,

        D = E_ij exp(-beta^2 ||y_i-y_j||^2 / 2)
            - 2(1+beta^2)^(-k/2)
              E_i exp(-beta^2 ||y_i||^2 / (2(1+beta^2)))
            + (1+2beta^2)^(-k/2).

    The ``biased`` version is the squared RKHS distance between the empirical
    measure and the Gaussian target, hence non-negative. The ``unbiased``
    version removes diagonal kernel terms and has expectation equal to the
    population discrepancy, but can be negative for a finite batch.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"unknown estimator: {estimator!r}")
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if y.dim() != 2:
        raise ValueError(f"expected shape (N, k), got {tuple(y.shape)}")

    n, k = y.shape
    if n < 1:
        raise ValueError("at least one sample is required")
    if estimator == "unbiased" and n < 2:
        raise ValueError("unbiased estimator requires at least two samples")

    b2 = beta * beta
    sq_norms = (y * y).sum(dim=-1)
    sq_dists = (
        sq_norms.unsqueeze(0)
        + sq_norms.unsqueeze(1)
        - 2.0 * (y @ y.t())
    ).clamp_min(0.0)
    gram = torch.exp(-0.5 * b2 * sq_dists)

    if estimator == "biased":
        term1 = gram.mean()
    else:
        term1 = (gram.sum() - torch.diagonal(gram).sum()) / (n * (n - 1))

    coef2 = (1.0 + b2) ** (-k / 2.0)
    term2 = 2.0 * coef2 * torch.exp(
        -b2 * sq_norms / (2.0 * (1.0 + b2))
    ).mean()
    term3 = (1.0 + 2.0 * b2) ** (-k / 2.0)
    return term1 - term2 + term3


def empirical_cf_discrepancy(
    samples: Tensor,
    nodes: Tensor,
    target_cf: Tensor,
    integration_weights: Tensor,
    estimator: EstimatorName = "biased",
) -> Tensor:
    """Quadrature estimate of a one-dimensional CF discrepancy.

    Args:
        samples: Shape (N, M), where M is the number of projections.
        nodes: Shape (T,), frequency nodes.
        target_cf: Shape (T,), real CF values of a symmetric target.
        integration_weights: Shape (T,), quadrature weights including w(t).
        estimator: ``biased`` for |phi_N-phi_0|^2, or ``unbiased`` for the
            U-statistic correction of |phi_P|^2.

    Returns:
        Tensor of shape (M,), one discrepancy per projection.
    """
    if samples.dim() != 2:
        raise ValueError("samples must have shape (N, M)")
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"unknown estimator: {estimator!r}")
    n = samples.shape[0]
    if n < 1:
        raise ValueError("at least one sample is required")
    if estimator == "unbiased" and n < 2:
        raise ValueError("unbiased estimator requires at least two samples")

    t_h = nodes.view(-1, 1, 1) * samples.unsqueeze(0)
    re = t_h.cos().mean(dim=1)
    im = t_h.sin().mean(dim=1)
    target = target_cf.view(-1, 1)

    phi_abs_sq = re.square() + im.square()
    if estimator == "biased":
        integrand = phi_abs_sq - 2.0 * target * re + target.square()
    else:
        phi_abs_sq_u = (n * phi_abs_sq - 1.0) / (n - 1.0)
        integrand = phi_abs_sq_u - 2.0 * target * re + target.square()

    return (integration_weights.view(-1, 1) * integrand).sum(dim=0)


def spherical_projection_even_moment_coefficient(dim: int, order: int) -> float:
    """Coefficient c with E_u[(u^T x)^order] = c ||x||^order.

    ``u`` is uniform on S^(d-1) and ``order`` must be even. For order 2m,

        c = (2m-1)!! / [d(d+2)...(d+2m-2)].
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    if order < 0 or order % 2:
        raise ValueError("order must be a non-negative even integer")
    if order == 0:
        return 1.0
    m = order // 2
    numerator = math.prod(range(1, 2 * m, 2))
    denominator = math.prod(dim + 2 * j for j in range(m))
    return numerator / denominator


def student_t_unit_variance_scale(nu: float) -> float:
    """Scale making a Student-t(nu) variable have unit variance."""
    if nu <= 2:
        raise ValueError("unit variance requires nu > 2")
    return math.sqrt((nu - 2.0) / nu)

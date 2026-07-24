"""Frequency quadrature rules for characteristic-function discrepancies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QuadratureRule:
    nodes: Tensor
    integration_weights: Tensor


def _trapezoid(
    t_max: float,
    n_knots: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if n_knots < 2:
        raise ValueError("n_knots must be at least two")
    nodes = torch.linspace(0.0, t_max, n_knots, device=device, dtype=dtype)
    delta = t_max / (n_knots - 1)
    weights = torch.full_like(nodes, 2.0 * delta)
    weights[0] = delta
    weights[-1] = delta
    return nodes, weights


def gaussian_kernel(
    *,
    lambda_: float = 1.0,
    n_knots: int = 33,
    t_max: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> QuadratureRule:
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    cutoff = float(t_max if t_max is not None else max(6.0, 4.0 * lambda_))
    nodes, trapezoid_weights = _trapezoid(
        cutoff,
        n_knots,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-0.5 * (nodes / lambda_).square())
    return QuadratureRule(nodes, trapezoid_weights * kernel)


def two_scale_gaussian_kernel(
    *,
    lambda_1: float = 0.5,
    lambda_2: float = 2.0,
    alpha: float = 0.5,
    n_knots: int = 33,
    t_max: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> QuadratureRule:
    if not 0 < lambda_1 < lambda_2:
        raise ValueError("require 0 < lambda_1 < lambda_2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    cutoff = float(t_max if t_max is not None else max(8.0, 4.0 * lambda_2))
    nodes, trapezoid_weights = _trapezoid(
        cutoff,
        n_knots,
        device=device,
        dtype=dtype,
    )
    first = torch.exp(-0.5 * (nodes / lambda_1).square())
    second = torch.exp(-0.5 * (nodes / lambda_2).square())
    kernel = alpha * first + (1.0 - alpha) * second
    return QuadratureRule(nodes, trapezoid_weights * kernel)

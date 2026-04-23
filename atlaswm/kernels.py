"""Integration kernels (weight functions) for CF-based distribution matching.

The Epps-Pulley family of statistics are integrals of the form

    T = int w(t) |phi_N(t) - phi_0(t)|^2 dt

where w is a weight function that controls which "frequencies" of the CF
mismatch are penalized.

  - Single Gaussian weight:     w(t) = exp(-t^2 / (2 lambda^2))
    Small lambda -> concentrates near t=0 -> body-sensitive (low moments).
    Large lambda -> spreads further in t  -> tail-sensitive (fine structure).

  - Two-scale Gaussian weight:  w(t) = a * exp(-t^2/(2 l1^2)) + (1-a) * exp(-t^2/(2 l2^2))
    Simultaneous body + tail sensitivity. With lambda_2 ~ 4 * lambda_1 and
    alpha = 0.5, provides complementary coverage. Runtime cost is identical
    to single-kernel since w(t) is precomputed at quadrature nodes once.

We integrate by trapezoidal quadrature on [0, t_max], exploiting the fact
that both phi_N and phi_0 are even (for symmetric targets) so the integrand
is even and the integral on [0, t_max] times 2 equals the integral on
[-t_max, t_max].
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class QuadratureRule:
    """A trapezoidal quadrature rule pre-baked with a weight kernel.

    Fields:
        nodes: (T,) quadrature points t_i.
        weights: (T,) trapezoidal weights w_i (including dt).
        kernel: (T,) kernel values w(t_i).
    """

    nodes: Tensor
    weights: Tensor
    kernel: Tensor

    @property
    def integration_weights(self) -> Tensor:
        """Pre-baked product w(t_i) * dt_i for direct summation.

        Usage:
            T = (quad.integration_weights * sq_diff).sum(dim=0)
        """
        return self.weights * self.kernel


def _trapezoid_nodes(
    t_min: float,
    t_max: float,
    n_knots: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Uniform trapezoidal nodes and weights on [t_min, t_max]."""
    if n_knots < 2:
        raise ValueError(f"n_knots must be >= 2, got {n_knots}")
    t = torch.linspace(t_min, t_max, n_knots, device=device, dtype=dtype)
    dt = (t_max - t_min) / (n_knots - 1)
    w = torch.full_like(t, dt)
    w[0] = dt / 2  # trapezoidal endpoint correction
    w[-1] = dt / 2
    return t, w


def gaussian_kernel(
    lambda_: float = 1.0,
    n_knots: int = 17,
    t_min: float = 0.0,
    t_max: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> QuadratureRule:
    """Single-scale Gaussian weight: w(t) = exp(-t^2 / (2 lambda^2)).

    Args:
        lambda_: Scale parameter. Smaller = more body-weighted.
        n_knots: Number of trapezoidal quadrature nodes.
        t_min, t_max: Integration range. We use symmetry to only integrate
            on [0, t_max] and multiply by 2.
        device, dtype: Optional torch options.

    Returns:
        QuadratureRule precomputed for direct summation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")
    t, w = _trapezoid_nodes(t_min, t_max, n_knots, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (t / lambda_) ** 2)
    # Factor of 2 because we exploit even symmetry and only integrate on [0, t_max]
    return QuadratureRule(nodes=t, weights=2 * w, kernel=kernel)


def two_scale_gaussian_kernel(
    lambda_1: float = 0.5,
    lambda_2: float = 2.0,
    alpha: float = 0.5,
    n_knots: int = 17,
    t_min: float = 0.0,
    t_max: float = 6.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> QuadratureRule:
    """Two-scale Gaussian weight for joint body-and-tail sensitivity.

    w(t) = alpha * exp(-t^2 / (2 l1^2)) + (1 - alpha) * exp(-t^2 / (2 l2^2))

    With lambda_1 < lambda_2:
      - The lambda_1 component concentrates near t=0 (body).
      - The lambda_2 component spreads further (tail).

    Defaults give lambda_2 / lambda_1 = 4, a well-separated scale ratio.
    Larger t_max is recommended relative to the single-scale default
    (6 vs 4) to capture the larger-lambda tail.

    Args:
        lambda_1: Small scale (body). Must be positive and < lambda_2.
        lambda_2: Large scale (tail).
        alpha: Mixture weight on the lambda_1 component. In (0, 1).
        n_knots, t_min, t_max, device, dtype: as in gaussian_kernel.

    Returns:
        QuadratureRule precomputed for direct summation.
    """
    if not (0 < lambda_1 < lambda_2):
        raise ValueError(
            f"Require 0 < lambda_1 < lambda_2, got {lambda_1}, {lambda_2}"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    t, w = _trapezoid_nodes(t_min, t_max, n_knots, device=device, dtype=dtype)
    k1 = torch.exp(-0.5 * (t / lambda_1) ** 2)
    k2 = torch.exp(-0.5 * (t / lambda_2) ** 2)
    kernel = alpha * k1 + (1 - alpha) * k2
    return QuadratureRule(nodes=t, weights=2 * w, kernel=kernel)

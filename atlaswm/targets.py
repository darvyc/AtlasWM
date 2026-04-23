"""Target distributions for CF-based distribution matching.

A target is specified by its characteristic function
    phi_0(t) = E[exp(i * t * X)]  where X ~ target.

For spherically-symmetric targets (which both Gaussian and isotropic
Student-t are), the CF depends only on |t|, and the 1D marginal CF is
real-valued and even.

Currently implemented:
  - StandardGaussian:  X ~ N(0, 1),    phi_0(t) = exp(-t^2 / 2)
  - StudentT:          X ~ t_nu,       phi_0(t) = K_{nu/2}(sqrt(nu)|t|)
                                                 * (sqrt(nu)|t|)^(nu/2)
                                                 / (Gamma(nu/2) * 2^(nu/2 - 1))
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Target(ABC):
    """Abstract base class for target distributions in AtlasReg."""

    @abstractmethod
    def char_fn_1d(self, t: Tensor) -> Tensor:
        """Real-valued characteristic function of the 1D marginal at t.

        All targets here are symmetric (even), so the CF is real.

        Args:
            t: Tensor of any shape.

        Returns:
            Tensor of the same shape as `t` with phi_0(t).
        """

    @abstractmethod
    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        """CF of the spherically-symmetric k-D target as a function of |t|^2.

        Args:
            t_norm_sq: Tensor of |t|^2 values.
            k: Ambient dimension of t.

        Returns:
            Tensor of phi_0(t) values.
        """


class StandardGaussian(Target):
    """Isotropic standard Gaussian: X ~ N(0, I)."""

    def char_fn_1d(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * t * t)

    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        # N(0, I_k) has phi(t) = exp(-|t|^2 / 2), independent of k
        return torch.exp(-0.5 * t_norm_sq)


class StudentT(Target):
    """Isotropic multivariate Student-t with nu degrees of freedom.

    Heavier tails than Gaussian (controlled by nu). As nu -> infinity,
    converges to standard Gaussian. Useful when the environment's natural
    latent distribution has non-Gaussian tails (e.g. low intrinsic
    dimensionality embedded in high-d latent space, or rare outlier events).

    The CF requires a modified Bessel function K_{nu/2} that PyTorch does
    not expose for arbitrary orders. We precompute phi_0 at quadrature
    nodes using SciPy once at module initialization.
    """

    def __init__(self, nu: float):
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.nu = float(nu)

    def char_fn_1d(self, t: Tensor) -> Tensor:
        raise NotImplementedError(
            "StudentT.char_fn_1d requires SciPy. "
            "Use StudentT.precompute_char_fn_1d(nu, nodes) once at setup."
        )

    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        raise NotImplementedError(
            "StudentT.char_fn_kd_norm requires SciPy. "
            "Use StudentT.precompute_char_fn_kd(nu, nodes, k) once at setup."
        )

    @staticmethod
    def precompute_char_fn_1d(nu: float, t_nodes: Tensor) -> Tensor:
        """Precompute the Student-t 1D CF at quadrature nodes using SciPy.

        Args:
            nu: Degrees of freedom.
            t_nodes: Tensor of quadrature node positions.

        Returns:
            Tensor of same shape as t_nodes with phi_0(t) values.
        """
        try:
            from scipy.special import kv, gamma
        except ImportError as e:
            raise ImportError(
                "Student-t target requires SciPy. Install with: pip install scipy"
            ) from e
        import numpy as np

        t_np = t_nodes.detach().cpu().numpy().astype(np.float64)
        abs_t = np.abs(t_np)
        result = np.ones_like(t_np)

        # Avoid the singular point t=0 where phi_0(0) = 1 by continuity.
        mask = abs_t > 1e-10
        arg = np.sqrt(nu) * abs_t[mask]
        coef = 1.0 / (gamma(nu / 2.0) * 2.0 ** (nu / 2.0 - 1.0))
        # kv is the modified Bessel function of the second kind, order nu/2.
        result[mask] = coef * kv(nu / 2.0, arg) * arg ** (nu / 2.0)

        return torch.from_numpy(result).to(
            device=t_nodes.device, dtype=t_nodes.dtype
        )

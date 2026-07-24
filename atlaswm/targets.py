"""Target distributions for characteristic-function distribution matching."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Target(ABC):
    """Abstract target distribution."""

    @abstractmethod
    def char_fn_1d(self, t: Tensor) -> Tensor:
        """Real characteristic function of a symmetric 1D marginal."""

    @abstractmethod
    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        """Radial characteristic function in k dimensions."""


class StandardGaussian(Target):
    """Isotropic standard Gaussian N(0, I)."""

    def char_fn_1d(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * t * t)

    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        del k
        return torch.exp(-0.5 * t_norm_sq)


class StudentT(Target):
    """Spherically symmetric Student-t target.

    ``scale`` is the usual multiplicative scale parameter. A scale-one
    Student-t has variance ``nu / (nu - 2)`` when ``nu > 2``; use
    ``sqrt((nu - 2) / nu)`` for a unit-variance marginal.
    """

    def __init__(self, nu: float, scale: float = 1.0):
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.nu = float(nu)
        self.scale = float(scale)

    def char_fn_1d(self, t: Tensor) -> Tensor:
        raise NotImplementedError(
            "StudentT.char_fn_1d requires SciPy precomputation; use "
            "StudentT.precompute_char_fn_1d."
        )

    def char_fn_kd_norm(self, t_norm_sq: Tensor, k: int) -> Tensor:
        raise NotImplementedError(
            "Multivariate Student-t CF precomputation is not implemented."
        )

    @staticmethod
    def precompute_char_fn_1d(
        nu: float,
        t_nodes: Tensor,
        scale: float = 1.0,
    ) -> Tensor:
        """Precompute the Student-t CF robustly using SciPy.

        The direct Bessel expression can overflow for large ``nu`` even though
        the characteristic function is bounded by one. We use the Bessel form
        where numerically finite and fall back to a Fourier integral otherwise.
        This work is done once at initialization, not in each training step.
        """
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        try:
            import numpy as np
            from scipy.integrate import quad
            from scipy.special import gammaln, kve
        except ImportError as exc:
            raise ImportError(
                "Student-t target requires SciPy. Install with: pip install scipy"
            ) from exc

        raw = t_nodes.detach().cpu().numpy().astype(np.float64)
        abs_t = np.abs(raw) * float(scale)
        result = np.ones_like(abs_t)
        mask = abs_t > 1e-12
        if not np.any(mask):
            return torch.from_numpy(result).to(device=t_nodes.device, dtype=t_nodes.dtype)

        order = nu / 2.0
        arg = np.sqrt(nu) * abs_t[mask]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            log_phi = (
                np.log(kve(order, arg))
                - arg
                + order * np.log(arg)
                - gammaln(order)
                - (order - 1.0) * np.log(2.0)
            )
            vals = np.exp(log_phi)

        bad = ~np.isfinite(vals) | (vals < 0.0) | (vals > 1.0 + 1e-10)
        if np.any(bad):
            log_norm = (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * (np.log(nu) + np.log(np.pi))
            )
            norm = float(np.exp(log_norm))

            def density(x: float) -> float:
                return norm * (1.0 + x * x / nu) ** (-(nu + 1.0) / 2.0)

            bad_indices = np.flatnonzero(bad)
            for idx in bad_indices:
                frequency = float(arg[idx] / np.sqrt(nu))
                integral, _ = quad(
                    density,
                    0.0,
                    np.inf,
                    weight="cos",
                    wvar=frequency,
                    epsabs=1e-11,
                    epsrel=1e-11,
                    limit=250,
                )
                vals[idx] = 2.0 * integral

        result[mask] = np.clip(vals, 0.0, 1.0)
        return torch.from_numpy(result).to(device=t_nodes.device, dtype=t_nodes.dtype)

"""Target characteristic functions for AtlasReg."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Target(ABC):
    @abstractmethod
    def char_fn_1d(self, frequencies: Tensor) -> Tensor:
        raise NotImplementedError


class StandardGaussian(Target):
    def char_fn_1d(self, frequencies: Tensor) -> Tensor:
        return torch.exp(-0.5 * frequencies.square())

    def char_fn_kd_norm(self, squared_norm: Tensor, k: int) -> Tensor:
        del k
        return torch.exp(-0.5 * squared_norm)


class StudentT(Target):
    def __init__(self, nu: float, scale: float = 1.0):
        if nu <= 0 or scale <= 0:
            raise ValueError("nu and scale must be positive")
        self.nu = float(nu)
        self.scale = float(scale)

    def char_fn_kd_norm(self, squared_norm: Tensor, k: int) -> Tensor:
        del squared_norm, k
        raise NotImplementedError("multivariate Student-t matching is not implemented")

    def char_fn_1d(self, frequencies: Tensor) -> Tensor:
        return self.precompute_char_fn_1d(self.nu, frequencies, scale=self.scale)

    @staticmethod
    def precompute_char_fn_1d(nu: float, frequencies: Tensor, scale: float = 1.0) -> Tensor:
        """Evaluate the Student-t CF with stable scaled-Bessel arithmetic."""
        if nu <= 0 or scale <= 0:
            raise ValueError("nu and scale must be positive")
        try:
            import numpy as np
            from scipy.integrate import quad
            from scipy.special import gammaln, kve
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Student-t targets require scipy") from exc

        raw = frequencies.detach().cpu().numpy().astype(np.float64)
        absolute = np.abs(raw) * float(scale)
        result = np.ones_like(absolute)
        active = absolute > 1e-12
        if not np.any(active):
            return torch.from_numpy(result).to(frequencies)

        order = nu / 2.0
        arguments = np.sqrt(nu) * absolute[active]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            log_values = (
                np.log(kve(order, arguments))
                - arguments
                + order * np.log(arguments)
                - gammaln(order)
                - (order - 1.0) * np.log(2.0)
            )
            values = np.exp(log_values)

        invalid = ~np.isfinite(values) | (values < 0.0) | (values > 1.0 + 1e-10)
        if np.any(invalid):
            log_normalizer = (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * (np.log(nu) + np.log(np.pi))
            )
            normalizer = float(np.exp(log_normalizer))

            def density(value: float) -> float:
                return normalizer * (1.0 + value * value / nu) ** (-(nu + 1.0) / 2.0)

            for index in np.flatnonzero(invalid):
                frequency = float(arguments[index] / np.sqrt(nu))
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
                values[index] = 2.0 * integral

        result[active] = np.clip(values, 0.0, 1.0)
        return torch.from_numpy(result).to(device=frequencies.device, dtype=frequencies.dtype)

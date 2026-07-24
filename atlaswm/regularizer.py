"""AtlasReg: structured characteristic-function matching for latent spaces.

AtlasReg approximates a sliced characteristic-function discrepancy between the
latent distribution and a chosen target. The implementation deliberately
separates:

* target matching from batch-standardized shape testing;
* biased non-negative empirical discrepancies from unbiased U-statistics;
* exact Gaussian closed forms from finite-frequency quadrature;
* deterministic cubature properties from stochastic random rotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from atlaswm.designs import cross_polytope, random_haar, random_rotation, simplex
from atlaswm.kernels import gaussian_kernel, two_scale_gaussian_kernel
from atlaswm.statistics import (
    EstimatorName,
    empirical_cf_discrepancy,
    gaussian_bhep_discrepancy,
    henze_zirkler_beta,
)
from atlaswm.targets import StandardGaussian, StudentT, Target

DesignName = Literal["cross_polytope", "simplex", "haar"]
KernelName = Literal["single", "two_scale"]
TargetName = Literal["gaussian", "student_t"]
OneDBackend = Literal["quadrature", "closed_form"]


@dataclass
class AtlasRegConfig:
    """Configuration for AtlasReg.

    Defaults perform genuine matching to N(0, I): projections are not
    studentized and k-dimensional subspaces are not whitened. Set
    ``standardize_1d=True`` for per-projection location-scale-invariant shape
    testing, or ``whiten_kd=True`` for covariance-standardized subspace tests.
    """

    design: DesignName = "cross_polytope"
    n_haar_projections: int = 1024
    rotate: bool = True
    deduplicate_antipodes: bool = True

    subspace_dim: int = 1
    n_subspaces: int = 1

    target: TargetName = "gaussian"
    student_t_nu: float = 5.0
    student_t_scale: float = 1.0

    standardize_1d: bool = False
    whiten_kd: bool = False

    estimator: EstimatorName = "biased"

    one_d_backend: OneDBackend = "quadrature"
    kernel: KernelName = "two_scale"
    lambda_: float = 1.0
    lambda_1: float = 0.5
    lambda_2: float = 2.0
    alpha: float = 0.5
    n_knots: int = 17

    hz_beta: Optional[float] = 1.0

    eps: float = 1e-6


class AtlasReg(nn.Module):
    """Characteristic-function regularizer for latent embeddings.

    The biased estimator is non-negative. Its value on a finite Gaussian batch
    is generally positive because the empirical distribution is not identical
    to the population target. The unbiased estimator removes this expected
    floor but can be negative on an individual batch.
    """

    def __init__(self, dim: int, config: Optional[AtlasRegConfig] = None):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.config = config or AtlasRegConfig()
        cfg = self.config
        self._validate_config()

        if cfg.target == "gaussian":
            self.target: Target = StandardGaussian()
        else:
            self.target = StudentT(cfg.student_t_nu, scale=cfg.student_t_scale)

        if cfg.design == "cross_polytope":
            base_design = cross_polytope(dim)
            if cfg.deduplicate_antipodes:
                base_design = base_design[:dim]
            self.register_buffer("base_design", base_design, persistent=False)
        elif cfg.design == "simplex":
            self.register_buffer("base_design", simplex(dim), persistent=False)
        else:
            self.base_design = None

        if cfg.subspace_dim == 1 and cfg.one_d_backend == "quadrature":
            if cfg.kernel == "single":
                rule = gaussian_kernel(lambda_=cfg.lambda_, n_knots=cfg.n_knots)
            else:
                rule = two_scale_gaussian_kernel(
                    lambda_1=cfg.lambda_1,
                    lambda_2=cfg.lambda_2,
                    alpha=cfg.alpha,
                    n_knots=cfg.n_knots,
                )
            self.register_buffer("quad_nodes", rule.nodes, persistent=False)
            self.register_buffer(
                "quad_iweights", rule.integration_weights, persistent=False
            )
            if cfg.target == "gaussian":
                target_cf = StandardGaussian().char_fn_1d(rule.nodes)
            else:
                target_cf = StudentT.precompute_char_fn_1d(
                    cfg.student_t_nu,
                    rule.nodes,
                    scale=cfg.student_t_scale,
                )
            self.register_buffer("target_cf", target_cf, persistent=False)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.design not in ("cross_polytope", "simplex", "haar"):
            raise ValueError(f"Unknown design: {cfg.design!r}")
        if cfg.target not in ("gaussian", "student_t"):
            raise ValueError(f"Unknown target: {cfg.target!r}")
        if cfg.kernel not in ("single", "two_scale"):
            raise ValueError(f"Unknown kernel: {cfg.kernel!r}")
        if cfg.one_d_backend not in ("quadrature", "closed_form"):
            raise ValueError(f"Unknown one_d_backend: {cfg.one_d_backend!r}")
        if cfg.estimator not in ("biased", "unbiased"):
            raise ValueError(f"Unknown estimator: {cfg.estimator!r}")
        if not 1 <= cfg.subspace_dim <= self.dim:
            raise ValueError("subspace_dim must lie in [1, dim]")
        if cfg.n_subspaces < 1:
            raise ValueError("n_subspaces must be positive")
        if cfg.n_haar_projections < 1:
            raise ValueError("n_haar_projections must be positive")
        if cfg.eps <= 0:
            raise ValueError("eps must be positive")
        if cfg.student_t_nu <= 0 or cfg.student_t_scale <= 0:
            raise ValueError("Student-t nu and scale must be positive")
        if cfg.one_d_backend == "closed_form" and cfg.target != "gaussian":
            raise ValueError("closed_form 1D backend requires a Gaussian target")
        if cfg.estimator == "unbiased" and (cfg.standardize_1d or cfg.whiten_kd):
            raise ValueError(
                "the unbiased iid formula is invalid after batch-dependent "
                "standardization or whitening"
            )
        if cfg.hz_beta is not None and cfg.hz_beta <= 0:
            raise ValueError("hz_beta must be positive or None")

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim = {self.dim}, got {z.shape[-1]}")
        z = z.reshape(-1, self.dim)
        if z.shape[0] < 1:
            raise ValueError("at least one latent sample is required")
        if self.config.estimator == "unbiased" and z.shape[0] < 2:
            raise ValueError("unbiased estimator requires at least two samples")
        if self.config.subspace_dim == 1:
            return self._forward_1d(z)
        return self._forward_kd(z, self.config.subspace_dim)

    def _get_1d_projections(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        cfg = self.config
        if self.base_design is None:
            projections = random_haar(
                cfg.n_haar_projections,
                self.dim,
                device=device,
                dtype=dtype,
            )
        else:
            projections = self.base_design.to(device=device, dtype=dtype)
        if cfg.rotate:
            projections = projections @ random_rotation(
                self.dim, device=device, dtype=dtype
            )
        return projections

    def _prepare_1d_samples(self, h: Tensor) -> Tensor:
        if not self.config.standardize_1d:
            return h
        h = h - h.mean(dim=0, keepdim=True)
        std = h.std(dim=0, keepdim=True, unbiased=False)
        return h / std.clamp_min(self.config.eps)

    def _forward_1d(self, z: Tensor) -> Tensor:
        cfg = self.config
        projections = self._get_1d_projections(z.device, z.dtype)
        h = self._prepare_1d_samples(z @ projections.t())

        if cfg.one_d_backend == "quadrature":
            per_projection = empirical_cf_discrepancy(
                h,
                self.quad_nodes.to(device=z.device, dtype=z.dtype),
                self.target_cf.to(device=z.device, dtype=z.dtype),
                self.quad_iweights.to(device=z.device, dtype=z.dtype),
                estimator=cfg.estimator,
            )
            return per_projection.mean()

        values = []
        for column in h.unbind(dim=1):
            if cfg.kernel == "single":
                value = (
                    math.sqrt(2.0 * math.pi)
                    * cfg.lambda_
                    * gaussian_bhep_discrepancy(
                        column, beta=cfg.lambda_, estimator=cfg.estimator
                    )
                )
            else:
                value = (
                    cfg.alpha
                    * math.sqrt(2.0 * math.pi)
                    * cfg.lambda_1
                    * gaussian_bhep_discrepancy(
                        column, beta=cfg.lambda_1, estimator=cfg.estimator
                    )
                    + (1.0 - cfg.alpha)
                    * math.sqrt(2.0 * math.pi)
                    * cfg.lambda_2
                    * gaussian_bhep_discrepancy(
                        column, beta=cfg.lambda_2, estimator=cfg.estimator
                    )
                )
            values.append(value)
        return torch.stack(values).mean()

    def _sample_k_frame(
        self,
        device: torch.device,
        dtype: torch.dtype,
        k: int,
    ) -> Tensor:
        matrix = torch.randn(self.dim, k, device=device, dtype=dtype)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return (q * signs.unsqueeze(0)).t()

    def _prepare_kd_samples(self, y: Tensor) -> Tensor:
        if not self.config.whiten_kd:
            return y
        y = y - y.mean(dim=0, keepdim=True)
        n, k = y.shape
        cov = (y.t() @ y) / max(n, 1)
        cov = cov + self.config.eps * torch.eye(
            k, device=y.device, dtype=y.dtype
        )
        chol = torch.linalg.cholesky(cov)
        return torch.linalg.solve_triangular(chol, y.t(), upper=False).t()

    def _forward_kd(self, z: Tensor, k: int) -> Tensor:
        cfg = self.config
        if cfg.target != "gaussian":
            raise NotImplementedError(
                "k-D matching currently supports only a Gaussian target"
            )
        beta = cfg.hz_beta
        if beta is None:
            beta = henze_zirkler_beta(z.shape[0], k)

        values = []
        for _ in range(cfg.n_subspaces):
            frame = self._sample_k_frame(z.device, z.dtype, k)
            y = self._prepare_kd_samples(z @ frame.t())
            values.append(
                gaussian_bhep_discrepancy(
                    y,
                    beta=beta,
                    estimator=cfg.estimator,
                )
            )
        return torch.stack(values).mean()

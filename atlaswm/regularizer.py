"""Structured characteristic-function matching for latent representations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from atlaswm.designs import (
    RotationMode,
    cross_polytope,
    orthogonal_transform,
    random_haar,
    random_k_frame,
    simplex,
)
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
    design: DesignName = "cross_polytope"
    n_haar_projections: int = 1024
    rotation_mode: RotationMode = "haar"
    rotate: bool | None = None
    rotation_refresh_steps: int = 1
    resample_during_eval: bool = False
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
    n_knots: int = 33
    t_max: float | None = None

    hz_beta: float | None = 1.0
    eps: float = 1e-6
    projection_chunk_size: int = 256
    frequency_chunk_size: int = 16
    pair_chunk_size: int = 512

    def __post_init__(self) -> None:
        """Translate the original boolean rotation option to the explicit mode."""
        if self.rotate is not None:
            self.rotation_mode = "haar" if self.rotate else "none"


class AtlasReg(nn.Module):
    """Approximate a sliced characteristic-function discrepancy.

    The implementation exposes every approximation explicitly: finite samples,
    finite projection directions, finite frequency quadrature, and optional
    subspace sampling. No finite configuration is represented as an exact test
    of equality between arbitrary distributions.
    """

    def __init__(self, dim: int, config: AtlasRegConfig | None = None):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.config = config or AtlasRegConfig()
        self._validate_config()

        cfg = self.config
        self.target: Target
        if cfg.target == "gaussian":
            self.target = StandardGaussian()
        else:
            self.target = StudentT(cfg.student_t_nu, cfg.student_t_scale)

        if cfg.design == "cross_polytope":
            design = cross_polytope(dim)
            if cfg.deduplicate_antipodes:
                design = design[:dim]
            self.register_buffer("base_design", design, persistent=False)
        elif cfg.design == "simplex":
            self.register_buffer("base_design", simplex(dim), persistent=False)
        else:
            self.base_design = None

        self.register_buffer("cached_rotation", torch.empty(0), persistent=False)
        self.register_buffer(
            "rotation_age",
            torch.tensor(cfg.rotation_refresh_steps, dtype=torch.long),
            persistent=False,
        )

        if cfg.subspace_dim == 1 and cfg.one_d_backend == "quadrature":
            if cfg.kernel == "single":
                rule = gaussian_kernel(
                    lambda_=cfg.lambda_,
                    n_knots=cfg.n_knots,
                    t_max=cfg.t_max,
                )
            else:
                rule = two_scale_gaussian_kernel(
                    lambda_1=cfg.lambda_1,
                    lambda_2=cfg.lambda_2,
                    alpha=cfg.alpha,
                    n_knots=cfg.n_knots,
                    t_max=cfg.t_max,
                )
            self.register_buffer("quad_nodes", rule.nodes, persistent=False)
            self.register_buffer(
                "quad_weights",
                rule.integration_weights,
                persistent=False,
            )
            target_cf = self.target.char_fn_1d(rule.nodes)
            self.register_buffer("target_cf", target_cf, persistent=False)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.design not in ("cross_polytope", "simplex", "haar"):
            raise ValueError(f"unknown design: {cfg.design!r}")
        if cfg.rotation_mode not in ("haar", "signed_permutation", "none"):
            raise ValueError(f"unknown rotation mode: {cfg.rotation_mode!r}")
        if cfg.rotation_refresh_steps < 1:
            raise ValueError("rotation_refresh_steps must be positive")
        if cfg.target not in ("gaussian", "student_t"):
            raise ValueError(f"unknown target: {cfg.target!r}")
        if cfg.kernel not in ("single", "two_scale"):
            raise ValueError(f"unknown kernel: {cfg.kernel!r}")
        if cfg.one_d_backend not in ("quadrature", "closed_form"):
            raise ValueError(f"unknown backend: {cfg.one_d_backend!r}")
        if cfg.estimator not in ("biased", "unbiased"):
            raise ValueError(f"unknown estimator: {cfg.estimator!r}")
        if not 1 <= cfg.subspace_dim <= self.dim:
            raise ValueError("subspace_dim must lie in [1,dim]")
        if cfg.n_subspaces < 1 or cfg.n_haar_projections < 1:
            raise ValueError("projection counts must be positive")
        if cfg.student_t_nu <= 0 or cfg.student_t_scale <= 0:
            raise ValueError("Student-t parameters must be positive")
        if cfg.one_d_backend == "closed_form" and cfg.target != "gaussian":
            raise ValueError("closed_form requires a Gaussian target")
        if cfg.subspace_dim > 1 and cfg.target != "gaussian":
            raise ValueError("k-dimensional matching supports Gaussian targets only")
        if cfg.estimator == "unbiased" and (cfg.standardize_1d or cfg.whiten_kd):
            raise ValueError("unbiased iid estimators are invalid after batch normalization")
        if cfg.hz_beta is not None and cfg.hz_beta <= 0:
            raise ValueError("hz_beta must be positive or None")
        if cfg.eps <= 0:
            raise ValueError("eps must be positive")
        if min(
            cfg.projection_chunk_size,
            cfg.frequency_chunk_size,
            cfg.pair_chunk_size,
        ) < 1:
            raise ValueError("chunk sizes must be positive")

    def reset_randomization(self) -> None:
        """Discard cached random frames, useful before deterministic evaluation."""
        self.cached_rotation = self.cached_rotation.new_empty(0)
        self.rotation_age.fill_(self.config.rotation_refresh_steps)

    def _get_rotation(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        cfg = self.config
        if cfg.rotation_mode == "none":
            return torch.eye(self.dim, device=device, dtype=dtype)
        should_refresh = self.cached_rotation.numel() == 0
        should_refresh = should_refresh or int(self.rotation_age.item()) >= cfg.rotation_refresh_steps
        should_refresh = should_refresh and (self.training or cfg.resample_during_eval or self.cached_rotation.numel() == 0)
        if should_refresh:
            rotation = orthogonal_transform(
                self.dim,
                cfg.rotation_mode,
                device=device,
                dtype=dtype,
            )
            self.cached_rotation = rotation
            self.rotation_age.zero_()
        else:
            self.cached_rotation = self.cached_rotation.to(device=device, dtype=dtype)
        self.rotation_age.add_(1)
        return self.cached_rotation

    def _get_1d_directions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        cfg = self.config
        if self.base_design is None:
            return random_haar(
                cfg.n_haar_projections,
                self.dim,
                device=device,
                dtype=dtype,
            )
        directions = self.base_design.to(device=device, dtype=dtype)
        if cfg.rotation_mode != "none":
            directions = directions @ self._get_rotation(device, dtype)
        return directions

    def _prepare_1d(self, projected: Tensor) -> Tensor:
        if not self.config.standardize_1d:
            return projected
        centered = projected - projected.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False, keepdim=True)
        return centered / scale.clamp_min(self.config.eps)

    def _prepare_kd(self, projected: Tensor) -> Tensor:
        if not self.config.whiten_kd:
            return projected
        centered = projected - projected.mean(dim=0, keepdim=True)
        sample_count, dim = centered.shape
        covariance = centered.t() @ centered / max(sample_count, 1)
        covariance = covariance + self.config.eps * torch.eye(
            dim,
            device=projected.device,
            dtype=projected.dtype,
        )
        cholesky = torch.linalg.cholesky(covariance)
        return torch.linalg.solve_triangular(cholesky, centered.t(), upper=False).t()

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {latent.shape[-1]}")
        latent = latent.reshape(-1, self.dim)
        if latent.shape[0] < 1:
            raise ValueError("at least one latent sample is required")
        if self.config.estimator == "unbiased" and latent.shape[0] < 2:
            raise ValueError("unbiased estimator requires at least two samples")
        if self.config.subspace_dim == 1:
            return self._forward_1d(latent)
        return self._forward_kd(latent)

    def _forward_1d(self, latent: Tensor) -> Tensor:
        cfg = self.config
        directions = self._get_1d_directions(latent.device, latent.dtype)
        total = latent.new_zeros(())
        count = 0
        for start in range(0, directions.shape[0], cfg.projection_chunk_size):
            current = directions[start : start + cfg.projection_chunk_size]
            projected = self._prepare_1d(latent @ current.t())
            if cfg.one_d_backend == "quadrature":
                values = empirical_cf_discrepancy(
                    projected,
                    self.quad_nodes.to(latent),
                    self.target_cf.to(latent),
                    self.quad_weights.to(latent),
                    estimator=cfg.estimator,
                    frequency_chunk_size=cfg.frequency_chunk_size,
                )
            else:
                values_list = []
                for column in projected.unbind(dim=1):
                    if cfg.kernel == "single":
                        value = math.sqrt(2.0 * math.pi) * cfg.lambda_ * gaussian_bhep_discrepancy(
                            column,
                            beta=cfg.lambda_,
                            estimator=cfg.estimator,
                            pair_chunk_size=cfg.pair_chunk_size,
                        )
                    else:
                        first = math.sqrt(2.0 * math.pi) * cfg.lambda_1 * gaussian_bhep_discrepancy(
                            column,
                            beta=cfg.lambda_1,
                            estimator=cfg.estimator,
                            pair_chunk_size=cfg.pair_chunk_size,
                        )
                        second = math.sqrt(2.0 * math.pi) * cfg.lambda_2 * gaussian_bhep_discrepancy(
                            column,
                            beta=cfg.lambda_2,
                            estimator=cfg.estimator,
                            pair_chunk_size=cfg.pair_chunk_size,
                        )
                        value = cfg.alpha * first + (1.0 - cfg.alpha) * second
                    values_list.append(value)
                values = torch.stack(values_list)
            total = total + values.sum()
            count += values.numel()
        return total / count

    def _forward_kd(self, latent: Tensor) -> Tensor:
        cfg = self.config
        if cfg.target != "gaussian":
            raise NotImplementedError("k-dimensional matching supports Gaussian targets only")
        beta = cfg.hz_beta
        if beta is None:
            beta = henze_zirkler_beta(latent.shape[0], cfg.subspace_dim)
        values = []
        for _ in range(cfg.n_subspaces):
            frame = random_k_frame(
                self.dim,
                cfg.subspace_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
            projected = self._prepare_kd(latent @ frame.t())
            values.append(
                gaussian_bhep_discrepancy(
                    projected,
                    beta=beta,
                    estimator=cfg.estimator,
                    pair_chunk_size=cfg.pair_chunk_size,
                )
            )
        return torch.stack(values).mean()

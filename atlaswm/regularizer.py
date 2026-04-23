"""AtlasReg: the core anti-collapse regularizer for AtlasWM.

Generalizes SIGReg (Balestriero & LeCun, 2025) along four orthogonal axes:

  (1) PROJECTION: deterministic spherical t-designs + random rotation,
      replacing M stochastic Haar samples.
  (2) SUBSPACE:   testing in k-dim subspaces (Henze-Zirkler) rather than
      only k=1 (Epps-Pulley).
  (3) KERNEL:     two-scale Gaussian weight for simultaneous body + tail
      sensitivity, at no runtime cost.
  (4) TARGET:     optional isotropic Student-t target for heavy-tail-
      robust matching.

Mathematical picture
--------------------
Cramer-Wold (1936): a distribution is determined by its 1D projections;
a fortiori, by its k-D projections for any k >= 1. So enforcing that
all (or enough) projections of Z match the target's projections is
equivalent to pushing Z toward the target.

Epps-Pulley (1983) test: compare CFs in L^2(w),
    T(h) = int w(t) |phi_N(t) - phi_0(t)|^2 dt
where h = Z u is a 1D projection.

Henze-Zirkler (1990) test: the k-D analogue with closed form
    T_beta = mean_{i,j} exp(-beta^2/2 |Y_i - Y_j|^2)
            - 2 (1+beta^2)^(-k/2) mean_i exp(-beta^2 / (2(1+beta^2)) |Y_i|^2)
            + (1+2 beta^2)^(-k/2)
where Y are standardized k-D projections of Z.

Spherical t-designs: a finite point set on S^(d-1) whose empirical
average equals Haar average for all polynomials of degree <= t. The
cross-polytope {+-e_i} is a 3-design. Composed with a random rotation
per step, this gives axis-alignment-free deterministic matching of
moments up to order 3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from atlaswm.designs import (
    cross_polytope,
    simplex,
    random_haar,
    random_rotation,
)
from atlaswm.targets import Target, StandardGaussian, StudentT
from atlaswm.kernels import (
    QuadratureRule,
    gaussian_kernel,
    two_scale_gaussian_kernel,
)


DesignName = Literal['cross_polytope', 'simplex', 'haar']
KernelName = Literal['single', 'two_scale']
TargetName = Literal['gaussian', 'student_t']


@dataclass
class AtlasRegConfig:
    """Configuration for AtlasReg.

    Fields are documented in README.md and docs/theory.md. Defaults are
    chosen to match the paper for the LeWM-equivalent setup: 1D
    projections, isotropic Gaussian target, two-scale kernel, and
    cross-polytope design with rotation.
    """

    # --- Projection ------------------------------------------------------
    design: DesignName = 'cross_polytope'
    n_haar_projections: int = 1024  # only used when design='haar'
    rotate: bool = True

    # --- Subspace dimension ---------------------------------------------
    subspace_dim: int = 1

    # --- Target ---------------------------------------------------------
    target: TargetName = 'gaussian'
    student_t_nu: float = 5.0  # only if target='student_t'

    # --- Kernel (1D case) -----------------------------------------------
    kernel: KernelName = 'two_scale'
    lambda_: float = 1.0
    lambda_1: float = 0.5
    lambda_2: float = 2.0
    alpha: float = 0.5
    n_knots: int = 17

    # --- Henze-Zirkler (k-D case) ---------------------------------------
    hz_beta: float = 1.0

    # --- Numerical stability --------------------------------------------
    eps: float = 1e-6


class AtlasReg(nn.Module):
    """Atlas anti-collapse regularizer.

    Forward pass takes latent embeddings Z and returns a scalar loss
    that is 0 iff Z is distributed as the target.

    Example:
        >>> reg = AtlasReg(dim=192)
        >>> z = torch.randn(256, 192)
        >>> loss = reg(z)
        >>> loss.backward()
    """

    def __init__(
        self,
        dim: int,
        config: Optional[AtlasRegConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.config = config or AtlasRegConfig()
        cfg = self.config

        # Resolve target
        if cfg.target == 'gaussian':
            self.target: Target = StandardGaussian()
        elif cfg.target == 'student_t':
            self.target = StudentT(cfg.student_t_nu)
        else:
            raise ValueError(f"Unknown target: {cfg.target!r}")

        # Precompute fixed design (for deterministic schemes)
        if cfg.design == 'cross_polytope':
            base_design = cross_polytope(dim)
            self.register_buffer('base_design', base_design, persistent=False)
        elif cfg.design == 'simplex':
            base_design = simplex(dim)
            self.register_buffer('base_design', base_design, persistent=False)
        elif cfg.design == 'haar':
            # Resampled each call; no buffer
            self.base_design = None
        else:
            raise ValueError(f"Unknown design: {cfg.design!r}")

        # Precompute kernel and target CF at quadrature nodes (1D case)
        if cfg.subspace_dim == 1:
            if cfg.kernel == 'single':
                rule = gaussian_kernel(
                    lambda_=cfg.lambda_, n_knots=cfg.n_knots
                )
            elif cfg.kernel == 'two_scale':
                rule = two_scale_gaussian_kernel(
                    lambda_1=cfg.lambda_1,
                    lambda_2=cfg.lambda_2,
                    alpha=cfg.alpha,
                    n_knots=cfg.n_knots,
                )
            else:
                raise ValueError(f"Unknown kernel: {cfg.kernel!r}")

            self.register_buffer('quad_nodes', rule.nodes, persistent=False)
            self.register_buffer(
                'quad_iweights',
                rule.integration_weights,
                persistent=False,
            )

            # Precompute target CF at nodes
            if cfg.target == 'gaussian':
                target_cf = StandardGaussian().char_fn_1d(rule.nodes)
            elif cfg.target == 'student_t':
                target_cf = StudentT.precompute_char_fn_1d(
                    cfg.student_t_nu, rule.nodes
                )
            else:
                raise ValueError(f"Unknown target: {cfg.target!r}")
            self.register_buffer('target_cf', target_cf, persistent=False)
        elif cfg.subspace_dim < 1:
            raise ValueError("subspace_dim must be >= 1")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """Compute the Atlas regularizer loss.

        Args:
            z: Latent embeddings of shape (..., d). Non-batch dimensions
                are flattened together. E.g., (B, T, d) -> treated as
                B*T samples of dim d.

        Returns:
            Scalar loss (zero-dim tensor).
        """
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim = {self.dim}, got {z.shape[-1]}"
            )
        z = z.reshape(-1, self.dim)  # (N, d)
        k = self.config.subspace_dim
        if k == 1:
            return self._forward_1d(z)
        else:
            return self._forward_kd(z, k=k)

    # ------------------------------------------------------------------
    # 1D (Epps-Pulley) path
    # ------------------------------------------------------------------

    def _get_1d_projections(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Build the current 1D projection matrix U of shape (M, d)."""
        cfg = self.config
        if self.base_design is not None:
            U = self.base_design.to(device=device, dtype=dtype)
        else:
            U = random_haar(
                cfg.n_haar_projections,
                self.dim,
                device=device,
                dtype=dtype,
            )
        if cfg.rotate:
            R = random_rotation(self.dim, device=device, dtype=dtype)
            U = U @ R
        return U

    def _forward_1d(self, z: Tensor) -> Tensor:
        """Epps-Pulley-style loss on 1D projections.

        Steps:
          1. Project: H = Z U^T, shape (N, M).
          2. Standardize each column of H to zero mean, unit std.
          3. Evaluate empirical CF phi_N(t_i) at quadrature nodes.
          4. Compute |phi_N(t_i) - phi_0(t_i)|^2.
          5. Quadrature sum, average over projections.

        Standardization inside the regularizer means we're testing
        *distributional shape* isotropy, decoupled from the overall
        scale. The encoder is free to use any scale.
        """
        cfg = self.config
        eps = cfg.eps
        device, dtype = z.device, z.dtype

        U = self._get_1d_projections(device, dtype)  # (M, d)
        H = z @ U.t()  # (N, M)

        # Standardize: zero mean, unit variance per projection.
        H = H - H.mean(dim=0, keepdim=True)
        H = H / (H.std(dim=0, keepdim=True) + eps)

        # Empirical CF at quadrature nodes.
        #   phi_N(t) = (1/N) sum_n exp(i t h_n)
        # Broadcasting: t (T,) x H (N, M) -> tH (T, N, M)
        t = self.quad_nodes.to(device=device, dtype=dtype)
        tH = t.view(-1, 1, 1) * H.unsqueeze(0)  # (T, N, M)
        re = tH.cos().mean(dim=1)  # (T, M)
        im = tH.sin().mean(dim=1)  # (T, M)

        # |phi_N - phi_0|^2 = (Re(phi_N) - phi_0_real)^2 + Im(phi_N)^2
        # (target is real since it's symmetric)
        target = self.target_cf.to(device=device, dtype=dtype).view(-1, 1)
        sq_diff = (re - target) ** 2 + im ** 2  # (T, M)

        # Quadrature + average over projections
        iweights = self.quad_iweights.to(device=device, dtype=dtype).view(-1, 1)
        per_proj = (iweights * sq_diff).sum(dim=0)  # (M,)
        return per_proj.mean()

    # ------------------------------------------------------------------
    # k-D (Henze-Zirkler) path
    # ------------------------------------------------------------------

    def _sample_k_frame(
        self, device: torch.device, dtype: torch.dtype, k: int
    ) -> Tensor:
        """Sample k orthonormal vectors in R^d (a k-frame on the Stiefel mfd).

        Uses QR of a random Gaussian (d, k) matrix, which gives a uniform
        sample on the Stiefel manifold V_k(R^d).

        Returns:
            Tensor of shape (k, d).
        """
        A = torch.randn(k, self.dim, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A.t(), mode='reduced')  # (d, k)
        return Q.t()

    def _forward_kd(self, z: Tensor, k: int) -> Tensor:
        """Henze-Zirkler statistic on a random k-D subspace.

        Uses the closed-form expression for the HZ test vs.
        multivariate standard normal (no quadrature needed):

            T_beta = E_{i,j}[exp(-beta^2/2 |Y_i - Y_j|^2)]
                     - 2 (1+beta^2)^(-k/2) E_i[exp(-b^2/(2(1+b^2)) |Y_i|^2)]
                     + (1+2 beta^2)^(-k/2)

        where Y are the whitened k-D projections of Z.

        Only Gaussian target is supported in k-D (>1); for Student-t,
        use subspace_dim=1.
        """
        if self.config.target != 'gaussian':
            raise NotImplementedError(
                "k-D subspace testing currently supports only the Gaussian "
                "target. Set subspace_dim=1 for the Student-t target."
            )

        cfg = self.config
        device, dtype = z.device, z.dtype

        U = self._sample_k_frame(device, dtype, k)  # (k, d)
        Y = z @ U.t()  # (N, k)

        # Whiten: center and unit-covariance.
        Y = Y - Y.mean(dim=0, keepdim=True)
        N = Y.shape[0]
        cov = (Y.t() @ Y) / max(N - 1, 1)
        cov = cov + cfg.eps * torch.eye(k, device=device, dtype=dtype)
        L = torch.linalg.cholesky(cov)
        # Y @ L^{-T}  (row-wise whitening)
        Y = torch.linalg.solve_triangular(L, Y.t(), upper=False).t()

        beta = cfg.hz_beta
        b2 = beta * beta

        # Pairwise squared distances via the polarization identity.
        sq_norms = (Y * Y).sum(dim=-1)  # (N,)
        dots = Y @ Y.t()  # (N, N)
        sq_dists = (
            sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * dots
        ).clamp_min(0.0)

        term1 = torch.exp(-0.5 * b2 * sq_dists).mean()
        coef2 = (1.0 + b2) ** (-k / 2.0)
        term2 = 2.0 * coef2 * torch.exp(
            -b2 / (2.0 * (1.0 + b2)) * sq_norms
        ).mean()
        term3 = (1.0 + 2.0 * b2) ** (-k / 2.0)

        return term1 - term2 + term3

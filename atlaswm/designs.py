"""Spherical t-designs for deterministic distribution matching.

A spherical t-design is a finite set of points {u_i} on the unit sphere
S^(d-1) whose empirical average equals the uniform (Haar) average over
S^(d-1) for all polynomials of degree <= t:

    (1/M) sum_i p(u_i)  =  int_{S^(d-1)} p(u) dsigma(u)   for all polys p with deg(p) <= t

For our use case (Epps-Pulley / Henze-Zirkler regularizers), a t-design
gives *exact* matching of moments up to order t of the 1D projections'
distribution, rather than the O(1/sqrt(M)) Monte Carlo rate.

This module provides three designs:
  - cross_polytope: 2d points, is a spherical 3-design
  - simplex:         d+1 points, is a spherical 2-design
  - random_haar:     arbitrary M points, is a stochastic estimator

Reference:
  Delsarte, Goethals, Seidel. "Spherical codes and designs." Geom. Dedicata 6, 1977.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def cross_polytope(
    dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Cross-polytope vertices in R^d: the 2d points {+-e_i}.

    This set is a spherical 3-design. The sign-flip symmetry automatically
    annihilates all odd moments, and the {+-e_i} set integrates every
    quadratic form against Haar exactly up to the overall factor 1/d.

    Args:
        dim: Ambient dimension d.
        device: Optional torch device.
        dtype: Optional torch dtype.

    Returns:
        Tensor of shape (2*dim, dim). Each row is a unit vector.
    """
    I = torch.eye(dim, device=device, dtype=dtype)
    return torch.cat([I, -I], dim=0)


def simplex(
    dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Regular simplex vertices on S^(d-1): d+1 equidistant unit vectors.

    Forms a spherical 2-design.

    Construction: embed d+1 canonical basis vectors in R^(d+1), subtract
    their centroid (which places them on a d-dim hyperplane), then use
    SVD to obtain d-dimensional coordinates and normalize.

    Args:
        dim: Ambient dimension d.
        device: Optional torch device.
        dtype: Optional torch dtype.

    Returns:
        Tensor of shape (dim+1, dim). Each row is a unit vector.
    """
    n = dim + 1
    E = torch.eye(n, device=device, dtype=dtype)
    centered = E - E.mean(dim=0, keepdim=True)
    # SVD gives d-dim coordinates. Center has rank d in R^(d+1).
    U, S, _ = torch.linalg.svd(centered, full_matrices=False)
    coords = U[:, :dim] * S[:dim]
    coords = coords / coords.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return coords


def random_haar(
    n_points: int,
    dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Uniform random samples on S^(d-1) (Haar measure).

    Baseline for comparison against deterministic designs. Samples
    i.i.d. standard Gaussian vectors and normalizes.

    Args:
        n_points: Number of samples M.
        dim: Ambient dimension d.
        device, dtype: Optional torch options.
        generator: Optional torch random generator.

    Returns:
        Tensor of shape (n_points, dim).
    """
    z = torch.randn(
        n_points, dim, device=device, dtype=dtype, generator=generator
    )
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_rotation(
    dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample a uniform random rotation matrix (Haar measure on O(d)).

    Uses the QR-sign-fix construction of Mezzadri (2007):
    take A ~ Normal(d x d), compute QR = A, then flip signs of columns
    of Q so that diagonal(R) > 0.

    Args:
        dim: Matrix dimension d.
        device, dtype: Optional torch options.
        generator: Optional torch random generator.

    Returns:
        Tensor of shape (dim, dim). Orthogonal.

    Reference:
        Mezzadri (2007). "How to generate random matrices from the classical
        compact groups." Notices of the AMS 54(5).
    """
    A = torch.randn(
        dim, dim, device=device, dtype=dtype, generator=generator
    )
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diagonal(R))
    # Map zeros (measure-zero event) to 1 to avoid NaN
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(0)


def get_design(
    name: str,
    dim: int,
    n_points: Optional[int] = None,
    rotate: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Factory for spherical designs.

    Args:
        name: One of 'cross_polytope', 'simplex', 'haar'.
        dim: Ambient dimension.
        n_points: Required for 'haar'. Ignored for others.
        rotate: If True, compose the design with a random rotation. This
            prevents the encoder from exploiting axis-aligned structure
            when using the cross-polytope.
        device, dtype, generator: Optional torch options.

    Returns:
        Projection matrix of shape (M, dim) where M depends on the design.
    """
    if name == 'cross_polytope':
        U = cross_polytope(dim, device=device, dtype=dtype)
    elif name == 'simplex':
        U = simplex(dim, device=device, dtype=dtype)
    elif name == 'haar':
        if n_points is None:
            raise ValueError("n_points required for haar design")
        U = random_haar(
            n_points, dim, device=device, dtype=dtype, generator=generator
        )
    else:
        raise ValueError(f"Unknown design: {name!r}")

    if rotate:
        R = random_rotation(
            dim, device=device, dtype=dtype, generator=generator
        )
        U = U @ R

    return U

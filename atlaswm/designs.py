"""Projection designs and orthogonal randomization utilities."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

RotationMode = Literal["haar", "signed_permutation", "none"]


def _validate_dim(dim: int) -> None:
    if dim < 1:
        raise ValueError("dim must be positive")


def cross_polytope(
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the ``2d`` vertices ``{+e_i, -e_i}`` of the cross-polytope."""
    _validate_dim(dim)
    eye = torch.eye(dim, device=device, dtype=dtype)
    return torch.cat((eye, -eye), dim=0)


def simplex(
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return ``d+1`` regular-simplex vertices embedded in ``R^d``."""
    _validate_dim(dim)
    n = dim + 1
    eye = torch.eye(n, device=device, dtype=dtype)
    centered = eye - eye.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    points = u[:, :dim] * s[:dim]
    return points / points.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_haar(
    n_points: int,
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample independent uniform directions on the unit sphere."""
    if n_points < 1:
        raise ValueError("n_points must be positive")
    _validate_dim(dim)
    values = torch.randn(
        n_points,
        dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def haar_rotation(
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a Haar-distributed orthogonal matrix using QR sign correction."""
    _validate_dim(dim)
    matrix = torch.randn(
        dim,
        dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def signed_permutation(
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a fast orthogonal signed-permutation matrix.

    This transformation is not Haar distributed. It is useful when exact
    orthogonality and axis randomization are required without dense QR cost.
    """
    _validate_dim(dim)
    permutation = torch.randperm(dim, device=device, generator=generator)
    signs = torch.randint(
        0,
        2,
        (dim,),
        device=device,
        generator=generator,
    ).mul_(2).sub_(1)
    if dtype is not None:
        signs = signs.to(dtype=dtype)
    matrix = torch.zeros(dim, dim, device=device, dtype=dtype)
    matrix[torch.arange(dim, device=device), permutation] = signs
    return matrix


def orthogonal_transform(
    dim: int,
    mode: RotationMode,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return an orthogonal transform for the requested randomization mode."""
    if mode == "haar":
        return haar_rotation(
            dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if mode == "signed_permutation":
        return signed_permutation(
            dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if mode == "none":
        return torch.eye(dim, device=device, dtype=dtype)
    raise ValueError(f"unknown rotation mode: {mode!r}")


def random_k_frame(
    ambient_dim: int,
    subspace_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a Haar-distributed orthonormal ``k``-frame with shape ``(k,d)``."""
    _validate_dim(ambient_dim)
    if not 1 <= subspace_dim <= ambient_dim:
        raise ValueError("subspace_dim must lie in [1, ambient_dim]")
    matrix = torch.randn(
        ambient_dim,
        subspace_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.unsqueeze(0)).t()


# Stable public aliases.
def random_rotation(
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a Haar orthogonal matrix."""
    return haar_rotation(dim, device=device, dtype=dtype, generator=generator)


def get_design(
    name: str,
    dim: int,
    n_points: int | None = None,
    rotate: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct a named projection design with optional Haar rotation."""
    if name == "cross_polytope":
        design = cross_polytope(dim, device=device, dtype=dtype)
    elif name == "simplex":
        design = simplex(dim, device=device, dtype=dtype)
    elif name == "haar":
        if n_points is None:
            raise ValueError("n_points is required for the Haar design")
        design = random_haar(
            n_points, dim, device=device, dtype=dtype, generator=generator
        )
    else:
        raise ValueError(f"unknown design: {name!r}")
    if rotate:
        design = design @ haar_rotation(
            dim, device=device, dtype=dtype, generator=generator
        )
    return design

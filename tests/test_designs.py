"""Tests for atlaswm.designs."""

import torch
import pytest

from atlaswm.designs import (
    cross_polytope,
    simplex,
    random_haar,
    random_rotation,
    get_design,
)


class TestCrossPolytope:
    @pytest.mark.parametrize("d", [2, 4, 16, 192])
    def test_shape_and_count(self, d):
        U = cross_polytope(d)
        assert U.shape == (2 * d, d)

    @pytest.mark.parametrize("d", [2, 16, 192])
    def test_unit_norm(self, d):
        U = cross_polytope(d)
        norms = U.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("d", [4, 16])
    def test_is_two_design(self, d):
        """The empirical average of u u^T over the cross-polytope equals I/d,
        which is the Haar average of u u^T on S^(d-1).
        """
        U = cross_polytope(d, dtype=torch.float64)
        M = U.shape[0]
        outer = torch.einsum('mi,mj->ij', U, U) / M
        expected = torch.eye(d, dtype=torch.float64) / d
        assert torch.allclose(outer, expected, atol=1e-10)

    @pytest.mark.parametrize("d", [4, 8])
    def test_odd_moments_zero(self, d):
        """Cross-polytope has sign-flip symmetry -> all odd moments vanish."""
        U = cross_polytope(d, dtype=torch.float64)
        # First moment
        m1 = U.mean(dim=0)
        assert torch.allclose(m1, torch.zeros(d, dtype=torch.float64), atol=1e-10)
        # Third moment (symmetric tensor)
        m3 = torch.einsum('mi,mj,mk->ijk', U, U, U) / U.shape[0]
        assert torch.allclose(m3, torch.zeros_like(m3), atol=1e-10)


class TestSimplex:
    @pytest.mark.parametrize("d", [2, 4, 16])
    def test_shape_and_count(self, d):
        U = simplex(d)
        assert U.shape == (d + 1, d)

    @pytest.mark.parametrize("d", [2, 4, 16])
    def test_unit_norm(self, d):
        U = simplex(d)
        norms = U.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("d", [3, 4, 8])
    def test_is_two_design(self, d):
        U = simplex(d, dtype=torch.float64)
        M = U.shape[0]
        outer = torch.einsum('mi,mj->ij', U, U) / M
        expected = torch.eye(d, dtype=torch.float64) / d
        assert torch.allclose(outer, expected, atol=1e-10)

    @pytest.mark.parametrize("d", [3, 4, 8])
    def test_equidistant(self, d):
        """All pairwise distances between simplex vertices are equal."""
        U = simplex(d, dtype=torch.float64)
        dists = torch.cdist(U, U)
        # Off-diagonal should all be equal
        mask = ~torch.eye(d + 1, dtype=torch.bool)
        off = dists[mask]
        assert torch.allclose(off, off[0].expand_as(off), atol=1e-10)


class TestHaar:
    @pytest.mark.parametrize("M,d", [(100, 16), (1000, 32)])
    def test_shape(self, M, d):
        U = random_haar(M, d)
        assert U.shape == (M, d)

    @pytest.mark.parametrize("M,d", [(100, 16), (1000, 32)])
    def test_unit_norm(self, M, d):
        U = random_haar(M, d)
        norms = U.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_determinism_with_generator(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        U1 = random_haar(50, 8, generator=g1)
        U2 = random_haar(50, 8, generator=g2)
        assert torch.allclose(U1, U2)


class TestRandomRotation:
    @pytest.mark.parametrize("d", [4, 16, 64])
    def test_orthogonal(self, d):
        R = random_rotation(d, dtype=torch.float64)
        I = torch.eye(d, dtype=torch.float64)
        assert torch.allclose(R @ R.T, I, atol=1e-10)
        assert torch.allclose(R.T @ R, I, atol=1e-10)


class TestGetDesign:
    @pytest.mark.parametrize("name", ['cross_polytope', 'simplex'])
    def test_deterministic_designs_rotation_preserves_norms(self, name):
        U = get_design(name, dim=16, rotate=True, dtype=torch.float64)
        norms = U.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    def test_haar_requires_n_points(self):
        with pytest.raises(ValueError):
            get_design('haar', dim=16)

    def test_unknown_design_raises(self):
        with pytest.raises(ValueError):
            get_design('mystery', dim=16)

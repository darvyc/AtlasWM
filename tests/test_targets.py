"""Tests for atlaswm.targets."""

import math
import torch
import pytest

from atlaswm.targets import StandardGaussian, StudentT


class TestStandardGaussian:
    def test_zero(self):
        g = StandardGaussian()
        assert torch.isclose(g.char_fn_1d(torch.tensor(0.0)), torch.tensor(1.0))

    def test_matches_known_values(self):
        g = StandardGaussian()
        t = torch.tensor([0.0, 0.5, 1.0, 2.0])
        expected = torch.tensor([1.0, math.exp(-0.125), math.exp(-0.5), math.exp(-2.0)])
        assert torch.allclose(g.char_fn_1d(t), expected, atol=1e-6)

    def test_kd_norm_matches_1d(self):
        """For Gaussian, phi(t) = exp(-|t|^2/2) is dimension-independent."""
        g = StandardGaussian()
        t = torch.tensor([0.5, 1.0, 2.0])
        phi1 = g.char_fn_1d(t)
        phi_kd = g.char_fn_kd_norm(t * t, k=4)
        assert torch.allclose(phi1, phi_kd, atol=1e-6)


class TestStudentT:
    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            StudentT(nu=0.0)
        with pytest.raises(ValueError):
            StudentT(nu=-1.0)

    def test_precompute_requires_scipy(self):
        """If scipy is available, precompute returns reasonable values."""
        pytest.importorskip("scipy")
        t = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
        phi = StudentT.precompute_char_fn_1d(nu=5.0, t_nodes=t)
        # phi(0) = 1, phi decreasing in |t|, bounded in [0, 1]
        assert torch.isclose(phi[0], torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
        assert (phi >= 0).all()
        assert (phi <= 1.0 + 1e-6).all()
        assert phi[1] > phi[2] > phi[3]  # monotone decreasing for |t| >= 0

    def test_limit_to_gaussian_as_nu_inf(self):
        """As nu -> infinity, Student-t -> Gaussian."""
        pytest.importorskip("scipy")
        t = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        phi_t = StudentT.precompute_char_fn_1d(nu=1000.0, t_nodes=t)
        phi_g = StandardGaussian().char_fn_1d(t)
        # Should agree to within a few percent
        assert torch.allclose(phi_t, phi_g, rtol=0.05)

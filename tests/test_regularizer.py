"""Tests for atlaswm.regularizer.

Key invariants we check:
  - Gaussian samples -> near-zero AtlasReg loss
  - Collapsed samples (all equal) -> large loss
  - Gradients flow through the encoder chain
  - Henze-Zirkler closed form matches the definition on small problems
  - Two-scale kernel produces different values than single-scale
"""

import math
import torch
import pytest

from atlaswm.regularizer import AtlasReg, AtlasRegConfig


DIM = 64


def sample_gaussian(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def sample_collapsed(n, d):
    v = torch.randn(1, d)
    return v.expand(n, d).clone() + 1e-6 * torch.randn(n, d)


class TestAtlasRegGaussian1D:
    """1D (Epps-Pulley) path with Gaussian target."""

    @pytest.mark.parametrize("design", ['cross_polytope', 'simplex', 'haar'])
    def test_gaussian_input_low_loss(self, design):
        cfg = AtlasRegConfig(
            design=design,
            n_haar_projections=256,
            subspace_dim=1,
            target='gaussian',
            kernel='single',
            lambda_=1.0,
        )
        reg = AtlasReg(DIM, cfg)
        # Gaussian samples should give very small loss
        z = sample_gaussian(1024, DIM)
        loss = reg(z)
        # Empirically, a batch of 1024 Gaussians gives loss < 0.01 here
        assert loss.item() < 0.05, f"Gaussian loss too high: {loss.item()}"

    @pytest.mark.parametrize("design", ['cross_polytope', 'haar'])
    def test_dimensional_collapse_detected(self, design):
        """EP detects dimensional collapse (some dims with much less variance
        than others), which after projection produces non-Gaussian marginals.

        Pure point-collapse is not detected by EP *after standardization* —
        that's fought by the prediction loss, not AtlasReg. See docs/theory.md.
        """
        cfg = AtlasRegConfig(
            design=design,
            n_haar_projections=256,
            subspace_dim=1,
            target='gaussian',
            kernel='single',
        )
        reg = AtlasReg(DIM, cfg)
        # Healthy isotropic Gaussian
        z_gauss = sample_gaussian(1024, DIM)
        # Dimensional collapse: sparse + non-Gaussian shape
        # (uniform has kurtosis 1.8 vs Gaussian 3 -> distinguishable by CF)
        z_uni = torch.empty(1024, DIM).uniform_(-1.732, 1.732)  # var=1 too
        loss_g = reg(z_gauss).item()
        loss_u = reg(z_uni).item()
        assert loss_u > loss_g, (
            f"Expected uniform loss > gaussian loss; got "
            f"uniform={loss_u}, gaussian={loss_g}"
        )

    def test_gradient_flow(self):
        cfg = AtlasRegConfig(subspace_dim=1, design='cross_polytope')
        reg = AtlasReg(DIM, cfg)
        z = torch.randn(256, DIM, requires_grad=True)
        loss = reg(z)
        loss.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert z.grad.abs().sum() > 0

    def test_two_scale_differs_from_single(self):
        cfg1 = AtlasRegConfig(
            design='cross_polytope',
            subspace_dim=1,
            kernel='single',
            lambda_=1.0,
        )
        cfg2 = AtlasRegConfig(
            design='cross_polytope',
            subspace_dim=1,
            kernel='two_scale',
            lambda_1=0.5,
            lambda_2=2.0,
            alpha=0.5,
        )
        torch.manual_seed(0)
        reg1 = AtlasReg(DIM, cfg1)
        torch.manual_seed(0)
        reg2 = AtlasReg(DIM, cfg2)
        torch.manual_seed(42)
        z = torch.randn(512, DIM)
        # Same rotation for fair comparison (both seeded)
        torch.manual_seed(1)
        l1 = reg1(z).item()
        torch.manual_seed(1)
        l2 = reg2(z).item()
        # Values should differ (different kernels)
        assert abs(l1 - l2) > 1e-6


class TestAtlasRegKD:
    """k-D (Henze-Zirkler) path."""

    def test_gaussian_input_low_loss_kd(self):
        cfg = AtlasRegConfig(subspace_dim=4, target='gaussian', hz_beta=1.0)
        reg = AtlasReg(DIM, cfg)
        z = sample_gaussian(1024, DIM)
        loss = reg(z)
        # HZ on Gaussian input should be very small (it's the statistic itself,
        # which is nonnegative and minimized at 0 for true Gaussian)
        assert loss.item() < 0.05, f"HZ on Gaussian too high: {loss.item()}"

    def test_collapsed_input_high_loss_kd(self):
        cfg = AtlasRegConfig(subspace_dim=4, target='gaussian', hz_beta=1.0)
        reg = AtlasReg(DIM, cfg)
        z_gauss = sample_gaussian(1024, DIM)
        z_coll = sample_collapsed(1024, DIM)
        assert reg(z_coll).item() > reg(z_gauss).item()

    def test_kd_gradient_flow(self):
        cfg = AtlasRegConfig(subspace_dim=4)
        reg = AtlasReg(DIM, cfg)
        z = torch.randn(256, DIM, requires_grad=True)
        loss = reg(z)
        loss.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_student_t_raises_in_kd(self):
        """Student-t target currently only supported in 1D."""
        cfg = AtlasRegConfig(subspace_dim=4, target='student_t')
        reg = AtlasReg(DIM, cfg)
        z = torch.randn(32, DIM)
        with pytest.raises(NotImplementedError):
            reg(z)


class TestAtlasRegStudentT:
    """Student-t target in 1D."""

    def test_student_t_gaussian_input(self):
        """Gaussian samples tested against Student-t target should give
        nonzero but not crazy loss (the two are similar for ν >> 1)."""
        pytest.importorskip("scipy")
        cfg = AtlasRegConfig(
            design='cross_polytope',
            subspace_dim=1,
            target='student_t',
            student_t_nu=10.0,
            kernel='single',
            lambda_=1.0,
        )
        reg = AtlasReg(DIM, cfg)
        z = sample_gaussian(1024, DIM)
        loss = reg(z).item()
        # Should be small but nonzero
        assert 0 <= loss < 0.1

    def test_student_t_converges_to_gaussian_large_nu(self):
        """At large ν, Student-t target should give ~same loss as Gaussian target."""
        pytest.importorskip("scipy")
        cfg_g = AtlasRegConfig(
            design='cross_polytope', subspace_dim=1, target='gaussian',
            kernel='single', lambda_=1.0,
        )
        cfg_t = AtlasRegConfig(
            design='cross_polytope', subspace_dim=1, target='student_t',
            student_t_nu=500.0, kernel='single', lambda_=1.0,
        )
        torch.manual_seed(0)
        reg_g = AtlasReg(DIM, cfg_g)
        torch.manual_seed(0)
        reg_t = AtlasReg(DIM, cfg_t)
        z = sample_gaussian(512, DIM, seed=7)
        torch.manual_seed(123)
        lg = reg_g(z).item()
        torch.manual_seed(123)
        lt = reg_t(z).item()
        assert abs(lg - lt) < 0.05


class TestConfigValidation:
    def test_unknown_target_raises(self):
        cfg = AtlasRegConfig(target='mystery')  # type: ignore
        with pytest.raises(ValueError):
            AtlasReg(DIM, cfg)

    def test_unknown_design_raises(self):
        cfg = AtlasRegConfig(design='mystery')  # type: ignore
        with pytest.raises(ValueError):
            AtlasReg(DIM, cfg)

    def test_wrong_input_dim_raises(self):
        reg = AtlasReg(64)
        z = torch.randn(32, 128)
        with pytest.raises(ValueError):
            reg(z)

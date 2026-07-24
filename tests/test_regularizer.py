"""Tests for AtlasReg.

The suite checks target behaviour, collapse sensitivity, gradient flow,
multivariate Gaussian matching, Student-t limits, and configuration validation.
"""

import pytest
import torch

from atlaswm.regularizer import AtlasReg, AtlasRegConfig


DIM = 64


def sample_gaussian(n, d, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=generator, dtype=torch.float32)


def sample_collapsed(n, d):
    vector = torch.randn(1, d)
    return vector.expand(n, d).clone() + 1e-6 * torch.randn(n, d)


class TestAtlasRegGaussian1D:
    """One-dimensional characteristic-function path with Gaussian target."""

    @pytest.mark.parametrize("design", ["cross_polytope", "simplex", "haar"])
    def test_gaussian_input_low_loss(self, design):
        config = AtlasRegConfig(
            design=design,
            n_haar_projections=256,
            subspace_dim=1,
            target="gaussian",
            kernel="single",
            lambda_=1.0,
        )
        regularizer = AtlasReg(DIM, config)
        latent = sample_gaussian(1024, DIM)
        loss = regularizer(latent)
        assert loss.item() < 0.05, f"Gaussian loss too high: {loss.item()}"

    @pytest.mark.parametrize("design", ["cross_polytope", "haar"])
    def test_dimensional_collapse_detected(self, design):
        """Raw Gaussian target matching penalizes a low-rank latent law.

        The collapsed sample contains variance in only four of sixty-four
        coordinates. Both samples are evaluated with identical random projection
        state so the assertion measures the distributions rather than direction
        resampling noise.
        """
        config = AtlasRegConfig(
            design=design,
            n_haar_projections=256,
            subspace_dim=1,
            target="gaussian",
            standardize_1d=False,
            kernel="single",
        )
        regularizer = AtlasReg(DIM, config)
        healthy = sample_gaussian(1024, DIM, seed=0)
        collapsed = torch.zeros(1024, DIM)
        collapsed[:, :4] = sample_gaussian(1024, 4, seed=1)

        torch.manual_seed(123)
        healthy_loss = regularizer(healthy).item()
        torch.manual_seed(123)
        collapsed_loss = regularizer(collapsed).item()

        assert collapsed_loss > healthy_loss + 0.01, (
            "Expected low-rank loss to exceed isotropic Gaussian loss; got "
            f"collapsed={collapsed_loss}, gaussian={healthy_loss}"
        )

    def test_gradient_flow(self):
        config = AtlasRegConfig(subspace_dim=1, design="cross_polytope")
        regularizer = AtlasReg(DIM, config)
        latent = torch.randn(256, DIM, requires_grad=True)
        loss = regularizer(latent)
        loss.backward()
        assert latent.grad is not None
        assert torch.isfinite(latent.grad).all()
        assert latent.grad.abs().sum() > 0

    def test_two_scale_differs_from_single(self):
        single_config = AtlasRegConfig(
            design="cross_polytope",
            subspace_dim=1,
            kernel="single",
            lambda_=1.0,
        )
        two_scale_config = AtlasRegConfig(
            design="cross_polytope",
            subspace_dim=1,
            kernel="two_scale",
            lambda_1=0.5,
            lambda_2=2.0,
            alpha=0.5,
        )
        regularizer_single = AtlasReg(DIM, single_config)
        regularizer_two_scale = AtlasReg(DIM, two_scale_config)
        latent = sample_gaussian(512, DIM, seed=42)

        torch.manual_seed(1)
        single_loss = regularizer_single(latent).item()
        torch.manual_seed(1)
        two_scale_loss = regularizer_two_scale(latent).item()
        assert abs(single_loss - two_scale_loss) > 1e-6


class TestAtlasRegKD:
    """Multivariate Gaussian BHEP path."""

    def test_gaussian_input_low_loss_kd(self):
        config = AtlasRegConfig(subspace_dim=4, target="gaussian", hz_beta=1.0)
        regularizer = AtlasReg(DIM, config)
        latent = sample_gaussian(1024, DIM)
        loss = regularizer(latent)
        assert loss.item() < 0.05, f"BHEP loss on Gaussian too high: {loss.item()}"

    def test_collapsed_input_high_loss_kd(self):
        config = AtlasRegConfig(subspace_dim=4, target="gaussian", hz_beta=1.0)
        regularizer = AtlasReg(DIM, config)
        gaussian = sample_gaussian(1024, DIM)
        collapsed = sample_collapsed(1024, DIM)
        torch.manual_seed(10)
        gaussian_loss = regularizer(gaussian).item()
        torch.manual_seed(10)
        collapsed_loss = regularizer(collapsed).item()
        assert collapsed_loss > gaussian_loss

    def test_kd_gradient_flow(self):
        config = AtlasRegConfig(subspace_dim=4)
        regularizer = AtlasReg(DIM, config)
        latent = torch.randn(256, DIM, requires_grad=True)
        loss = regularizer(latent)
        loss.backward()
        assert latent.grad is not None
        assert torch.isfinite(latent.grad).all()

    def test_student_t_raises_in_kd(self):
        config = AtlasRegConfig(subspace_dim=4, target="student_t")
        regularizer = AtlasReg(DIM, config)
        latent = torch.randn(32, DIM)
        with pytest.raises(NotImplementedError):
            regularizer(latent)


class TestAtlasRegStudentT:
    """Student-t target in the one-dimensional path."""

    def test_student_t_gaussian_input(self):
        pytest.importorskip("scipy")
        config = AtlasRegConfig(
            design="cross_polytope",
            subspace_dim=1,
            target="student_t",
            student_t_nu=10.0,
            kernel="single",
            lambda_=1.0,
        )
        regularizer = AtlasReg(DIM, config)
        latent = sample_gaussian(1024, DIM)
        loss = regularizer(latent).item()
        assert 0 <= loss < 0.1

    def test_student_t_converges_to_gaussian_large_nu(self):
        pytest.importorskip("scipy")
        gaussian_config = AtlasRegConfig(
            design="cross_polytope",
            subspace_dim=1,
            target="gaussian",
            kernel="single",
            lambda_=1.0,
        )
        student_config = AtlasRegConfig(
            design="cross_polytope",
            subspace_dim=1,
            target="student_t",
            student_t_nu=500.0,
            kernel="single",
            lambda_=1.0,
        )
        gaussian_regularizer = AtlasReg(DIM, gaussian_config)
        student_regularizer = AtlasReg(DIM, student_config)
        latent = sample_gaussian(512, DIM, seed=7)

        torch.manual_seed(123)
        gaussian_loss = gaussian_regularizer(latent).item()
        torch.manual_seed(123)
        student_loss = student_regularizer(latent).item()
        assert abs(gaussian_loss - student_loss) < 0.05


class TestConfigValidation:
    def test_unknown_target_raises(self):
        config = AtlasRegConfig(target="mystery")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            AtlasReg(DIM, config)

    def test_unknown_design_raises(self):
        config = AtlasRegConfig(design="mystery")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            AtlasReg(DIM, config)

    def test_wrong_input_dim_raises(self):
        regularizer = AtlasReg(64)
        latent = torch.randn(32, 128)
        with pytest.raises(ValueError):
            regularizer(latent)

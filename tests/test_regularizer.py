import pytest
import torch

from atlaswm.regularizer import AtlasReg, AtlasRegConfig


def test_regularizer_gradients_and_collapse_penalty():
    torch.manual_seed(0)
    regularizer = AtlasReg(
        12,
        AtlasRegConfig(
            rotation_mode="none",
            kernel="single",
            n_knots=17,
            projection_chunk_size=5,
            frequency_chunk_size=4,
        ),
    )
    healthy = torch.randn(128, 12, requires_grad=True)
    collapsed = torch.zeros(128, 12)
    loss = regularizer(healthy)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(healthy.grad).all()
    assert regularizer(collapsed) > loss.detach()


def test_projection_chunking_is_exact():
    torch.manual_seed(2)
    latent = torch.randn(72, 9)
    common = dict(
        rotation_mode="none",
        kernel="single",
        n_knots=21,
        frequency_chunk_size=5,
    )
    first = AtlasReg(9, AtlasRegConfig(**common, projection_chunk_size=9))(latent)
    second = AtlasReg(9, AtlasRegConfig(**common, projection_chunk_size=2))(latent)
    assert torch.allclose(first, second, atol=1e-7, rtol=1e-6)


def test_evaluation_rotation_is_stable_until_reset():
    torch.manual_seed(3)
    regularizer = AtlasReg(
        6,
        AtlasRegConfig(
            rotation_mode="haar",
            rotation_refresh_steps=1,
            resample_during_eval=False,
        ),
    )
    regularizer.eval()
    latent = torch.randn(64, 6)
    first = regularizer(latent)
    second = regularizer(latent)
    assert torch.equal(first, second)
    regularizer.reset_randomization()
    third = regularizer(latent)
    assert not torch.equal(first, third)


def test_configuration_rejects_invalid_unbiased_normalization():
    with pytest.raises(ValueError):
        AtlasReg(8, AtlasRegConfig(estimator="unbiased", standardize_1d=True))
    with pytest.raises(ValueError):
        AtlasReg(8, AtlasRegConfig(subspace_dim=4, target="student_t"))(torch.randn(8, 8))


def test_kd_path_is_finite():
    regularizer = AtlasReg(
        10,
        AtlasRegConfig(
            subspace_dim=3,
            n_subspaces=2,
            whiten_kd=True,
            hz_beta=None,
            pair_chunk_size=16,
        ),
    )
    value = regularizer(torch.randn(32, 10))
    assert torch.isfinite(value)

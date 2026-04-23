"""Integration tests for the full AtlasWM model."""

import torch
import pytest

from atlaswm import AtlasWM, AtlasRegConfig


@pytest.fixture
def small_model():
    """A tiny AtlasWM suitable for unit testing on CPU."""
    return AtlasWM(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        action_dim=2,
        history_length=4,
        encoder_depth=2,
        encoder_heads=2,
        predictor_depth=2,
        predictor_heads=4,
        reg_config=AtlasRegConfig(
            design='cross_polytope',
            subspace_dim=1,
            kernel='single',
            lambda_=1.0,
            n_knots=9,
        ),
    )


def test_forward_encode(small_model):
    obs = torch.randn(2, 4, 3, 32, 32)  # (B, T, C, H, W)
    z = small_model.encode(obs)
    assert z.shape == (2, 4, 32)
    assert torch.isfinite(z).all()


def test_forward_predict(small_model):
    B, T, D = 2, 4, 32
    z = torch.randn(B, T, D)
    a = torch.randn(B, T, 2)
    z_next = small_model.predict(z, a)
    assert z_next.shape == (B, T, D)
    assert torch.isfinite(z_next).all()


def test_training_step(small_model):
    B, T = 2, 4
    obs = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)
    loss_dict = small_model.training_step(obs, actions, lambda_reg=0.1)
    assert 'total' in loss_dict
    assert 'pred' in loss_dict
    assert 'reg' in loss_dict
    assert torch.isfinite(loss_dict['total'])


def test_training_step_backprop(small_model):
    B, T = 2, 4
    obs = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)
    loss_dict = small_model.training_step(obs, actions, lambda_reg=0.1)
    loss_dict['total'].backward()
    # Check at least some encoder params got gradients
    enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in small_model.encoder.parameters()
    )
    pred_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in small_model.predictor.parameters()
    )
    assert enc_grad, "Encoder received no gradients"
    assert pred_grad, "Predictor received no gradients"

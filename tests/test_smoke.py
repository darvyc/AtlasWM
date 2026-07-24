import torch

import atlaswm


def test_public_api_and_end_to_end_gradient():
    model = atlaswm.AtlasWM(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        action_dim=2,
        history_length=3,
        encoder_depth=1,
        encoder_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_dropout=0.0,
        reg_config=atlaswm.AtlasRegConfig(
            rotation_mode="signed_permutation",
            kernel="single",
            n_knots=9,
        ),
    )
    observations = torch.randn(2, 3, 3, 16, 16)
    actions = torch.randn(2, 3, 2)
    losses = model.training_step(observations, actions)
    losses["total"].backward()
    assert atlaswm.__version__ == "2.0.0"
    assert any(parameter.grad is not None for parameter in model.parameters())

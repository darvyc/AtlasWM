import types

import torch

from atlaswm.predictor import Predictor


def test_rollout_conditions_final_token_on_current_action():
    predictor = Predictor(
        embed_dim=4,
        action_dim=2,
        history_length=8,
        depth=0,
        n_heads=1,
    )
    calls = []

    def fake_forward(self, z, actions):
        calls.append(actions.detach().clone())
        return torch.zeros_like(z)

    predictor.forward = types.MethodType(fake_forward, predictor)
    z0 = torch.zeros(1, 2, 4)
    proposed = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predictor.rollout(z0, proposed)

    assert torch.equal(calls[0][0, -1], proposed[0, 0])
    assert torch.equal(calls[1][0, -2], proposed[0, 0])
    assert torch.equal(calls[1][0, -1], proposed[0, 1])


def test_adaln_modulation_is_zero_initialized():
    predictor = Predictor(
        embed_dim=8,
        action_dim=2,
        history_length=3,
        depth=2,
        n_heads=2,
        dropout=0.0,
    )
    for module in predictor.modules():
        if module.__class__.__name__ == "AdaLN":
            assert torch.count_nonzero(module.to_scale_shift.weight) == 0
            assert torch.count_nonzero(module.to_scale_shift.bias) == 0

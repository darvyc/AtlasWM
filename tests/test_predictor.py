import types

import pytest
import torch

from atlaswm.predictor import AdaLN, Predictor


def test_rollout_uses_current_and_historical_actions():
    predictor = Predictor(
        embed_dim=4,
        action_dim=2,
        history_length=8,
        depth=0,
        n_heads=1,
        dropout=0.0,
    )
    calls = []

    def fake_forward(self, latent, actions):
        calls.append(actions.detach().clone())
        return torch.zeros_like(latent)

    predictor.forward = types.MethodType(fake_forward, predictor)
    context = torch.zeros(1, 2, 4)
    history = torch.tensor([[[9.0, 8.0]]])
    proposed = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predictor.rollout(context, proposed, history)
    assert torch.equal(calls[0][0, 0], history[0, 0])
    assert torch.equal(calls[0][0, -1], proposed[0, 0])
    assert torch.equal(calls[1][0, -1], proposed[0, 1])


def test_rollout_rejects_missing_multi_frame_action_history():
    predictor = Predictor(embed_dim=4, action_dim=2, history_length=4, depth=0, n_heads=1)
    with pytest.raises(ValueError, match="context_actions"):
        predictor.rollout(torch.zeros(1, 2, 4), torch.zeros(1, 1, 2))


def test_single_context_needs_no_history():
    predictor = Predictor(embed_dim=4, action_dim=2, history_length=4, depth=1, n_heads=1)
    output = predictor.rollout(torch.zeros(2, 1, 4), torch.zeros(2, 3, 2))
    assert output.shape == (2, 3, 4)


def test_adaln_is_zero_initialized():
    predictor = Predictor(embed_dim=8, action_dim=2, history_length=3, depth=2, n_heads=2)
    for module in predictor.modules():
        if isinstance(module, AdaLN):
            assert torch.count_nonzero(module.modulation.weight) == 0
            assert torch.count_nonzero(module.modulation.bias) == 0

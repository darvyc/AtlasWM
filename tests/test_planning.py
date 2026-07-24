import pytest
import torch
import torch.nn as nn

from atlaswm.planning import CEMPlanner, PlanResult


class LinearPredictor:
    def rollout(self, context_latent, future_actions, context_actions=None):
        del context_actions
        start = context_latent[:, -1:, :]
        increments = future_actions.cumsum(dim=1)
        return start + increments


class LinearWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_dim = 1
        self.history_length = 3
        self.predictor = LinearPredictor()

    def encode(self, observations):
        if observations.ndim == 5:
            return observations.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        return observations.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)


def test_cem_moves_towards_goal_and_returns_metadata():
    model = LinearWorldModel()
    planner = CEMPlanner(
        model,
        horizon=2,
        n_samples=256,
        n_iters=6,
        n_elites=32,
        action_low=-1.0,
        action_high=1.0,
        update_rate=0.7,
    )
    current = torch.zeros(1, 2, 2)
    goal = torch.ones(1, 2, 2)
    result = planner.plan(
        current,
        goal,
        generator=torch.Generator().manual_seed(4),
        return_result=True,
    )
    assert isinstance(result, PlanResult)
    assert result.actions.shape == (2, 1)
    assert result.actions.sum() > 0.5
    assert result.cost < 0.2


def test_planner_rejects_missing_context_actions():
    model = LinearWorldModel()
    planner = CEMPlanner(model, horizon=1, n_samples=8, n_iters=1, n_elites=2)
    context = torch.zeros(2, 1, 2, 2)
    with pytest.raises(ValueError, match="context_actions"):
        planner.plan(torch.zeros(1, 2, 2), torch.ones(1, 2, 2), context_observations=context)

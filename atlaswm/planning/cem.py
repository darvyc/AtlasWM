"""Cross-Entropy Method planning in learned latent dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

CostMode = Literal["terminal_l2", "terminal_cosine"]


@dataclass(frozen=True)
class PlanResult:
    actions: Tensor
    cost: float
    iterations: int
    final_std: Tensor


class CEMPlanner:
    def __init__(
        self,
        model,
        *,
        horizon: int = 5,
        n_samples: int = 512,
        n_iters: int = 8,
        n_elites: int = 64,
        init_std: float = 1.0,
        min_std: float = 1e-3,
        update_rate: float = 0.25,
        action_low: float | Tensor | None = None,
        action_high: float | Tensor | None = None,
        action_penalty: float = 0.0,
        smoothness_penalty: float = 0.0,
        cost_mode: CostMode = "terminal_l2",
        device: torch.device | None = None,
    ):
        if horizon < 1 or n_samples < 2 or n_iters < 1:
            raise ValueError("invalid CEM budget")
        if not 1 <= n_elites <= n_samples:
            raise ValueError("n_elites must lie in [1,n_samples]")
        if init_std <= 0 or min_std <= 0 or not 0 < update_rate <= 1:
            raise ValueError("invalid CEM distribution parameters")
        if action_penalty < 0 or smoothness_penalty < 0:
            raise ValueError("penalties must be non-negative")
        if cost_mode not in ("terminal_l2", "terminal_cosine"):
            raise ValueError(f"unknown cost mode: {cost_mode!r}")
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_elites = n_elites
        self.init_std = init_std
        self.min_std = min_std
        self.update_rate = update_rate
        self.action_low = action_low
        self.action_high = action_high
        self.action_penalty = action_penalty
        self.smoothness_penalty = smoothness_penalty
        self.cost_mode = cost_mode
        self.device = device or next(model.parameters()).device

    def _bound(self, value: float | Tensor | None, default: float, action_dim: int) -> Tensor:
        if value is None:
            return torch.full((action_dim,), default, device=self.device)
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.expand(action_dim)
        if tensor.shape != (action_dim,):
            raise ValueError(f"action bounds must be scalar or shape ({action_dim},)")
        return tensor

    def _cost(self, predicted: Tensor, goal: Tensor, actions: Tensor) -> Tensor:
        terminal = predicted[:, -1]
        if self.cost_mode == "terminal_l2":
            cost = (terminal - goal).square().mean(dim=-1)
        else:
            cost = 1.0 - torch.nn.functional.cosine_similarity(terminal, goal, dim=-1)
        if self.action_penalty:
            cost = cost + self.action_penalty * actions.square().mean(dim=(1, 2))
        if self.smoothness_penalty and actions.shape[1] > 1:
            differences = actions[:, 1:] - actions[:, :-1]
            cost = cost + self.smoothness_penalty * differences.square().mean(dim=(1, 2))
        return cost

    @torch.no_grad()
    def plan(
        self,
        current_observation: Tensor,
        goal_observation: Tensor,
        *,
        context_observations: Tensor | None = None,
        context_actions: Tensor | None = None,
        context_obs: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_result: bool = False,
    ) -> Tensor | PlanResult:
        """Optimize an action sequence.

        A multi-frame context requires the real actions connecting those frames.
        Missing historical actions are rejected rather than silently replaced.
        """
        self.model.eval()
        if context_obs is not None:
            if context_observations is not None:
                raise ValueError("provide only one context observation argument")
            context_observations = context_obs
        if context_observations is None:
            context_observations = current_observation.unsqueeze(0)
        if context_observations.ndim != 4:
            raise ValueError("context_observations must have shape (T,C,H,W)")
        context_length = context_observations.shape[0]
        if context_length > 1 and context_actions is None:
            raise ValueError("context_actions are required for multi-frame context")
        if context_actions is not None:
            expected = (context_length - 1, self.model.action_dim)
            if tuple(context_actions.shape) != expected:
                raise ValueError(f"context_actions must have shape {expected}")

        context = context_observations.unsqueeze(0).to(self.device)
        latent_context = self.model.encode(context)
        goal = self.model.encode(goal_observation.unsqueeze(0).to(self.device))
        historical_actions = None
        if context_actions is not None:
            historical_actions = context_actions.unsqueeze(0).to(self.device)

        action_dim = self.model.action_dim
        low = self._bound(self.action_low, -float("inf"), action_dim)
        high = self._bound(self.action_high, float("inf"), action_dim)
        if torch.any(low >= high):
            raise ValueError("every lower action bound must be below its upper bound")

        mean = torch.zeros(self.horizon, action_dim, device=self.device)
        std = torch.full_like(mean, self.init_std)
        best_cost = float("inf")
        best_actions = mean.clone()

        for _ in range(self.n_iters):
            noise = torch.randn(
                self.n_samples,
                self.horizon,
                action_dim,
                device=self.device,
                generator=generator,
            )
            candidates = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(low, high)
            expanded_context = latent_context.expand(self.n_samples, -1, -1)
            expanded_history = (
                None
                if historical_actions is None
                else historical_actions.expand(self.n_samples, -1, -1)
            )
            predicted = self.model.predictor.rollout(
                expanded_context,
                candidates,
                expanded_history,
            )
            costs = self._cost(predicted, goal.expand(self.n_samples, -1), candidates)
            iteration_best, iteration_index = costs.min(dim=0)
            if float(iteration_best) < best_cost:
                best_cost = float(iteration_best)
                best_actions = candidates[iteration_index].clone()
            elite_indices = costs.topk(self.n_elites, largest=False).indices
            elites = candidates[elite_indices]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(self.min_std)
            mean = (1.0 - self.update_rate) * mean + self.update_rate * elite_mean
            std = (1.0 - self.update_rate) * std + self.update_rate * elite_std

        actions_cpu = best_actions.detach().cpu()
        if return_result:
            return PlanResult(
                actions=actions_cpu,
                cost=best_cost,
                iterations=self.n_iters,
                final_std=std.detach().cpu(),
            )
        return actions_cpu

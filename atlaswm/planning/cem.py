"""Cross-Entropy Method (CEM) planner in the world-model's latent space.

At each planning step:
  1. Encode the current observation and the goal observation.
  2. Sample N candidate action sequences of length H from N(mu, sigma).
  3. Roll them out in latent space via the predictor.
  4. Score each by distance to the goal embedding at the final step.
  5. Take the top-K elites and update (mu, sigma) from their statistics.
  6. Repeat for n_iters iterations.
  7. Return the first action of the optimized sequence (or the full sequence
     for open-loop execution).

Receding-horizon MPC: in closed-loop control, call plan() at each env step
(or every K steps) with the fresh observation.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


class CEMPlanner:
    """CEM planner that operates in AtlasWM latent space.

    Args:
        model: An AtlasWM instance (needs encode() and predictor.rollout()).
        horizon: Planning horizon H (number of actions to optimize).
        n_samples: Number of candidate action sequences per iteration.
        n_iters: Number of CEM iterations.
        n_elites: Top-K elites used to update the sampling distribution.
        init_std: Initial standard deviation of the action sampling distribution.
        action_low, action_high: Box bounds on actions (None = unbounded).
        device: Optional device override.
    """

    def __init__(
        self,
        model,
        horizon: int = 5,
        n_samples: int = 300,
        n_iters: int = 30,
        n_elites: int = 30,
        init_std: float = 1.0,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_elites = n_elites
        self.init_std = init_std
        self.action_low = action_low
        self.action_high = action_high
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def plan(
        self,
        current_obs: Tensor,
        goal_obs: Tensor,
        context_obs: Optional[Tensor] = None,
    ) -> Tensor:
        """Run CEM and return the optimized action sequence.

        Args:
            current_obs: (C, H, W) current observation.
            goal_obs: (C, H, W) goal observation.
            context_obs: Optional (T0, C, H, W) observation history. If None,
                uses current_obs as a single-frame context.

        Returns:
            actions: (horizon, action_dim) optimized action sequence.
        """
        self.model.eval()
        device = self.device
        action_dim = self.model.action_dim
        H = self.horizon
        N = self.n_samples
        K = self.n_elites

        # Encode context and goal
        if context_obs is None:
            context_obs = current_obs.unsqueeze(0)  # (1, C, H, W)
        ctx = context_obs.unsqueeze(0).to(device)  # (1, T0, C, H, W)
        z_ctx = self.model.encode(ctx)  # (1, T0, D)

        goal = goal_obs.unsqueeze(0).to(device)  # (1, C, H, W)
        z_goal = self.model.encode(goal)  # (1, D)

        # Initialize sampling distribution
        mu = torch.zeros(H, action_dim, device=device)
        sigma = torch.full((H, action_dim), self.init_std, device=device)

        for it in range(self.n_iters):
            # Sample candidate action sequences: (N, H, A)
            eps = torch.randn(N, H, action_dim, device=device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            if self.action_low is not None or self.action_high is not None:
                lo = self.action_low if self.action_low is not None else -float('inf')
                hi = self.action_high if self.action_high is not None else float('inf')
                actions = actions.clamp(lo, hi)

            # Batch-expand context: (N, T0, D)
            z_ctx_batch = z_ctx.expand(N, -1, -1)
            z_rollout = self.model.predictor.rollout(z_ctx_batch, actions)
            # (N, H, D) — take the final predicted state
            z_final = z_rollout[:, -1, :]  # (N, D)

            # Cost = squared distance to goal embedding
            costs = ((z_final - z_goal) ** 2).sum(dim=-1)  # (N,)

            # Elites
            _, idx = costs.topk(K, largest=False)
            elite_actions = actions[idx]  # (K, H, A)

            # Update distribution
            mu = elite_actions.mean(dim=0)
            sigma = elite_actions.std(dim=0).clamp_min(1e-4)

        return mu.detach().cpu()

"""Prediction, representation and closed-loop control evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from atlaswm.data import ToyVisualEnv
from atlaswm.diagnostics import latent_diagnostics
from atlaswm.planning import CEMPlanner


def _prediction_metrics(model, latent: Tensor, actions: Tensor, *, max_horizon: int | None) -> dict[str, float]:
    predicted = model.predict(latent, actions)
    result = {
        "one_step_mse": float(
            torch.nn.functional.mse_loss(predicted[:, :-1], latent[:, 1:])
        )
    }
    available = latent.shape[1] - 1
    horizon_limit = available if max_horizon is None else min(max_horizon, available)
    for horizon in range(1, horizon_limit + 1):
        rollout = model.predictor.rollout(latent[:, :1], actions[:, :horizon])
        error = torch.nn.functional.mse_loss(rollout, latent[:, 1 : horizon + 1])
        result[f"rollout_mse_h{horizon}"] = float(error)
    perturbed = actions.clone()
    perturbed[:, 0] = perturbed[:, 0] + 0.1
    changed = model.predict(latent, perturbed)
    result["action_sensitivity"] = float(
        (changed[:, 0] - predicted[:, 0]).norm(dim=-1).mean()
    )
    return result


@torch.no_grad()
def evaluate_prediction_batch(
    model,
    observations: Tensor,
    actions: Tensor,
    *,
    max_horizon: int | None = None,
) -> dict[str, float]:
    model.eval()
    latent = model.encode(observations)
    result = _prediction_metrics(model, latent, actions, max_horizon=max_horizon)
    result.update(latent_diagnostics(latent))
    return result


def _weighted_merge(records: Iterable[tuple[int, dict[str, float]]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)
    for weight, record in records:
        for key, value in record.items():
            totals[key] += weight * value
            weights[key] += weight
    return {key: totals[key] / weights[key] for key in totals}


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    max_horizon: int | None = None,
    max_latent_samples: int = 8192,
) -> dict[str, float]:
    if max_latent_samples < 1:
        raise ValueError("max_latent_samples must be positive")
    model.eval()
    records = []
    latent_samples = []
    latent_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observations = batch[0].to(device)
        actions = batch[1].to(device)
        latent = model.encode(observations)
        records.append(
            (
                observations.shape[0],
                _prediction_metrics(model, latent, actions, max_horizon=max_horizon),
            )
        )
        if latent_count < max_latent_samples:
            flattened = latent.reshape(-1, latent.shape[-1]).cpu()
            remaining = max_latent_samples - latent_count
            latent_samples.append(flattened[:remaining])
            latent_count += min(flattened.shape[0], remaining)
    if not records:
        raise ValueError("evaluation loader produced no batches")
    result = _weighted_merge(records)
    result.update(latent_diagnostics(torch.cat(latent_samples)))
    result["latent_diagnostic_samples"] = float(latent_count)
    return result


def held_out_linear_probe(
    latent: Tensor,
    targets: Tensor,
    *,
    train_fraction: float = 0.8,
    ridge: float = 1e-6,
    seed: int = 0,
) -> dict[str, float]:
    latent = latent.reshape(-1, latent.shape[-1]).double()
    targets = targets.reshape(-1, targets.shape[-1]).double()
    if latent.shape[0] != targets.shape[0] or latent.shape[0] < 3:
        raise ValueError("probe arrays must contain at least three aligned samples")
    if not 0 < train_fraction < 1 or ridge < 0:
        raise ValueError("invalid probe split or ridge value")
    permutation = torch.randperm(
        latent.shape[0], generator=torch.Generator().manual_seed(seed)
    )
    latent = latent[permutation]
    targets = targets[permutation]
    split = min(latent.shape[0] - 1, max(2, int(latent.shape[0] * train_fraction)))
    x_train, x_test = latent[:split], latent[split:]
    y_train, y_test = targets[:split], targets[split:]
    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_train = x_train - x_mean
    x_test = x_test - x_mean
    y_train = y_train - y_mean
    identity = torch.eye(x_train.shape[1], dtype=x_train.dtype, device=x_train.device)
    weights = torch.linalg.solve(
        x_train.t() @ x_train + ridge * identity,
        x_train.t() @ y_train,
    )
    prediction = x_test @ weights + y_mean
    residual = (prediction - y_test).square().sum()
    total = (y_test - y_test.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return {
        "linear_probe_mse": float((prediction - y_test).square().mean()),
        "linear_probe_r2": float(1.0 - residual / total),
        "linear_probe_train_samples": float(x_train.shape[0]),
        "linear_probe_test_samples": float(x_test.shape[0]),
    }


@torch.no_grad()
def evaluate_state_probe_loader(
    model,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    latent_batches = []
    state_batches = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if len(batch) < 3:
            raise ValueError("state-probe evaluation requires a third batch element")
        observations = batch[0].to(device)
        latent_batches.append(model.encode(observations)[:, -1].cpu())
        state_batches.append(batch[2][:, -1].cpu())
    if not latent_batches:
        raise ValueError("state-probe loader produced no batches")
    return held_out_linear_probe(
        torch.cat(latent_batches), torch.cat(state_batches), seed=seed
    )


@torch.no_grad()
def evaluate_toy_control(
    model,
    planner: CEMPlanner,
    *,
    episodes: int = 20,
    max_steps: int = 40,
    seed: int = 0,
) -> dict[str, float]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    successes = 0
    final_distances = []
    steps_to_success = []
    for episode in range(episodes):
        environment = ToyVisualEnv(seed=seed + episode)
        observation, goal_observation = environment.reset()
        observation_history = [observation]
        action_history: list[Tensor] = []
        success = False
        distance = float("inf")
        for step in range(max_steps):
            context_length = min(len(observation_history), model.history_length)
            context = torch.stack(observation_history[-context_length:])
            context_actions = None
            if context_length > 1:
                context_actions = torch.stack(action_history[-(context_length - 1) :])
            sequence = planner.plan(
                observation_history[-1],
                goal_observation,
                context_observations=context,
                context_actions=context_actions,
            )
            action = sequence[0]
            observation, _, success, info = environment.step(action)
            action_history.append(action)
            observation_history.append(observation)
            distance = info["goal_distance"]
            if success:
                successes += 1
                steps_to_success.append(step + 1)
                break
        final_distances.append(distance)
    return {
        "control_success_rate": successes / episodes,
        "control_final_goal_distance": sum(final_distances) / episodes,
        "control_steps_to_success": (
            sum(steps_to_success) / len(steps_to_success)
            if steps_to_success
            else float(max_steps)
        ),
        "control_episodes": float(episodes),
    }

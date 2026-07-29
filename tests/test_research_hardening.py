from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from atlaswm.model import AtlasWM


class RecordingRegularizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, latent):
        self.shapes.append(tuple(latent.shape))
        return latent.square().mean()


def make_model(**kwargs):
    return AtlasWM(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        action_dim=2,
        history_length=4,
        encoder_depth=1,
        encoder_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_dropout=0.0,
        **kwargs,
    )


def test_stop_gradient_removes_gradient_from_target_only_frame():
    torch.manual_seed(7)
    shared = make_model(regularizer=RecordingRegularizer(), target_mode="shared")
    stopped = make_model(regularizer=RecordingRegularizer(), target_mode="stop_gradient")
    stopped.load_state_dict(shared.state_dict(), strict=False)
    observations_shared = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
    observations_stopped = observations_shared.detach().clone().requires_grad_(True)
    actions = torch.randn(2, 2, 2)
    shared.training_step(observations_shared, actions, lambda_reg=0.0)["total"].backward()
    stopped.training_step(observations_stopped, actions, lambda_reg=0.0)["total"].backward()
    assert observations_shared.grad is not None
    assert observations_stopped.grad is not None
    assert observations_shared.grad[:, 1].abs().sum() > 0
    assert observations_stopped.grad[:, 1].abs().sum() == pytest.approx(0.0, abs=1e-12)
    assert observations_stopped.grad[:, 0].abs().sum() > 0


def test_ema_target_updates_by_configured_decay():
    model = make_model(
        regularizer=RecordingRegularizer(),
        target_mode="ema",
        ema_decay=0.5,
    )
    assert model.target_encoder is not None
    target_parameter = next(model.target_encoder.parameters())
    online_parameter = next(model.encoder.parameters())
    original = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(2.0)
    model.update_target_encoder()
    assert torch.allclose(target_parameter, original + 1.0)
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_ema_target_stays_in_eval_mode():
    model = make_model(regularizer=RecordingRegularizer(), target_mode="ema")
    model.train()
    assert model.encoder.training
    assert model.predictor.training
    assert model.target_encoder is not None
    assert not model.target_encoder.training


def test_per_time_regularization_does_not_pool_trajectory_time():
    regularizer = RecordingRegularizer()
    model = make_model(regularizer=regularizer, regularizer_scope="per_time")
    observations = torch.randn(3, 4, 3, 8, 8)
    actions = torch.randn(3, 4, 2)
    model.training_step(observations, actions)
    assert regularizer.shapes == [(3, 8)] * 4


def test_transition_regularization_targets_dynamics_not_only_marginal():
    regularizer = RecordingRegularizer()
    model = make_model(
        regularizer=regularizer,
        regularizer_scope="transition",
        target_mode="stop_gradient",
    )
    observations = torch.randn(3, 4, 3, 8, 8)
    actions = torch.randn(3, 4, 2)
    losses = model.training_step(observations, actions)
    assert regularizer.shapes == [(3, 3, 8)]
    assert losses["reg_marginal"] == 0
    assert losses["reg_transition"] > 0


def load_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "run_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_small_sample_ci_uses_student_t_not_normal_approximation():
    benchmark = load_benchmark_module()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    standard_error = torch.tensor(values).std(unbiased=True).item() / math.sqrt(5)
    assert benchmark.confidence_interval_95(values) == pytest.approx(
        2.776 * standard_error,
        rel=1e-6,
    )
    assert benchmark.confidence_interval_95(values) > 1.96 * standard_error


def test_nested_selection_selects_best_weight_without_test_seed_leakage():
    benchmark = load_benchmark_module()
    records = [
        {"lambda_reg": 0.01, "one_step_mse": 0.8},
        {"lambda_reg": 0.01, "one_step_mse": 1.0},
        {"lambda_reg": 0.1, "one_step_mse": 0.4},
        {"lambda_reg": 0.1, "one_step_mse": 0.6},
    ]
    chosen, summaries = benchmark.select_weight(records, "one_step_mse", "min")
    assert chosen == 0.1
    assert len(summaries) == 2


def base_config():
    return {
        "seed": 1,
        "data": {
            "name": "toy",
            "n_trajectories": 4,
            "traj_length": 4,
            "sub_length": 2,
            "img_size": 8,
            "split": {"train": 0.8, "validation": 0.1},
        },
        "model": {
            "img_size": 8,
            "patch_size": 4,
            "embed_dim": 8,
            "action_dim": 2,
            "history_length": 2,
            "encoder_depth": 1,
            "encoder_heads": 2,
            "predictor_depth": 1,
            "predictor_heads": 2,
        },
        "regularizer": {"name": "none"},
        "trainer": {"epochs": 1, "batch_size": 2, "lambda_reg": 0.0},
        "output": {"dir": "unused"},
    }


def test_config_builds_ema_and_transition_scope():
    from atlaswm.config import build_model, validate_config

    config = base_config()
    config["model"].update(
        {"target_mode": "ema", "ema_decay": 0.99, "regularizer_scope": "transition"}
    )
    validate_config(config)
    model = build_model(config)
    assert model.target_mode == "ema"
    assert model.ema_decay == pytest.approx(0.99)
    assert model.regularizer_scope == "transition"


def test_config_rejects_invalid_target_semantics():
    from atlaswm.config import validate_config

    config = base_config()
    config["model"].update({"target_mode": "ema", "detach_prediction_target": True})
    with pytest.raises(ValueError, match="conflicts"):
        validate_config(config)

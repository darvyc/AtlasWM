from pathlib import Path

import pytest
import torch

from atlaswm.config import build_dataset, build_model, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_public_configs_are_internally_consistent():
    validate_config(load_config(ROOT / "configs/default.yaml"))
    validate_config(load_config(ROOT / "configs/pusht.yaml"))
    validate_config(load_config(ROOT / "configs/hz_4d.yaml"))
    validate_config(load_config(ROOT / "configs/smoke.yaml"))


def test_history_length_mismatch_is_rejected():
    config = load_config(ROOT / "configs/smoke.yaml")
    config["model"]["history_length"] = config["data"]["sub_length"] - 1
    with pytest.raises(ValueError, match="history_length"):
        validate_config(config)


def test_smoke_config_builds_and_executes_training_step():
    config = load_config(ROOT / "configs/smoke.yaml")
    dataset = build_dataset(config)
    model = build_model(config)
    batch = [dataset[index] for index in range(2)]
    observations = torch.stack([item[0] for item in batch])
    actions = torch.stack([item[1] for item in batch])
    losses = model.training_step(observations, actions, lambda_reg=0.1)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())


def test_builds_all_regularizer_baselines():
    config = load_config(ROOT / "configs/smoke.yaml")
    for name in ("none", "covariance", "full_gaussian_mmd"):
        config["regularizer"] = (
            {"name": name, "beta": 1.0}
            if name == "full_gaussian_mmd"
            else {"name": name}
        )
        model = build_model(config)
        observations = torch.randn(2, 3, 3, 16, 16)
        actions = torch.randn(2, 3, 2)
        losses = model.training_step(observations, actions, lambda_reg=0.1)
        assert torch.isfinite(losses["total"])

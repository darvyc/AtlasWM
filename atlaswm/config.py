"""Configuration loading, validation and component construction."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from atlaswm.baselines import CovarianceRegularizer, FullGaussianMMDRegularizer, ZeroRegularizer
from atlaswm.data import (
    ToyEnvConfig,
    ToyTrajectoryDataset,
    TrajectoryArrayDataset,
    TrajectoryManifestDataset,
    TrajectoryNPZDataset,
)
from atlaswm.model import AtlasWM
from atlaswm.regularizer import AtlasRegConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid override {pair!r}; expected key=value")
        key, raw_value = pair.split("=", 1)
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return result


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_update(output[key], value)
        else:
            output[key] = value
    return output


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"seed", "data", "model", "regularizer", "trainer", "output"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    data = config["data"]
    model = config["model"]
    trainer = config["trainer"]
    if data["sub_length"] < 2:
        raise ValueError("data.sub_length must be at least two")
    if model["history_length"] < data["sub_length"]:
        raise ValueError(
            "model.history_length must be at least data.sub_length so every training window fits"
        )
    if model["img_size"] != data["img_size"]:
        raise ValueError("model.img_size and data.img_size must match")
    if model["img_size"] % model["patch_size"]:
        raise ValueError("model.img_size must be divisible by model.patch_size")
    if min(model["embed_dim"], model["action_dim"], model["history_length"]) < 1:
        raise ValueError("model dimensions must be positive")
    if trainer["epochs"] < 1 or trainer["batch_size"] < 1:
        raise ValueError("trainer.epochs and trainer.batch_size must be positive")
    if trainer["lambda_reg"] < 0:
        raise ValueError("trainer.lambda_reg must be non-negative")
    split = data.get("split", {"train": 0.8, "validation": 0.1})
    if split["train"] <= 0 or split["validation"] < 0:
        raise ValueError("invalid data split fractions")
    if split["train"] + split["validation"] >= 1:
        raise ValueError("train and validation split must sum to less than one")


def _regularizer_config(values: dict[str, Any]) -> AtlasRegConfig:
    accepted = AtlasRegConfig.__dataclass_fields__.keys()
    unknown = set(values) - set(accepted) - {"name", "beta"}
    if unknown:
        raise ValueError(f"unknown regularizer fields: {sorted(unknown)}")
    return AtlasRegConfig(**{key: values[key] for key in accepted if key in values})


def build_model(config: dict[str, Any]) -> AtlasWM:
    model = config["model"]
    regularizer_values = config["regularizer"]
    name = regularizer_values.get("name", "atlas")
    regularizer = None
    reg_config = None
    if name == "atlas":
        reg_config = _regularizer_config(regularizer_values)
    elif name == "covariance":
        regularizer = CovarianceRegularizer()
    elif name == "full_gaussian_mmd":
        regularizer = FullGaussianMMDRegularizer(
            beta=float(regularizer_values.get("beta", 1.0)),
            pair_chunk_size=int(regularizer_values.get("pair_chunk_size", 512)),
        )
    elif name == "none":
        regularizer = ZeroRegularizer()
    else:
        raise ValueError(f"unknown regularizer name: {name!r}")
    return AtlasWM(
        img_size=model["img_size"],
        patch_size=model["patch_size"],
        embed_dim=model["embed_dim"],
        action_dim=model["action_dim"],
        history_length=model["history_length"],
        encoder_depth=model["encoder_depth"],
        encoder_heads=model["encoder_heads"],
        predictor_depth=model["predictor_depth"],
        predictor_heads=model["predictor_heads"],
        predictor_dropout=model.get("predictor_dropout", 0.1),
        reg_config=reg_config,
        regularizer=regularizer,
        detach_prediction_target=model.get("detach_prediction_target", False),
    )


def build_dataset(config: dict[str, Any]):
    data = config["data"]
    name = data["name"]
    if name == "toy":
        environment = ToyEnvConfig(
            img_size=data["img_size"],
            action_noise=data.get("action_noise", 0.02),
        )
        return ToyTrajectoryDataset(
            n_trajectories=data["n_trajectories"],
            traj_length=data["traj_length"],
            sub_length=data["sub_length"],
            seed=config["seed"],
            config=environment,
        )
    if name == "trajectory_npy":
        return TrajectoryArrayDataset(
            data["path"],
            data["sub_length"],
            observation_file=data.get("observation_file", "observations.npy"),
            action_file=data.get("action_file", "actions.npy"),
            state_file=data.get("state_file", "states.npy"),
            normalize_images=data.get("normalize_images", True),
            channel_last=data.get("channel_last", False),
        )
    if name == "trajectory_npz":
        return TrajectoryNPZDataset(
            data["path"],
            data["sub_length"],
            obs_key=data.get("obs_key", "obs"),
            action_key=data.get("action_key", "actions"),
            state_key=data.get("state_key"),
            normalize_images=data.get("normalize_images", True),
            channel_last=data.get("channel_last", False),
        )
    if name == "trajectory_manifest":
        return TrajectoryManifestDataset(data["manifest"], data["sub_length"])
    raise ValueError(f"unknown dataset name: {name!r}")

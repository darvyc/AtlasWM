"""Training command-line interface for AtlasWM.

Usage:
    atlaswm-train --config configs/default.yaml
    atlaswm-train --config configs/default.yaml regularizer.subspace_dim=4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from atlaswm.data import ToyTrajectoryDataset, TrajectoryNPZDataset
from atlaswm.model import AtlasWM
from atlaswm.regularizer import AtlasRegConfig
from atlaswm.train import TrainState, train_one_epoch


def _parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override {pair!r} (expected KEY=VALUE)")
        key, value = pair.split("=", 1)
        parsed = yaml.safe_load(value)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parsed
    return out


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` or an explicit PyTorch device string."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict) -> Dataset:
    """Construct the dataset declared by a resolved configuration."""
    data = cfg["data"]
    name = data["name"]
    if name == "toy":
        return ToyTrajectoryDataset(
            n_trajectories=data["n_trajectories"],
            traj_length=data["traj_length"],
            sub_length=data["sub_length"],
            seed=cfg["seed"],
        )
    if name == "trajectory_npz":
        return TrajectoryNPZDataset(
            path=data["path"],
            sub_length=data["sub_length"],
            obs_key=data.get("obs_key", "obs"),
            action_key=data.get("action_key", "actions"),
            state_key=data.get("state_key"),
            normalize_images=data.get("normalize_images", True),
            channel_last=data.get("channel_last", False),
        )
    raise ValueError(
        f"Unknown dataset {name!r}; expected 'toy' or 'trajectory_npz'"
    )


def build_model(cfg: dict) -> AtlasWM:
    """Construct AtlasWM from a resolved configuration dictionary."""
    model_cfg = cfg["model"]
    regularizer_cfg = cfg["regularizer"]
    reg_cfg = AtlasRegConfig(
        design=regularizer_cfg.get("design", "cross_polytope"),
        n_haar_projections=regularizer_cfg.get("n_haar_projections", 1024),
        rotate=regularizer_cfg.get("rotate", True),
        deduplicate_antipodes=regularizer_cfg.get("deduplicate_antipodes", True),
        subspace_dim=regularizer_cfg.get("subspace_dim", 1),
        n_subspaces=regularizer_cfg.get("n_subspaces", 1),
        target=regularizer_cfg.get("target", "gaussian"),
        student_t_nu=regularizer_cfg.get("student_t_nu", 5.0),
        student_t_scale=regularizer_cfg.get("student_t_scale", 1.0),
        standardize_1d=regularizer_cfg.get("standardize_1d", False),
        whiten_kd=regularizer_cfg.get("whiten_kd", False),
        estimator=regularizer_cfg.get("estimator", "biased"),
        one_d_backend=regularizer_cfg.get("one_d_backend", "quadrature"),
        kernel=regularizer_cfg.get("kernel", "two_scale"),
        lambda_=regularizer_cfg.get("lambda_", 1.0),
        lambda_1=regularizer_cfg.get("lambda_1", 0.5),
        lambda_2=regularizer_cfg.get("lambda_2", 2.0),
        alpha=regularizer_cfg.get("alpha", 0.5),
        n_knots=regularizer_cfg.get("n_knots", 17),
        hz_beta=regularizer_cfg.get("hz_beta", 1.0),
        eps=regularizer_cfg.get("eps", 1e-6),
    )
    return AtlasWM(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        action_dim=model_cfg["action_dim"],
        history_length=model_cfg["history_length"],
        encoder_depth=model_cfg["encoder_depth"],
        encoder_heads=model_cfg["encoder_heads"],
        predictor_depth=model_cfg["predictor_depth"],
        predictor_heads=model_cfg["predictor_heads"],
        predictor_dropout=model_cfg.get("predictor_dropout", 0.1),
        reg_config=reg_cfg,
    )


def _package_version() -> str:
    try:
        return importlib.metadata.version("atlaswm")
    except importlib.metadata.PackageNotFoundError:
        return "1.0.0"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(dataset: Dataset, cfg: dict) -> dict[str, Any]:
    if isinstance(dataset, TrajectoryNPZDataset):
        return {
            "kind": "trajectory_npz",
            "path": str(dataset.archive_path),
            "sha256": _sha256_file(dataset.archive_path),
            "trajectories": int(dataset.obs.shape[0]),
            "trajectory_length": int(dataset.obs.shape[1]),
        }

    serialized = json.dumps(
        {"seed": cfg["seed"], "data": cfg["data"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "kind": "toy",
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "trajectories": int(dataset.obs.shape[0]),
        "trajectory_length": int(dataset.obs.shape[1]),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: AtlasWM,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    state: TrainState,
    dataset_fingerprint: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "state": {"step": state.step, "epoch": state.epoch},
        "loss_history": state.loss_history,
        "rng_state": _rng_state(),
        "atlaswm_version": _package_version(),
        "git_commit": _git_commit(),
        "dataset_fingerprint": dataset_fingerprint,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AtlasWM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE overrides.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.overrides:
        _deep_update(cfg, _parse_overrides(args.overrides))

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = resolve_device(cfg["trainer"].get("device", "auto"))
    print(f"Device: {device}")

    dataset = build_dataset(cfg)
    data_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
        generator=data_generator,
    )
    model = build_model(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["trainer"]["lr"],
        weight_decay=cfg["trainer"].get("weight_decay", 0.0),
    )
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(dataset, cfg)

    state = TrainState()
    for epoch in range(cfg["trainer"]["epochs"]):
        print(f"=== Epoch {epoch + 1} / {cfg['trainer']['epochs']} ===")
        train_one_epoch(
            model,
            loader,
            optimizer,
            lambda_reg=cfg["trainer"]["lambda_reg"],
            device=device,
            state=state,
            grad_clip=cfg["trainer"].get("grad_clip", 1.0),
            log_every=cfg["trainer"].get("log_every", 50),
        )
        if (epoch + 1) % cfg["output"].get("save_every", 1) == 0:
            checkpoint = output_dir / f"ckpt_epoch{epoch + 1}.pt"
            _save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                state=state,
                dataset_fingerprint=fingerprint,
            )
            print(f"Saved: {checkpoint}")

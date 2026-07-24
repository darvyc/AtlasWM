"""Training CLI for AtlasWM.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml regularizer.subspace_dim=4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from atlaswm.data import ToyTrajectoryDataset
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
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict):
    data = cfg["data"]
    if data["name"] == "toy":
        return ToyTrajectoryDataset(
            n_trajectories=data["n_trajectories"],
            traj_length=data["traj_length"],
            sub_length=data["sub_length"],
            seed=cfg["seed"],
        )
    raise NotImplementedError(
        f"Dataset {data['name']!r} not included in the toy scaffold."
    )


def build_model(cfg: dict) -> AtlasWM:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AtlasWM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE overrides.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.overrides:
        _deep_update(cfg, _parse_overrides(args.overrides))

    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["trainer"].get("device", "auto"))
    print(f"Device: {device}")

    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "state": {"step": state.step, "epoch": state.epoch},
                },
                checkpoint,
            )
            print(f"Saved: {checkpoint}")

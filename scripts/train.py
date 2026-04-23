"""Training CLI for AtlasWM.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml regularizer.subspace_dim=4

Simple dotted-key overrides are supported: any positional arg of the form
KEY=VALUE where KEY is a dotted path into the config overrides that leaf.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

# Make the package importable when running from source
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswm import AtlasWM, AtlasRegConfig
from atlaswm.data import ToyTrajectoryDataset
from atlaswm.train import TrainState, train_one_epoch


def _parse_overrides(pairs: list[str]) -> dict:
    """Parse KEY=VALUE overrides into a nested dict."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override {pair!r} (expected KEY=VALUE)")
        key, value = pair.split("=", 1)
        # Try to parse value as YAML (gives int/float/bool/str autocasting)
        parsed = yaml.safe_load(value)
        # Navigate into `out` by dotted key
        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = parsed
    return out


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
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
        f"Dataset {data['name']!r} not included in toy scaffold. "
        "Provide your own Dataset or extend this factory."
    )


def build_model(cfg: dict) -> AtlasWM:
    m = cfg["model"]
    r = cfg["regularizer"]
    reg_cfg = AtlasRegConfig(
        design=r.get("design", "cross_polytope"),
        n_haar_projections=r.get("n_haar_projections", 1024),
        rotate=r.get("rotate", True),
        subspace_dim=r.get("subspace_dim", 1),
        target=r.get("target", "gaussian"),
        student_t_nu=r.get("student_t_nu", 5.0),
        kernel=r.get("kernel", "two_scale"),
        lambda_=r.get("lambda_", 1.0),
        lambda_1=r.get("lambda_1", 0.5),
        lambda_2=r.get("lambda_2", 2.0),
        alpha=r.get("alpha", 0.5),
        n_knots=r.get("n_knots", 17),
        hz_beta=r.get("hz_beta", 1.0),
    )
    return AtlasWM(
        img_size=m["img_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        action_dim=m["action_dim"],
        history_length=m["history_length"],
        encoder_depth=m["encoder_depth"],
        encoder_heads=m["encoder_heads"],
        predictor_depth=m["predictor_depth"],
        predictor_heads=m["predictor_heads"],
        predictor_dropout=m.get("predictor_dropout", 0.1),
        reg_config=reg_cfg,
    )


def main():
    parser = argparse.ArgumentParser(description="Train AtlasWM.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE overrides.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.overrides:
        _deep_update(cfg, _parse_overrides(args.overrides))

    torch.manual_seed(cfg["seed"])

    device = resolve_device(cfg["trainer"].get("device", "auto"))
    print(f"Device: {device}")

    print("Building dataset...")
    ds = build_dataset(cfg)
    print(f"  Dataset size: {len(ds)} sub-trajectories")

    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    print("Building model...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["trainer"]["lr"],
        weight_decay=cfg["trainer"].get("weight_decay", 0.0),
    )

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState()
    for epoch in range(cfg["trainer"]["epochs"]):
        print(f"=== Epoch {epoch + 1} / {cfg['trainer']['epochs']} ===")
        train_one_epoch(
            model,
            loader,
            opt,
            lambda_reg=cfg["trainer"]["lambda_reg"],
            device=device,
            state=state,
            grad_clip=cfg["trainer"].get("grad_clip", 1.0),
            log_every=cfg["trainer"].get("log_every", 50),
        )
        if (epoch + 1) % cfg["output"].get("save_every", 1) == 0:
            ckpt = out_dir / f"ckpt_epoch{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                    "state": {"step": state.step, "epoch": state.epoch},
                },
                ckpt,
            )
            print(f"  Saved: {ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    main()

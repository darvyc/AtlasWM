"""Command-line entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from atlaswm.config import (
    build_dataset,
    build_model,
    deep_update,
    load_config,
    parse_overrides,
    validate_config,
)
from atlaswm.data import split_by_trajectory
from atlaswm.evaluation import (
    evaluate_loader,
    evaluate_state_probe_loader,
    evaluate_toy_control,
)
from atlaswm.planning import CEMPlanner
from atlaswm.training import (
    Trainer,
    TrainerConfig,
    dataset_fingerprint,
    load_checkpoint,
    seed_worker,
    set_global_seed,
    system_information,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _split_dataset(dataset, data_config: dict, seed: int):
    split = data_config.get("split", {"train": 0.8, "validation": 0.1})
    if hasattr(dataset, "n_trajectories") and hasattr(dataset, "windows_per_trajectory"):
        return split_by_trajectory(
            dataset,
            train_fraction=float(split["train"]),
            validation_fraction=float(split["validation"]),
            seed=seed,
        )
    total = len(dataset)
    train_size = int(total * split["train"])
    validation_size = int(total * split["validation"])
    test_size = total - train_size - validation_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=generator,
    )


def _loader(dataset, trainer_config: dict, device: torch.device, seed: int, shuffle: bool):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(trainer_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(trainer_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(trainer_config.get("num_workers", 0)) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _scheduler(optimizer, total_steps: int, warmup_steps: int):
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train AtlasWM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.overrides:
        config = deep_update(config, parse_overrides(args.overrides))
    validate_config(config)

    seed = int(config["seed"])
    trainer_values = config["trainer"]
    set_global_seed(seed, deterministic=bool(trainer_values.get("deterministic", False)))
    device = resolve_device(trainer_values.get("device", "auto"))

    dataset = build_dataset(config)
    train_dataset, validation_dataset, test_dataset = _split_dataset(
        dataset,
        config["data"],
        seed,
    )
    train_loader = _loader(train_dataset, trainer_values, device, seed, True)
    validation_loader = _loader(validation_dataset, trainer_values, device, seed + 1, False)
    test_loader = _loader(test_dataset, trainer_values, device, seed + 2, False)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_values["lr"]),
        weight_decay=float(trainer_values.get("weight_decay", 0.0)),
    )
    total_steps = len(train_loader) * int(trainer_values["epochs"])
    scheduler = _scheduler(
        optimizer,
        total_steps,
        int(trainer_values.get("warmup_steps", 0)),
    )
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (output_dir / "system.json").open("w", encoding="utf-8") as handle:
        json.dump(system_information(), handle, indent=2, sort_keys=True)

    fingerprint = dataset_fingerprint(dataset)
    trainer_config = TrainerConfig(
        epochs=int(trainer_values["epochs"]),
        lambda_reg=float(trainer_values["lambda_reg"]),
        grad_clip=trainer_values.get("grad_clip", 1.0),
        amp=bool(trainer_values.get("amp", True)),
        log_every=int(trainer_values.get("log_every", 50)),
        validate_every=int(trainer_values.get("validate_every", 1)),
        checkpoint_every=int(trainer_values.get("checkpoint_every", 1)),
        deterministic=bool(trainer_values.get("deterministic", False)),
    )
    trainer = Trainer(
        model,
        optimizer,
        device=device,
        config=trainer_config,
        output_dir=output_dir,
        resolved_config=config,
        fingerprint=fingerprint,
        scheduler=scheduler,
    )
    state = None
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    final_state = trainer.fit(
        train_loader,
        validation_loader=validation_loader,
        state=state,
    )
    evaluation = evaluate_loader(
        model,
        test_loader,
        device,
        max_horizon=min(config["data"]["sub_length"] - 1, 10),
    )
    evaluation.update(
        {
            "epoch": final_state.epoch,
            "step": final_state.step,
            "parameters": model.parameter_count(),
        }
    )
    try:
        evaluation.update(
            evaluate_state_probe_loader(
                model,
                test_loader,
                device,
                max_batches=config.get("evaluation", {}).get("probe_max_batches"),
            )
        )
    except ValueError:
        pass
    control = config.get("evaluation", {}).get("toy_control", {})
    if config["data"]["name"] == "toy" and control.get("enabled", False):
        planner = CEMPlanner(
            model,
            horizon=int(control.get("horizon", 5)),
            n_samples=int(control.get("n_samples", 256)),
            n_iters=int(control.get("n_iters", 6)),
            n_elites=int(control.get("n_elites", 32)),
            action_low=float(control.get("action_low", -2.0)),
            action_high=float(control.get("action_high", 2.0)),
            smoothness_penalty=float(control.get("smoothness_penalty", 0.0)),
        )
        evaluation.update(
            evaluate_toy_control(
                model,
                planner,
                episodes=int(control.get("episodes", 20)),
                max_steps=int(control.get("max_steps", 40)),
                seed=seed + 1000,
            )
        )
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=True)
    print(json.dumps(evaluation, indent=2, sort_keys=True))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AtlasWM checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    device = resolve_device(config["trainer"].get("device", "auto"))
    dataset = build_dataset(config)
    _, _, test_dataset = _split_dataset(dataset, config["data"], int(config["seed"]))
    loader = _loader(test_dataset, config["trainer"], device, int(config["seed"]) + 2, False)
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, restore_rng=False)
    metrics = evaluate_loader(
        model,
        loader,
        device,
        max_batches=args.max_batches,
        max_horizon=min(config["data"]["sub_length"] - 1, 10),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))

"""Reproducible, resumable training infrastructure."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from atlaswm.diagnostics import latent_diagnostics


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_validation: float = float("inf")
    loss_history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 10
    lambda_reg: float = 0.1
    grad_clip: float | None = 1.0
    amp: bool = True
    log_every: int = 50
    validate_every: int = 1
    checkpoint_every: int = 1
    deterministic: bool = False


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def system_information() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": torch.cuda.device_count(),
        "hostname": platform.node(),
    }


def git_commit() -> str | None:
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


def dataset_fingerprint(dataset) -> dict[str, Any]:
    payload = (
        dataset.fingerprint_payload()
        if hasattr(dataset, "fingerprint_payload")
        else {"kind": type(dataset).__name__, "length": len(dataset)}
    )
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return {"sha256": hashlib.sha256(serialized).hexdigest(), "payload": payload}


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainState,
    resolved_config: dict[str, Any],
    fingerprint: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "state": asdict(state),
        "config": resolved_config,
        "dataset_fingerprint": fingerprint,
        "rng_state": _rng_state(),
        "git_commit": git_commit(),
        "system": system_information(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    restore_rng: bool = True,
) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return TrainState(**payload["state"])


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        config: TrainerConfig,
        output_dir: str | Path,
        resolved_config: dict[str, Any],
        fingerprint: dict[str, Any],
        scheduler=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config = resolved_config
        self.fingerprint = fingerprint
        self.scheduler = scheduler
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.use_amp = config.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.use_amp,
        )

    def train_epoch(self, loader, state: TrainState) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "pred": 0.0, "reg": 0.0}
        examples = 0
        started = time.perf_counter()
        final_latent: Tensor | None = None
        for batch in loader:
            observations = batch[0].to(self.device, non_blocking=True)
            actions = batch[1].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                losses = self.model.training_step(
                    observations,
                    actions,
                    lambda_reg=self.config.lambda_reg,
                )
            self.scaler.scale(losses["total"]).backward()
            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            batch_size = observations.shape[0]
            examples += batch_size
            state.step += 1
            record = {
                key: float(loss.detach())
                for key, loss in losses.items()
            }
            state.loss_history.append({"step": float(state.step), **record})
            for key in totals:
                totals[key] += record[key] * batch_size
            if self.config.log_every and state.step % self.config.log_every == 0:
                self.logger.write(
                    {
                        "type": "train_step",
                        "step": state.step,
                        "epoch": state.epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        **record,
                    }
                )
            final_latent = self.model.encode(observations).detach()
        if not examples:
            raise ValueError("training loader produced no batches")
        elapsed = time.perf_counter() - started
        summary = {key: value / examples for key, value in totals.items()}
        summary["examples_per_second"] = examples / max(elapsed, 1e-12)
        if final_latent is not None:
            summary.update(latent_diagnostics(final_latent))
        return summary

    @torch.no_grad()
    def validate(self, loader) -> dict[str, float]:
        self.model.eval()
        totals = {"total": 0.0, "pred": 0.0, "reg": 0.0}
        examples = 0
        final_latent: Tensor | None = None
        for batch in loader:
            observations = batch[0].to(self.device, non_blocking=True)
            actions = batch[1].to(self.device, non_blocking=True)
            losses = self.model.training_step(
                observations,
                actions,
                lambda_reg=self.config.lambda_reg,
            )
            batch_size = observations.shape[0]
            examples += batch_size
            for key in totals:
                totals[key] += float(losses[key]) * batch_size
            final_latent = self.model.encode(observations)
        if not examples:
            raise ValueError("validation loader produced no batches")
        summary = {key: value / examples for key, value in totals.items()}
        if final_latent is not None:
            summary.update(latent_diagnostics(final_latent))
        return summary

    def fit(
        self,
        train_loader,
        *,
        validation_loader=None,
        state: TrainState | None = None,
    ) -> TrainState:
        state = state or TrainState()
        for _ in range(state.epoch, self.config.epochs):
            train_metrics = self.train_epoch(train_loader, state)
            state.epoch += 1
            self.logger.write(
                {
                    "type": "train_epoch",
                    "epoch": state.epoch,
                    "step": state.step,
                    **train_metrics,
                }
            )
            validation_metrics = None
            if (
                validation_loader is not None
                and state.epoch % self.config.validate_every == 0
            ):
                validation_metrics = self.validate(validation_loader)
                self.logger.write(
                    {
                        "type": "validation_epoch",
                        "epoch": state.epoch,
                        "step": state.step,
                        **validation_metrics,
                    }
                )
                if validation_metrics["total"] < state.best_validation:
                    state.best_validation = validation_metrics["total"]
                    save_checkpoint(
                        self.output_dir / "checkpoint_best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        state=state,
                        resolved_config=self.resolved_config,
                        fingerprint=self.fingerprint,
                    )
            if state.epoch % self.config.checkpoint_every == 0:
                save_checkpoint(
                    self.output_dir / "checkpoint_last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    state=state,
                    resolved_config=self.resolved_config,
                    fingerprint=self.fingerprint,
                )
        return state


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_reg: float,
    device: torch.device,
    state: TrainState,
    grad_clip: float | None = 1.0,
    log_every: int = 50,
    on_step: Callable[[TrainState, dict[str, Tensor]], None] | None = None,
) -> None:
    """Compatibility wrapper for small examples."""
    model.train()
    for batch in loader:
        observations, actions = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_step(observations, actions, lambda_reg=lambda_reg)
        losses["total"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        state.step += 1
        state.loss_history.append(
            {"step": float(state.step), **{key: float(value) for key, value in losses.items()}}
        )
        if on_step is not None:
            on_step(state, losses)
        if log_every and state.step % log_every == 0:
            print(state.loss_history[-1])
    state.epoch += 1

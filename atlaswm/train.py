"""Minimal training loop for AtlasWM.

This is intentionally light. For serious experimentation, use a proper
training framework (Lightning, Accelerate, etc.) and plug the model in.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    loss_history: list = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_reg: float,
    device: torch.device,
    state: TrainState,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 50,
    on_step: Optional[Callable[[TrainState, dict], None]] = None,
) -> None:
    """Run one epoch of training.

    Args:
        model: AtlasWM instance.
        loader: DataLoader yielding (obs, actions, ...extras) batches.
        optimizer: Torch optimizer.
        lambda_reg: AtlasReg weight.
        device: Device for training.
        state: Mutable training state.
        grad_clip: Max-norm for gradient clipping, or None to disable.
        log_every: Print loss every N steps.
        on_step: Optional callback(state, loss_dict).
    """
    model.train()
    for batch in loader:
        obs, actions = batch[0].to(device), batch[1].to(device)
        loss_dict = model.training_step(obs, actions, lambda_reg=lambda_reg)
        optimizer.zero_grad(set_to_none=True)
        loss_dict['total'].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        state.step += 1
        state.loss_history.append({
            'step': state.step,
            'total': float(loss_dict['total'].item()),
            'pred': float(loss_dict['pred'].item()),
            'reg': float(loss_dict['reg'].item()),
        })
        if on_step is not None:
            on_step(state, loss_dict)
        if log_every > 0 and state.step % log_every == 0:
            last = state.loss_history[-1]
            print(
                f"[step {state.step:6d}] "
                f"total={last['total']:.4f}  "
                f"pred={last['pred']:.4f}  "
                f"reg={last['reg']:.4f}"
            )
    state.epoch += 1

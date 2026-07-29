"""End-to-end AtlasWM model with explicit target-network and temporal regularization controls."""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atlaswm.encoder import ViTEncoder
from atlaswm.predictor import Predictor
from atlaswm.regularizer import AtlasReg, AtlasRegConfig

TargetMode = Literal["shared", "stop_gradient", "ema"]
RegularizerScope = Literal["marginal", "per_time", "transition", "marginal_transition"]


class AtlasWM(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 192,
        action_dim: int = 2,
        history_length: int = 8,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dropout: float = 0.1,
        reg_config: AtlasRegConfig | None = None,
        regularizer: nn.Module | None = None,
        detach_prediction_target: bool = False,
        target_mode: TargetMode = "shared",
        ema_decay: float = 0.996,
        regularizer_scope: RegularizerScope = "marginal",
    ):
        super().__init__()
        if regularizer is not None and reg_config is not None:
            raise ValueError("provide either regularizer or reg_config, not both")
        if target_mode not in ("shared", "stop_gradient", "ema"):
            raise ValueError(f"unknown target_mode: {target_mode!r}")
        if detach_prediction_target:
            if target_mode not in ("shared", "stop_gradient"):
                raise ValueError("detach_prediction_target conflicts with target_mode='ema'")
            target_mode = "stop_gradient"
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0,1)")
        if regularizer_scope not in (
            "marginal",
            "per_time",
            "transition",
            "marginal_transition",
        ):
            raise ValueError(f"unknown regularizer_scope: {regularizer_scope!r}")

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.target_mode: TargetMode = target_mode
        self.ema_decay = float(ema_decay)
        self.regularizer_scope: RegularizerScope = regularizer_scope
        self.detach_prediction_target = target_mode != "shared"
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
        )
        self.predictor = Predictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            history_length=history_length,
            depth=predictor_depth,
            n_heads=predictor_heads,
            dropout=predictor_dropout,
        )
        self.regularizer = regularizer or AtlasReg(embed_dim, reg_config)
        self.target_encoder: ViTEncoder | None = None
        if target_mode == "ema":
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder.requires_grad_(False)
            self.target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def encode(self, observations: Tensor) -> Tensor:
        return self.encoder(observations)

    def predict(self, latent: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(latent, actions)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update an EMA target encoder from the current online encoder."""
        if self.target_encoder is None:
            return
        one_minus_decay = 1.0 - self.ema_decay
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(self.ema_decay).add_(online, alpha=one_minus_decay)
        for target, online in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            if target.dtype.is_floating_point:
                target.mul_(self.ema_decay).add_(online, alpha=one_minus_decay)
            else:
                target.copy_(online)

    def _prediction_target(self, observations: Tensor, online_latent: Tensor) -> Tensor:
        if self.target_mode == "shared":
            return online_latent[:, 1:]
        if self.target_mode == "stop_gradient":
            return online_latent[:, 1:].detach()
        if self.target_encoder is None:
            raise RuntimeError("EMA target mode requires a target encoder")
        with torch.no_grad():
            return self.target_encoder(observations[:, 1:])

    def _regularizer_terms(self, latent: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        zero = latent.new_zeros(())
        marginal = zero
        transition = zero
        if self.regularizer_scope in ("marginal", "marginal_transition"):
            marginal = self.regularizer(latent)
        elif self.regularizer_scope == "per_time":
            marginal = torch.stack(
                [self.regularizer(latent[:, index]) for index in range(latent.shape[1])]
            ).mean()
        if self.regularizer_scope in ("transition", "marginal_transition"):
            transition_delta = target - latent[:, :-1]
            transition = self.regularizer(transition_delta)
        if self.regularizer_scope == "marginal_transition":
            aggregate = 0.5 * (marginal + transition)
        else:
            aggregate = marginal + transition
        return aggregate, marginal, transition

    def training_step(
        self,
        observations: Tensor,
        actions: Tensor,
        lambda_reg: float = 0.1,
    ) -> dict[str, Tensor]:
        if observations.ndim != 5:
            raise ValueError("observations must have shape (B,T,C,H,W)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B,T,A)")
        if observations.shape[:2] != actions.shape[:2]:
            raise ValueError("observations and actions must share batch/time dimensions")
        if actions.shape[-1] != self.action_dim:
            raise ValueError("action dimension mismatch")
        if observations.shape[1] < 2:
            raise ValueError("training trajectories require at least two frames")
        if observations.shape[1] > self.history_length:
            raise ValueError(
                f"trajectory length {observations.shape[1]} exceeds history_length {self.history_length}"
            )
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")

        if self.training and self.target_mode == "ema":
            self.update_target_encoder()
        latent = self.encoder(observations)
        predicted = self.predictor(latent, actions)
        target = self._prediction_target(observations, latent)
        prediction_loss = F.mse_loss(predicted[:, :-1], target)
        regularizer_loss, marginal_loss, transition_loss = self._regularizer_terms(latent, target)
        total = prediction_loss + lambda_reg * regularizer_loss
        return {
            "total": total,
            "pred": prediction_loss,
            "reg": regularizer_loss,
            "reg_marginal": marginal_loss,
            "reg_transition": transition_loss,
        }

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

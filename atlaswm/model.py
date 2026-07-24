"""End-to-end AtlasWM model."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atlaswm.encoder import ViTEncoder
from atlaswm.predictor import Predictor
from atlaswm.regularizer import AtlasReg, AtlasRegConfig


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
    ):
        super().__init__()
        if regularizer is not None and reg_config is not None:
            raise ValueError("provide either regularizer or reg_config, not both")
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.detach_prediction_target = detach_prediction_target
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

    def encode(self, observations: Tensor) -> Tensor:
        return self.encoder(observations)

    def predict(self, latent: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(latent, actions)

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

        latent = self.encoder(observations)
        predicted = self.predictor(latent, actions)
        target = latent[:, 1:]
        if self.detach_prediction_target:
            target = target.detach()
        prediction_loss = F.mse_loss(predicted[:, :-1], target)
        regularizer_loss = self.regularizer(latent)
        total = prediction_loss + lambda_reg * regularizer_loss
        return {
            "total": total,
            "pred": prediction_loss,
            "reg": regularizer_loss,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

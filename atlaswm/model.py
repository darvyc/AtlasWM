"""End-to-end joint-embedding predictive world model."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atlaswm.encoder import ViTEncoder
from atlaswm.predictor import Predictor
from atlaswm.regularizer import AtlasReg, AtlasRegConfig


class AtlasWM(nn.Module):
    """Compose the image encoder, action-conditioned predictor, and AtlasReg.

    The objective is

        L = L_pred + lambda_reg * L_reg.

    Because both branches of ``L_pred`` are trainable, a constant encoder and
    constant predictor are a zero-prediction-loss solution. AtlasReg is the
    component that makes that collapsed empirical distribution disagree with a
    non-degenerate target.
    """

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
        reg_config: Optional[AtlasRegConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
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
        self.regularizer = AtlasReg(embed_dim, reg_config)

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observations shaped (B,C,H,W) or (B,T,C,H,W)."""
        return self.encoder(obs)

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """Predict the next latent at each teacher-forced position."""
        return self.predictor(z, actions)

    def training_step(
        self,
        obs: Tensor,
        actions: Tensor,
        lambda_reg: float = 0.1,
    ) -> dict[str, Tensor]:
        """Compute prediction, regularization, and total losses."""
        if obs.dim() != 5:
            raise ValueError("obs must have shape (B, T, C, H, W)")
        if actions.dim() != 3:
            raise ValueError("actions must have shape (B, T, A)")
        if obs.shape[:2] != actions.shape[:2]:
            raise ValueError("obs and actions must share batch and time dimensions")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action dim {self.action_dim}, got {actions.shape[-1]}"
            )
        if obs.shape[1] < 2:
            raise ValueError("training trajectories require at least two frames")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")

        z = self.encoder(obs)
        z_next_pred = self.predictor(z, actions)
        pred_loss = F.mse_loss(z_next_pred[:, :-1], z[:, 1:])
        reg_loss = self.regularizer(z)
        total = pred_loss + lambda_reg * reg_loss
        return {"total": total, "pred": pred_loss, "reg": reg_loss}

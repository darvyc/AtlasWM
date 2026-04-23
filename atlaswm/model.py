"""AtlasWM: end-to-end joint-embedding predictive world model.

Composes the ViT encoder, the causal transformer predictor, and the
AtlasReg regularizer into a single trainable module. Exposes a
training_step that returns the structured loss dictionary.

Training objective:
    L = L_pred + lambda * AtlasReg(z)
      = MSE(z_pred[t], z_target[t+1])  +  lambda * regularizer(z)

No EMAs, no stop-gradients, no target network. The predictor and
encoder are jointly optimized end-to-end.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atlaswm.encoder import ViTEncoder
from atlaswm.predictor import Predictor
from atlaswm.regularizer import AtlasReg, AtlasRegConfig


class AtlasWM(nn.Module):
    """End-to-end JEPA world model.

    Args:
        img_size: Observation spatial resolution (assumed square).
        patch_size: ViT patch size.
        embed_dim: Latent dimension d.
        action_dim: Action vector dimension.
        history_length: Max context length for the predictor.
        encoder_depth / encoder_heads: ViT config.
        predictor_depth / predictor_heads / predictor_dropout: predictor config.
        reg_config: AtlasReg configuration (or None for defaults).
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

    # ------------------------------------------------------------------
    # Basic ops
    # ------------------------------------------------------------------

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observations to latent embeddings.

        obs: (B, T, C, H, W) -> (B, T, D),  or (B, C, H, W) -> (B, D).
        """
        return self.encoder(obs)

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """Run the predictor (teacher forcing). See Predictor.forward."""
        return self.predictor(z, actions)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        obs: Tensor,
        actions: Tensor,
        lambda_reg: float = 0.1,
    ) -> dict:
        """Compute the full training loss on a minibatch of trajectories.

        Args:
            obs: (B, T, C, H, W) raw pixel observations.
            actions: (B, T, A) actions. actions[:, t] is the action taken
                at step t, i.e., the one that produced obs[:, t+1] from obs[:, t].
            lambda_reg: Weight on the AtlasReg term.

        Returns:
            Dict with keys 'total', 'pred', 'reg'.
        """
        B, T = obs.shape[:2]

        # Encode all frames at once.
        z = self.encoder(obs)  # (B, T, D)

        # Predict next-step embeddings via teacher forcing.
        z_next_pred = self.predictor(z, actions)  # (B, T, D)

        # Prediction loss: z_next_pred[:, t] should match z[:, t+1]
        # (teacher forcing). We use MSE.
        pred_loss = F.mse_loss(
            z_next_pred[:, :-1], z[:, 1:].detach() if False else z[:, 1:]
        )
        # Note: We do NOT detach z[:, 1:] above — gradients flow through
        # both encoder and predictor from the prediction loss.

        # Regularizer on all embeddings (flattens B*T).
        reg_loss = self.regularizer(z)

        total = pred_loss + lambda_reg * reg_loss
        return {'total': total, 'pred': pred_loss, 'reg': reg_loss}

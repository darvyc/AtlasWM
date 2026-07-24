"""Causal transformer predictor for AtlasWM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaLN(nn.Module):
    """Action-conditioned LayerNorm with zero-initialized modulation."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class CausalAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(batch, length, dim))


class PredictorBlock(nn.Module):
    """Pre-norm transformer block with AdaLN action conditioning."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = CausalAttention(dim, n_heads, dropout=dropout)
        self.norm2 = AdaLN(dim, cond_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x, cond))
        return x + self.mlp(self.norm2(x, cond))


class Predictor(nn.Module):
    """Causal transformer predicting z_(t+1) from z_(<=t), a_(<=t)."""

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 2,
        history_length: int = 8,
        depth: int = 6,
        n_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, history_length, embed_dim))
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    embed_dim,
                    embed_dim,
                    n_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, affine=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, AdaLN):
                nn.init.zeros_(module.to_scale_shift.weight)
                nn.init.zeros_(module.to_scale_shift.bias)

    def forward(self, z: Tensor, actions: Tensor) -> Tensor:
        """Predict next-step embeddings at every teacher-forced position."""
        if z.dim() != 3 or actions.dim() != 3:
            raise ValueError("z and actions must have shapes (B, T, D/A)")
        batch, length, dim = z.shape
        if actions.shape[:2] != (batch, length):
            raise ValueError("z and actions must share batch and time dimensions")
        if dim != self.embed_dim:
            raise ValueError(f"expected embed dim {self.embed_dim}, got {dim}")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action dim {self.action_dim}, got {actions.shape[-1]}"
            )
        if length > self.pos_embed.shape[1]:
            raise ValueError(
                f"history length {length} exceeds positional capacity "
                f"{self.pos_embed.shape[1]}"
            )

        cond = self.action_embed(actions)
        x = z + self.pos_embed[:, :length]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        return self.proj_head(x.reshape(batch * length, dim)).view(
            batch, length, dim
        )

    def rollout(
        self,
        z0: Tensor,
        actions: Tensor,
        context_actions: Optional[Tensor] = None,
    ) -> Tensor:
        """Autoregressively roll out a proposed action sequence.

        The final latent token in each call is conditioned on the action that
        should produce the next latent. For T0 context states,
        ``context_actions`` has T0-1 known transition actions. When omitted,
        historical actions are represented by zeros.
        """
        if z0.dim() != 3 or actions.dim() != 3:
            raise ValueError("z0 and actions must have shapes (B, T0/H, D/A)")
        batch, context_length, dim = z0.shape
        if context_length < 1:
            raise ValueError("z0 must contain at least one context state")
        if dim != self.embed_dim:
            raise ValueError(f"expected embed dim {self.embed_dim}, got {dim}")
        if actions.shape[0] != batch or actions.shape[-1] != self.action_dim:
            raise ValueError("actions have incompatible batch or action dimension")

        if context_actions is None:
            context_actions = torch.zeros(
                batch,
                context_length - 1,
                self.action_dim,
                device=z0.device,
                dtype=z0.dtype,
            )
        expected_shape = (batch, context_length - 1, self.action_dim)
        if tuple(context_actions.shape) != expected_shape:
            raise ValueError(
                f"context_actions must have shape {expected_shape}, got "
                f"{tuple(context_actions.shape)}"
            )

        horizon = actions.shape[1]
        max_context = self.pos_embed.shape[1]
        z_history = z0
        predictions = []

        for step in range(horizon):
            aligned_actions = torch.cat(
                [context_actions, actions[:, : step + 1]], dim=1
            )
            length = z_history.shape[1]
            if aligned_actions.shape[1] != length:
                raise RuntimeError("internal action/latent alignment failure")

            if length > max_context:
                z_in = z_history[:, -max_context:]
                a_in = aligned_actions[:, -max_context:]
            else:
                z_in = z_history
                a_in = aligned_actions

            z_new = self.forward(z_in, a_in)[:, -1:, :]
            predictions.append(z_new)
            z_history = torch.cat([z_history, z_new], dim=1)

        if not predictions:
            return z0.new_empty(batch, 0, self.embed_dim)
        return torch.cat(predictions, dim=1)

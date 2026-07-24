"""Causal action-conditioned latent dynamics predictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaLN(nn.Module):
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(values) * (1.0 + scale) + shift


class CausalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, values: Tensor) -> Tensor:
        batch, length, dim = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, dim))


class PredictorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        condition_dim: int,
        n_heads: int,
        dropout: float,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = AdaLN(dim, condition_dim)
        self.attention = CausalAttention(dim, n_heads, dropout)
        self.norm2 = AdaLN(dim, condition_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        values = values + self.attention(self.norm1(values, condition))
        return values + self.mlp(self.norm2(values, condition))


class Predictor(nn.Module):
    """Predict ``z_(t+1)`` from causal latent and action histories."""

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
        if history_length < 1 or depth < 0:
            raise ValueError("history_length must be positive and depth non-negative")
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, history_length, embed_dim)
        )
        self.blocks = nn.ModuleList(
            PredictorBlock(embed_dim, embed_dim, n_heads, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, AdaLN):
                nn.init.zeros_(module.modulation.weight)
                nn.init.zeros_(module.modulation.bias)

    def forward(self, latent: Tensor, actions: Tensor) -> Tensor:
        if latent.ndim != 3 or actions.ndim != 3:
            raise ValueError("latent and actions must have shape (B,T,D/A)")
        batch, length, dim = latent.shape
        if actions.shape[:2] != (batch, length):
            raise ValueError("latent and actions must share batch and time dimensions")
        if dim != self.embed_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("latent or action dimension mismatch")
        if length > self.history_length:
            raise ValueError(
                f"sequence length {length} exceeds predictor capacity {self.history_length}"
            )
        condition = self.action_embedding(actions)
        values = latent + self.positional_embedding[:, :length]
        for block in self.blocks:
            values = block(values, condition)
        return self.output_head(self.norm(values))

    def rollout(
        self,
        context_latent: Tensor,
        future_actions: Tensor,
        context_actions: Tensor | None = None,
    ) -> Tensor:
        """Autoregressively predict future latent states.

        ``context_actions`` must contain the real actions connecting the context
        states and therefore has length ``context_length - 1``. Missing action
        history is accepted only for a single-state context.
        """
        if context_latent.ndim != 3 or future_actions.ndim != 3:
            raise ValueError("context_latent and future_actions must be rank three")
        batch, context_length, dim = context_latent.shape
        if context_length < 1 or dim != self.embed_dim:
            raise ValueError("invalid context latent shape")
        if future_actions.shape[0] != batch or future_actions.shape[-1] != self.action_dim:
            raise ValueError("invalid future action shape")
        expected_context_shape = (batch, context_length - 1, self.action_dim)
        if context_actions is None:
            if context_length != 1:
                raise ValueError(
                    "context_actions are required when more than one context state is supplied"
                )
            context_actions = future_actions.new_empty(batch, 0, self.action_dim)
        if tuple(context_actions.shape) != expected_context_shape:
            raise ValueError(
                f"context_actions must have shape {expected_context_shape}, got {tuple(context_actions.shape)}"
            )

        latent_history = context_latent
        predictions = []
        horizon = future_actions.shape[1]
        for step in range(horizon):
            aligned_actions = torch.cat(
                (context_actions, future_actions[:, : step + 1]),
                dim=1,
            )
            if aligned_actions.shape[1] != latent_history.shape[1]:
                raise RuntimeError("internal action/latent alignment failure")
            latent_input = latent_history[:, -self.history_length :]
            action_input = aligned_actions[:, -self.history_length :]
            next_latent = self.forward(latent_input, action_input)[:, -1:]
            predictions.append(next_latent)
            latent_history = torch.cat((latent_history, next_latent), dim=1)
        if not predictions:
            return context_latent.new_empty(batch, 0, self.embed_dim)
        return torch.cat(predictions, dim=1)

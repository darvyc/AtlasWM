"""Causal transformer predictor for AtlasWM.

Predicts the next-step latent embedding z_{t+1} from a history of
(z_{1:t}, a_{1:t}). Actions are injected via Adaptive LayerNorm (AdaLN)
with zero-init so that action conditioning ramps up progressively
during training.

Architecture:
  - Input tokens: z_1, z_2, ..., z_T (each a d-dim vector)
  - Learned positional embedding added per position
  - Causal mask so position t only attends to positions <= t
  - 6 transformer blocks with AdaLN action conditioning, 16 heads, 10% dropout
  - Output projection with BatchNorm (same reason as encoder)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaLN(nn.Module):
    """Adaptive LayerNorm: scale/shift modulated by the action embedding.

    Zero-initialized so the block acts as identity at training start.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, T, D), cond: (B, T, cond_dim)
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class CausalAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # F.scaled_dot_product_attention has a built-in causal mask.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


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
        x = x + self.mlp(self.norm2(x, cond))
        return x


class Predictor(nn.Module):
    """Causal transformer that predicts z_{t+1} from (z_{1:t}, a_{1:t}).

    Args:
        embed_dim: Latent dimension d.
        action_dim: Action vector dimension.
        history_length: Maximum context length T_max (for positional emb).
        depth: Number of transformer blocks.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

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
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, history_length, embed_dim)
        )

        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection head with BatchNorm (see encoder for rationale)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, affine=True),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor, actions: Tensor) -> Tensor:
        """Predict next-step embeddings for all positions (teacher forcing).

        Args:
            z: (B, T, D) latent history.
            actions: (B, T, A) action history.

        Returns:
            z_next: (B, T, D). Position t is the prediction for z_{t+1}
                (autoregressive, causal — position t never sees positions > t).
        """
        B, T, D = z.shape
        if T > self.pos_embed.shape[1]:
            raise ValueError(
                f"history length {T} exceeds pos_embed capacity "
                f"{self.pos_embed.shape[1]}"
            )
        a_emb = self.action_embed(actions)  # (B, T, D)
        x = z + self.pos_embed[:, :T]
        for blk in self.blocks:
            x = blk(x, a_emb)
        x = self.norm(x)
        # BN wants (B*T, D)
        x_flat = x.reshape(B * T, D)
        out = self.proj_head(x_flat).view(B, T, D)
        return out

    def rollout(
        self,
        z0: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Autoregressive rollout from initial context for planning.

        The predictor's convention: at position t, given (z_{<=t}, a_{<=t}),
        it outputs a prediction of z_{t+1}. So to unroll H steps into the
        future from a context of length T0, we repeatedly:
          1. Build inputs (z_hist, a_hist) where a_hist has actions_executed
             so far (padded with zeros for pre-context steps).
          2. Forward through the predictor.
          3. Take the last-position output as the prediction for the next
             latent state.
          4. Append the prediction and the newly-executed action to history.

        Context is clipped to the pos_embed window to respect history_length.

        Args:
            z0: (B, T0, D) initial context embeddings.
            actions: (B, H, A) action sequence to execute.

        Returns:
            z_pred: (B, H, D) predicted embeddings for the H future steps.
        """
        B, T0, D = z0.shape
        H = actions.shape[1]
        max_ctx = self.pos_embed.shape[1]

        z_hist = z0
        # Pad the context with zero actions (these pre-context actions are
        # never "real"; zero-init AdaLN makes them no-ops anyway).
        a_hist = torch.zeros(
            B, T0, self.action_dim, device=z0.device, dtype=z0.dtype
        )

        preds = []
        for h in range(H):
            # Append the new action for this step.
            a_new = actions[:, h:h + 1]
            a_input = torch.cat([a_hist, a_new], dim=1)  # (B, T0+h+1, A)
            # For the predictor input we use z_hist as is, aligned with
            # a_input[:, :len(z_hist)]. We feed both at length L = len(z_hist).
            L = z_hist.shape[1]
            if L > max_ctx:
                # Keep last max_ctx steps.
                z_in = z_hist[:, -max_ctx:]
                a_in = a_input[:, L - max_ctx:L]
            else:
                z_in = z_hist
                a_in = a_input[:, :L]
            pred = self.forward(z_in, a_in)        # (B, L, D)
            z_new = pred[:, -1:, :]                # (B, 1, D) next-step pred
            preds.append(z_new)
            z_hist = torch.cat([z_hist, z_new], dim=1)
            a_hist = torch.cat([a_hist, a_new], dim=1)
        return torch.cat(preds, dim=1)             # (B, H, D)

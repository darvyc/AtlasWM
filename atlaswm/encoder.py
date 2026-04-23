"""Vision Transformer encoder for AtlasWM.

A minimal ViT-Tiny (~5M params) that maps image observations to a
d-dimensional latent embedding via the [CLS] token.

Design choices:
  - Projection head with BatchNorm after the ViT. The final ViT layer
    applies LayerNorm, which would prevent AtlasReg from optimizing the
    distribution of embeddings effectively (LN removes exactly the
    variance structure we're trying to shape). The BN projection
    re-opens that degree of freedom.
  - Patch size 14 on 224x224 images -> 16x16 = 256 patches + 1 CLS.

This module is self-contained (no external ViT dependency) but kept
small for clarity. For production-scale experiments, swap in timm or
HuggingFace ViT-Tiny.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEmbed(nn.Module):
    """Conv-based patch embedding. Equivalent to linear patch projection."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} not divisible by patch_size {patch_size}"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer with a BN projection head for post-LN decorrelation.

    Matches the ViT-Tiny default from the paper:
        depth=12, heads=3, dim=192, patch=14

    The projection head is a 1-layer MLP with BatchNorm. Its purpose
    is to remove the LayerNorm constraint imposed by the final transformer
    block; without this, AtlasReg cannot modulate the embedding variance.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # BN projection head to decouple from LayerNorm.
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, affine=True),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode images to latent embeddings.

        Args:
            x: (B, C, H, W) float images in roughly [-1, 1] or [0, 1].
               If input has an extra time dim (B, T, C, H, W), it is
               flattened before encoding and reshaped back after.

        Returns:
            (B, D) or (B, T, D) latent embeddings.
        """
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            z = self._forward_flat(x)
            return z.view(B, T, -1)
        return self._forward_flat(x)

    def _forward_flat(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        return self.proj_head(cls_out)

"""Vision Transformer encoder without batch-coupled latent normalization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if img_size < 1 or patch_size < 1 or img_size % patch_size:
            raise ValueError("img_size must be positive and divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (B,C,H,W)")
        if images.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {images.shape[1]}")
        if images.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"expected image size {(self.img_size, self.img_size)}, got {tuple(images.shape[-2:])}"
            )
        return self.projection(images).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
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
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, dim))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class ViTEncoder(nn.Module):
    """Map each frame independently to a latent vector.

    Normalization is per token and per sample. No operation computes statistics
    across batch elements or across time, preventing future-frame leakage through
    batch normalization during trajectory training.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        projection_hidden_ratio: float = 2.0,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        hidden = max(embed_dim, int(embed_dim * projection_hidden_ratio))
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        trajectory_shape: tuple[int, int] | None = None
        if images.ndim == 5:
            batch, time = images.shape[:2]
            trajectory_shape = (batch, time)
            images = images.reshape(batch * time, *images.shape[2:])
        elif images.ndim != 4:
            raise ValueError("images must have shape (B,C,H,W) or (B,T,C,H,W)")

        patches = self.patch_embed(images)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        values = torch.cat((cls, patches), dim=1)
        values = self.dropout(values + self.positional_embedding)
        for block in self.blocks:
            values = block(values)
        latent = self.projection_head(self.norm(values)[:, 0])
        if trajectory_shape is not None:
            return latent.reshape(*trajectory_shape, self.embed_dim)
        return latent

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

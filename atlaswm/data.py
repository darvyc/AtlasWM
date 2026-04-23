"""Minimal trajectory dataset + synthetic 2D toy environment.

The toy environment: an agent (red dot) moves in a 2D square. The agent's
position (x, y) is integrated from actions (dx, dy). Observations are
rendered as 64x64 RGB images with the agent as a small red disk on a
white background.

Useful for smoke-testing AtlasWM end-to-end without any external deps
(no gym, no mujoco, no datasets). The toy has intrinsic dimensionality 2
embedded in image space, which is a good stress test for the regularizer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ToyEnvConfig:
    img_size: int = 64
    agent_radius: int = 3
    box_low: float = 0.1
    box_high: float = 0.9
    dt: float = 0.05
    action_noise: float = 0.02


def render_toy(
    positions: np.ndarray,
    cfg: ToyEnvConfig = ToyEnvConfig(),
) -> np.ndarray:
    """Render a batch of (x, y) positions to RGB images.

    Args:
        positions: (T, 2) array of (x, y) in [0, 1]^2.

    Returns:
        (T, 3, H, W) uint8 array.
    """
    T = positions.shape[0]
    H = W = cfg.img_size
    r = cfg.agent_radius
    imgs = np.full((T, 3, H, W), 255, dtype=np.uint8)  # white background

    # Black border
    imgs[:, :, 0, :] = 0
    imgs[:, :, -1, :] = 0
    imgs[:, :, :, 0] = 0
    imgs[:, :, :, -1] = 0

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for t in range(T):
        cx = int(positions[t, 0] * W)
        cy = int(positions[t, 1] * H)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        # Paint red where mask: R=255, G=0, B=0
        imgs[t, 0][mask] = 255
        imgs[t, 1][mask] = 0
        imgs[t, 2][mask] = 0
    return imgs


def generate_toy_trajectory(
    length: int,
    cfg: ToyEnvConfig = ToyEnvConfig(),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single trajectory of (positions, actions).

    The policy is a lightly-noised random walk within the box.

    Returns:
        positions: (length, 2)
        actions:   (length, 2)
    """
    if rng is None:
        rng = np.random.default_rng()
    positions = np.zeros((length, 2), dtype=np.float32)
    actions = np.zeros((length, 2), dtype=np.float32)
    positions[0] = rng.uniform(cfg.box_low, cfg.box_high, size=2)
    for t in range(length - 1):
        a = rng.normal(0, 1.0, size=2).astype(np.float32)
        # Add a soft restoring force toward center to keep the agent in frame
        center = np.array([0.5, 0.5], dtype=np.float32)
        a += 0.5 * (center - positions[t])
        actions[t] = a
        new_pos = positions[t] + cfg.dt * a
        # Reflect at box walls
        new_pos = np.clip(new_pos, cfg.box_low, cfg.box_high)
        positions[t + 1] = new_pos
    actions[-1] = actions[-2]  # pad
    return positions, actions


class ToyTrajectoryDataset(Dataset):
    """In-memory dataset of toy trajectories.

    Each item is a sub-trajectory of length `sub_length`, returned as
    (obs, actions) where:
        obs:     (sub_length, 3, img_size, img_size) float32 in [-1, 1]
        actions: (sub_length, 2) float32

    Args:
        n_trajectories: Number of full trajectories to generate.
        traj_length: Length of each full trajectory.
        sub_length: Sub-trajectory length to sample per __getitem__.
        seed: RNG seed.
        cfg: ToyEnvConfig.
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        traj_length: int = 64,
        sub_length: int = 8,
        seed: int = 0,
        cfg: ToyEnvConfig = ToyEnvConfig(),
    ):
        self.sub_length = sub_length
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.positions = []
        self.obs = []
        self.actions = []
        for _ in range(n_trajectories):
            pos, act = generate_toy_trajectory(traj_length, cfg, rng)
            imgs = render_toy(pos, cfg)
            # Normalize to [-1, 1]
            imgs_f = (imgs.astype(np.float32) / 127.5) - 1.0
            self.positions.append(pos)
            self.obs.append(imgs_f)
            self.actions.append(act)
        self.positions = np.stack(self.positions)    # (N, T, 2)
        self.obs = np.stack(self.obs)                 # (N, T, 3, H, W)
        self.actions = np.stack(self.actions)         # (N, T, 2)

    def __len__(self) -> int:
        return self.positions.shape[0] * (self.positions.shape[1] - self.sub_length + 1)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        n_per_traj = self.positions.shape[1] - self.sub_length + 1
        traj_idx = idx // n_per_traj
        start = idx % n_per_traj
        sl = slice(start, start + self.sub_length)
        obs = torch.from_numpy(self.obs[traj_idx, sl])
        act = torch.from_numpy(self.actions[traj_idx, sl])
        pos = torch.from_numpy(self.positions[traj_idx, sl])
        return obs, act, pos

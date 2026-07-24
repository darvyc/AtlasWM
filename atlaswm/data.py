"""Trajectory datasets and the included synthetic visual environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ToyEnvConfig:
    """Configuration for the synthetic two-dimensional visual environment."""

    img_size: int = 64
    agent_radius: int = 3
    box_low: float = 0.1
    box_high: float = 0.9
    dt: float = 0.05
    action_noise: float = 0.02

    def __post_init__(self) -> None:
        if self.img_size < 4:
            raise ValueError("img_size must be at least 4")
        if self.agent_radius < 1:
            raise ValueError("agent_radius must be positive")
        if not 0.0 <= self.box_low < self.box_high <= 1.0:
            raise ValueError("require 0 <= box_low < box_high <= 1")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.action_noise < 0:
            raise ValueError("action_noise must be non-negative")


def render_toy(
    positions: np.ndarray,
    cfg: Optional[ToyEnvConfig] = None,
) -> np.ndarray:
    """Render a trajectory of two-dimensional positions as RGB images.

    Args:
        positions: Array with shape ``(T, 2)`` and coordinates in ``[0, 1]``.
        cfg: Environment configuration.

    Returns:
        Unsigned-byte array with shape ``(T, 3, H, W)``.
    """
    cfg = cfg or ToyEnvConfig()
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"positions must have shape (T, 2), got {positions.shape}"
        )

    length = positions.shape[0]
    height = width = cfg.img_size
    radius = cfg.agent_radius
    images = np.full((length, 3, height, width), 255, dtype=np.uint8)

    images[:, :, 0, :] = 0
    images[:, :, -1, :] = 0
    images[:, :, :, 0] = 0
    images[:, :, :, -1] = 0

    yy, xx = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    for index in range(length):
        cx = int(np.clip(positions[index, 0] * width, 0, width - 1))
        cy = int(np.clip(positions[index, 1] * height, 0, height - 1))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        images[index, 0][mask] = 255
        images[index, 1][mask] = 0
        images[index, 2][mask] = 0
    return images


def generate_toy_trajectory(
    length: int,
    cfg: Optional[ToyEnvConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic position and action trajectory."""
    if length < 2:
        raise ValueError("length must be at least 2")
    cfg = cfg or ToyEnvConfig()
    rng = rng or np.random.default_rng()

    positions = np.zeros((length, 2), dtype=np.float32)
    actions = np.zeros((length, 2), dtype=np.float32)
    positions[0] = rng.uniform(cfg.box_low, cfg.box_high, size=2)

    center = np.array([0.5, 0.5], dtype=np.float32)
    for index in range(length - 1):
        action = rng.normal(0.0, 1.0, size=2).astype(np.float32)
        action += 0.5 * (center - positions[index])
        if cfg.action_noise:
            action += rng.normal(
                0.0, cfg.action_noise, size=2
            ).astype(np.float32)
        actions[index] = action
        positions[index + 1] = np.clip(
            positions[index] + cfg.dt * action,
            cfg.box_low,
            cfg.box_high,
        )

    actions[-1] = actions[-2]
    return positions, actions


class ToyTrajectoryDataset(Dataset):
    """In-memory visual trajectories from the synthetic environment."""

    def __init__(
        self,
        n_trajectories: int = 100,
        traj_length: int = 64,
        sub_length: int = 8,
        seed: int = 0,
        cfg: Optional[ToyEnvConfig] = None,
    ) -> None:
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be positive")
        if traj_length < 2:
            raise ValueError("traj_length must be at least 2")
        if not 2 <= sub_length <= traj_length:
            raise ValueError("sub_length must lie in [2, traj_length]")

        self.sub_length = sub_length
        self.cfg = cfg or ToyEnvConfig()
        rng = np.random.default_rng(seed)

        positions = []
        observations = []
        actions = []
        for _ in range(n_trajectories):
            position, action = generate_toy_trajectory(
                traj_length, self.cfg, rng
            )
            images = render_toy(position, self.cfg)
            images = images.astype(np.float32) / 127.5 - 1.0
            positions.append(position)
            observations.append(images)
            actions.append(action)

        self.positions = np.stack(positions)
        self.obs = np.stack(observations)
        self.actions = np.stack(actions)

    def __len__(self) -> int:
        windows_per_trajectory = self.obs.shape[1] - self.sub_length + 1
        return int(self.obs.shape[0] * windows_per_trajectory)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        windows_per_trajectory = self.obs.shape[1] - self.sub_length + 1
        trajectory_index = index // windows_per_trajectory
        start = index % windows_per_trajectory
        window = slice(start, start + self.sub_length)
        observations = torch.from_numpy(self.obs[trajectory_index, window])
        actions = torch.from_numpy(self.actions[trajectory_index, window])
        positions = torch.from_numpy(self.positions[trajectory_index, window])
        return observations, actions, positions


class TrajectoryNPZDataset(Dataset):
    """Sliding-window trajectory dataset stored in a NumPy NPZ archive.

    The archive must contain observation and action arrays. Accepted shapes are
    ``(N, T, C, H, W)`` and ``(N, T, A)`` for multiple trajectories or
    ``(T, C, H, W)`` and ``(T, A)`` for one trajectory. Channel-last image
    arrays are supported through ``channel_last=True``.

    Args:
        path: NPZ file or directory containing ``trajectories.npz``.
        sub_length: Number of consecutive steps returned per item.
        obs_key: Observation array key.
        action_key: Action array key.
        state_key: Optional aligned state or label array key.
        normalize_images: Map unsigned-byte observations to ``[-1, 1]``.
        channel_last: Convert ``(..., H, W, C)`` observations to channels first.
    """

    def __init__(
        self,
        path: str | Path,
        sub_length: int,
        obs_key: str = "obs",
        action_key: str = "actions",
        state_key: Optional[str] = None,
        normalize_images: bool = True,
        channel_last: bool = False,
    ) -> None:
        archive_path = Path(path)
        if archive_path.is_dir():
            archive_path = archive_path / "trajectories.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(f"trajectory archive not found: {archive_path}")
        if sub_length < 2:
            raise ValueError("sub_length must be at least 2")

        with np.load(archive_path, allow_pickle=False) as archive:
            if obs_key not in archive:
                raise KeyError(f"observation key {obs_key!r} not found in {archive_path}")
            if action_key not in archive:
                raise KeyError(f"action key {action_key!r} not found in {archive_path}")
            observations = np.asarray(archive[obs_key])
            actions = np.asarray(archive[action_key])
            states = None
            if state_key is not None:
                if state_key not in archive:
                    raise KeyError(f"state key {state_key!r} not found in {archive_path}")
                states = np.asarray(archive[state_key])

        if observations.ndim == 4:
            observations = observations[None, ...]
        if actions.ndim == 2:
            actions = actions[None, ...]
        if states is not None and states.ndim >= 1 and states.shape[:1] == actions.shape[1:2]:
            states = states[None, ...]

        if observations.ndim != 5:
            raise ValueError(
                "observations must have shape (N,T,C,H,W), (N,T,H,W,C), "
                f"or a single-trajectory equivalent; got {observations.shape}"
            )
        if actions.ndim != 3:
            raise ValueError(
                f"actions must have shape (N,T,A), got {actions.shape}"
            )
        if channel_last:
            observations = np.moveaxis(observations, -1, -3)

        if observations.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "observation and action trajectory dimensions must match: "
                f"{observations.shape[:2]} != {actions.shape[:2]}"
            )
        if states is not None and states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "state and action trajectory dimensions must match: "
                f"{states.shape[:2]} != {actions.shape[:2]}"
            )

        trajectory_length = observations.shape[1]
        if sub_length > trajectory_length:
            raise ValueError(
                f"sub_length {sub_length} exceeds trajectory length {trajectory_length}"
            )

        if normalize_images and np.issubdtype(observations.dtype, np.integer):
            observations = observations.astype(np.float32) / 127.5 - 1.0
        else:
            observations = observations.astype(np.float32, copy=False)
        actions = actions.astype(np.float32, copy=False)
        if states is not None:
            states = states.astype(np.float32, copy=False)

        self.archive_path = archive_path
        self.sub_length = sub_length
        self.obs = np.ascontiguousarray(observations)
        self.actions = np.ascontiguousarray(actions)
        self.states = None if states is None else np.ascontiguousarray(states)

    def __len__(self) -> int:
        windows_per_trajectory = self.obs.shape[1] - self.sub_length + 1
        return int(self.obs.shape[0] * windows_per_trajectory)

    def __getitem__(self, index: int):
        windows_per_trajectory = self.obs.shape[1] - self.sub_length + 1
        trajectory_index = index // windows_per_trajectory
        start = index % windows_per_trajectory
        window = slice(start, start + self.sub_length)

        observations = torch.from_numpy(self.obs[trajectory_index, window])
        actions = torch.from_numpy(self.actions[trajectory_index, window])
        if self.states is None:
            return observations, actions
        states = torch.from_numpy(self.states[trajectory_index, window])
        return observations, actions, states

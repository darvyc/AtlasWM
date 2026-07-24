"""Trajectory data interfaces and a deterministic visual control environment."""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class ToyEnvConfig:
    img_size: int = 64
    agent_radius: int = 3
    box_low: float = 0.1
    box_high: float = 0.9
    dt: float = 0.05
    restoring_force: float = 0.5
    action_noise: float = 0.02
    action_limit: float = 2.0

    def __post_init__(self) -> None:
        if self.img_size < 8 or self.agent_radius < 1:
            raise ValueError("invalid rendering dimensions")
        if not 0 <= self.box_low < self.box_high <= 1:
            raise ValueError("require 0 <= box_low < box_high <= 1")
        if self.dt <= 0 or self.action_noise < 0 or self.action_limit <= 0:
            raise ValueError("invalid dynamics parameters")


def render_toy(positions: np.ndarray, config: ToyEnvConfig | None = None) -> np.ndarray:
    cfg = config or ToyEnvConfig()
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (T,2)")
    images = np.full(
        (positions.shape[0], 3, cfg.img_size, cfg.img_size),
        255,
        dtype=np.uint8,
    )
    images[:, :, (0, -1), :] = 0
    images[:, :, :, (0, -1)] = 0
    yy, xx = np.meshgrid(
        np.arange(cfg.img_size), np.arange(cfg.img_size), indexing="ij"
    )
    for index, position in enumerate(positions):
        center_x = int(np.clip(position[0] * (cfg.img_size - 1), 0, cfg.img_size - 1))
        center_y = int(np.clip(position[1] * (cfg.img_size - 1), 0, cfg.img_size - 1))
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= cfg.agent_radius**2
        images[index, 0, mask] = 255
        images[index, 1, mask] = 0
        images[index, 2, mask] = 0
    return images


class ToyVisualEnv:
    def __init__(self, config: ToyEnvConfig | None = None, seed: int = 0):
        self.config = config or ToyEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        position: np.ndarray | None = None,
        goal: np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        self.position = (
            self.rng.uniform(cfg.box_low, cfg.box_high, size=2).astype(np.float32)
            if position is None
            else np.asarray(position, dtype=np.float32).copy()
        )
        self.goal = (
            self.rng.uniform(cfg.box_low, cfg.box_high, size=2).astype(np.float32)
            if goal is None
            else np.asarray(goal, dtype=np.float32).copy()
        )
        self.steps = 0
        return self.observation(), self.goal_observation()

    def observation(self) -> Tensor:
        image = render_toy(self.position[None], self.config)[0]
        return torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)

    def goal_observation(self) -> Tensor:
        image = render_toy(self.goal[None], self.config)[0]
        return torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)

    def step(self, action: np.ndarray | Tensor) -> tuple[Tensor, float, bool, dict[str, float]]:
        cfg = self.config
        action_array = (
            action.detach().cpu().numpy() if isinstance(action, Tensor) else np.asarray(action)
        )
        action_array = np.clip(action_array.astype(np.float32), -cfg.action_limit, cfg.action_limit)
        noise = self.rng.normal(0.0, cfg.action_noise, size=2).astype(np.float32)
        drift = cfg.restoring_force * (
            np.array([0.5, 0.5], dtype=np.float32) - self.position
        )
        self.position = np.clip(
            self.position + cfg.dt * (action_array + drift + noise),
            cfg.box_low,
            cfg.box_high,
        )
        self.steps += 1
        distance = float(np.linalg.norm(self.position - self.goal))
        success = distance <= 0.06
        return self.observation(), -distance, success, {"goal_distance": distance}


def generate_toy_trajectory(
    length: int,
    config: ToyEnvConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if length < 2:
        raise ValueError("length must be at least two")
    cfg = config or ToyEnvConfig()
    rng = rng or np.random.default_rng()
    positions = np.zeros((length, 2), dtype=np.float32)
    actions = np.zeros((length, 2), dtype=np.float32)
    positions[0] = rng.uniform(cfg.box_low, cfg.box_high, size=2)
    center = np.array([0.5, 0.5], dtype=np.float32)
    for index in range(length - 1):
        action = np.clip(
            rng.normal(0.0, 1.0, size=2).astype(np.float32),
            -cfg.action_limit,
            cfg.action_limit,
        )
        actions[index] = action
        noise = rng.normal(0.0, cfg.action_noise, size=2).astype(np.float32)
        drift = cfg.restoring_force * (center - positions[index])
        positions[index + 1] = np.clip(
            positions[index] + cfg.dt * (action + drift + noise),
            cfg.box_low,
            cfg.box_high,
        )
    actions[-1] = actions[-2]
    return positions, actions


class _WindowedTrajectoryDataset(Dataset):
    sub_length: int
    n_trajectories: int
    trajectory_length: int

    @property
    def windows_per_trajectory(self) -> int:
        return self.trajectory_length - self.sub_length + 1

    @property
    def trajectory_window_ranges(self) -> list[range]:
        width = self.windows_per_trajectory
        return [
            range(trajectory * width, (trajectory + 1) * width)
            for trajectory in range(self.n_trajectories)
        ]

    def __len__(self) -> int:
        return self.n_trajectories * self.windows_per_trajectory

    def _location(self, index: int) -> tuple[int, slice]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory = index // self.windows_per_trajectory
        start = index % self.windows_per_trajectory
        return trajectory, slice(start, start + self.sub_length)


class ToyTrajectoryDataset(_WindowedTrajectoryDataset):
    def __init__(
        self,
        n_trajectories: int = 100,
        traj_length: int = 64,
        sub_length: int = 8,
        seed: int = 0,
        config: ToyEnvConfig | None = None,
    ):
        if n_trajectories < 1 or traj_length < 2:
            raise ValueError("invalid trajectory counts")
        if not 2 <= sub_length <= traj_length:
            raise ValueError("sub_length must lie in [2,traj_length]")
        self.sub_length = sub_length
        self.n_trajectories = n_trajectories
        self.trajectory_length = traj_length
        self.config = config or ToyEnvConfig()
        self.seed = seed
        rng = np.random.default_rng(seed)
        positions, observations, actions = [], [], []
        for _ in range(n_trajectories):
            state, action = generate_toy_trajectory(traj_length, self.config, rng)
            image = render_toy(state, self.config).astype(np.float32) / 127.5 - 1.0
            positions.append(state)
            observations.append(image)
            actions.append(action)
        self.positions = np.stack(positions)
        self.obs = np.stack(observations)
        self.actions = np.stack(actions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        trajectory, window = self._location(index)
        return (
            torch.from_numpy(self.obs[trajectory, window]),
            torch.from_numpy(self.actions[trajectory, window]),
            torch.from_numpy(self.positions[trajectory, window]),
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "kind": "toy",
            "seed": self.seed,
            "n_trajectories": self.n_trajectories,
            "trajectory_length": self.trajectory_length,
            "sub_length": self.sub_length,
            "config": asdict(self.config),
        }


def _validate_trajectory_shapes(
    observations: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray | None,
    sub_length: int,
) -> tuple[int, int]:
    if observations.ndim != 5:
        raise ValueError("observations must have shape (N,T,C,H,W)")
    if actions.ndim != 3:
        raise ValueError("actions must have shape (N,T,A)")
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError("observation/action trajectory dimensions differ")
    if states is not None and states.shape[:2] != actions.shape[:2]:
        raise ValueError("state/action trajectory dimensions differ")
    if not 2 <= sub_length <= observations.shape[1]:
        raise ValueError("sub_length must lie in [2,trajectory_length]")
    return int(observations.shape[0]), int(observations.shape[1])


def _frame_tensor(values: np.ndarray, normalize_images: bool) -> Tensor:
    if normalize_images and np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        if info.min < 0:
            raise ValueError("integer image normalization requires non-negative data")
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        return tensor / (info.max / 2.0) - 1.0
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


class TrajectoryArrayDataset(_WindowedTrajectoryDataset):
    def __init__(
        self,
        path: str | Path,
        sub_length: int,
        *,
        observation_file: str = "observations.npy",
        action_file: str = "actions.npy",
        state_file: str | None = "states.npy",
        normalize_images: bool = True,
        channel_last: bool = False,
    ):
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.root = root
        self.observation_path = root / observation_file
        self.action_path = root / action_file
        self.state_path = None if state_file is None else root / state_file
        if not self.observation_path.is_file() or not self.action_path.is_file():
            raise FileNotFoundError("observations.npy and actions.npy are required")
        observations = np.load(self.observation_path, mmap_mode="r", allow_pickle=False)
        actions = np.load(self.action_path, mmap_mode="r", allow_pickle=False)
        states = (
            np.load(self.state_path, mmap_mode="r", allow_pickle=False)
            if self.state_path is not None and self.state_path.is_file()
            else None
        )
        if channel_last:
            if observations.ndim != 5:
                raise ValueError("channel-last observations must be rank five")
            observations = np.moveaxis(observations, -1, -3)
        self.sub_length = sub_length
        self.n_trajectories, self.trajectory_length = _validate_trajectory_shapes(
            observations, actions, states, sub_length
        )
        self.obs = observations
        self.actions = actions
        self.states = states
        self.normalize_images = normalize_images

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        trajectory, window = self._location(index)
        observations = _frame_tensor(self.obs[trajectory, window], self.normalize_images)
        actions = torch.from_numpy(
            np.array(self.actions[trajectory, window], dtype=np.float32, copy=True)
        )
        if self.states is None:
            return observations, actions
        states = torch.from_numpy(
            np.array(self.states[trajectory, window], dtype=np.float32, copy=True)
        )
        return observations, actions, states

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "kind": "npy",
            "root": str(self.root.resolve()),
            "observations": str(self.observation_path.resolve()),
            "actions": str(self.action_path.resolve()),
            "states": None if self.state_path is None else str(self.state_path.resolve()),
            "n_trajectories": self.n_trajectories,
            "trajectory_length": self.trajectory_length,
            "sub_length": self.sub_length,
        }


class TrajectoryNPZDataset(_WindowedTrajectoryDataset):
    def __init__(
        self,
        path: str | Path,
        sub_length: int,
        *,
        obs_key: str = "obs",
        action_key: str = "actions",
        state_key: str | None = None,
        normalize_images: bool = True,
        channel_last: bool = False,
    ):
        archive_path = Path(path)
        if archive_path.is_dir():
            archive_path = archive_path / "trajectories.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            observations = np.asarray(archive[obs_key])
            actions = np.asarray(archive[action_key])
            states = None if state_key is None else np.asarray(archive[state_key])
        if observations.ndim == 4:
            observations = observations[None]
        if actions.ndim == 2:
            actions = actions[None]
            if states is not None:
                states = states[None]
        if channel_last:
            observations = np.moveaxis(observations, -1, -3)
        self.sub_length = sub_length
        self.n_trajectories, self.trajectory_length = _validate_trajectory_shapes(
            observations, actions, states, sub_length
        )
        self.archive_path = archive_path
        self.obs = observations
        self.actions = actions.astype(np.float32, copy=False)
        self.states = None if states is None else states.astype(np.float32, copy=False)
        self.normalize_images = normalize_images

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        trajectory, window = self._location(index)
        observations = _frame_tensor(self.obs[trajectory, window], self.normalize_images)
        actions = torch.from_numpy(
            np.array(self.actions[trajectory, window], dtype=np.float32, copy=True)
        )
        if self.states is None:
            return observations, actions
        states = torch.from_numpy(
            np.array(self.states[trajectory, window], dtype=np.float32, copy=True)
        )
        return observations, actions, states

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "kind": "npz",
            "path": str(self.archive_path.resolve()),
            "n_trajectories": self.n_trajectories,
            "trajectory_length": self.trajectory_length,
            "sub_length": self.sub_length,
        }


class TrajectoryManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, sub_length: int):
        manifest_path = Path(manifest)
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        shards = payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("manifest must contain a non-empty shards list")
        self.manifest_path = manifest_path
        self.datasets = [
            TrajectoryArrayDataset(
                manifest_path.parent / shard["path"],
                sub_length,
                observation_file=shard.get("observation_file", "observations.npy"),
                action_file=shard.get("action_file", "actions.npy"),
                state_file=shard.get("state_file", "states.npy"),
                normalize_images=shard.get("normalize_images", True),
                channel_last=shard.get("channel_last", False),
            )
            for shard in shards
        ]
        self.cumulative = []
        self._trajectory_window_ranges: list[range] = []
        total = 0
        for dataset in self.datasets:
            shard_offset = total
            self._trajectory_window_ranges.extend(
                range(shard_offset + window.start, shard_offset + window.stop)
                for window in dataset.trajectory_window_ranges
            )
            total += len(dataset)
            self.cumulative.append(total)
        self.n_trajectories = len(self._trajectory_window_ranges)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        shard = bisect.bisect_right(self.cumulative, index)
        previous = 0 if shard == 0 else self.cumulative[shard - 1]
        return self.datasets[shard][index - previous]

    @property
    def trajectory_window_ranges(self) -> list[range]:
        return self._trajectory_window_ranges

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "kind": "manifest",
            "path": str(self.manifest_path.resolve()),
            "shards": [dataset.fingerprint_payload() for dataset in self.datasets],
        }


def split_by_trajectory(
    dataset: Dataset,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    if not hasattr(dataset, "trajectory_window_ranges"):
        raise TypeError("dataset does not expose trajectory_window_ranges")
    ranges = list(dataset.trajectory_window_ranges)
    trajectories = np.random.default_rng(seed).permutation(len(ranges))
    train_end = int(round(len(ranges) * train_fraction))
    validation_end = train_end + int(round(len(ranges) * validation_fraction))

    def indices(selected: np.ndarray) -> list[int]:
        output: list[int] = []
        for trajectory in selected.tolist():
            output.extend(ranges[trajectory])
        return output

    return (
        Subset(dataset, indices(trajectories[:train_end])),
        Subset(dataset, indices(trajectories[train_end:validation_end])),
        Subset(dataset, indices(trajectories[validation_end:])),
    )

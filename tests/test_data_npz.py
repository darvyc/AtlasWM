"""Tests for generic NPZ trajectory loading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlaswm.cli import build_dataset
from atlaswm.data import TrajectoryNPZDataset


def test_multitrajectory_npz_with_states(tmp_path):
    observations = np.zeros((2, 5, 3, 8, 8), dtype=np.uint8)
    observations[:, :, :, 2:6, 2:6] = 255
    actions = np.arange(2 * 5 * 2, dtype=np.float32).reshape(2, 5, 2)
    states = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    archive = tmp_path / "trajectories.npz"
    np.savez(archive, obs=observations, actions=actions, states=states)

    dataset = TrajectoryNPZDataset(
        archive,
        sub_length=3,
        state_key="states",
    )

    assert len(dataset) == 2 * (5 - 3 + 1)
    obs, act, state = dataset[0]
    assert obs.shape == (3, 3, 8, 8)
    assert act.shape == (3, 2)
    assert state.shape == (3, 3)
    assert obs.dtype == torch.float32
    assert act.dtype == torch.float32
    assert float(obs.min()) == -1.0
    assert float(obs.max()) == 1.0


def test_single_trajectory_and_directory_resolution(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    observations = np.random.default_rng(0).normal(
        size=(5, 3, 8, 8)
    ).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)
    states = np.random.default_rng(2).normal(size=(5, 4)).astype(np.float32)
    np.savez(
        directory / "trajectories.npz",
        obs=observations,
        actions=actions,
        states=states,
    )

    dataset = TrajectoryNPZDataset(
        directory,
        sub_length=4,
        state_key="states",
        normalize_images=False,
    )

    assert len(dataset) == 2
    obs, act, state = dataset[1]
    assert obs.shape == (4, 3, 8, 8)
    assert act.shape == (4, 2)
    assert state.shape == (4, 4)
    assert torch.allclose(obs, torch.from_numpy(observations[1:5]))


def test_channel_last_conversion(tmp_path):
    observations = np.zeros((1, 4, 8, 8, 3), dtype=np.uint8)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    archive = tmp_path / "channel_last.npz"
    np.savez(archive, obs=observations, actions=actions)

    dataset = TrajectoryNPZDataset(
        archive,
        sub_length=2,
        channel_last=True,
    )
    obs, act = dataset[0]

    assert obs.shape == (2, 3, 8, 8)
    assert act.shape == (2, 2)


def test_cli_builds_npz_dataset(tmp_path):
    observations = np.zeros((1, 4, 3, 8, 8), dtype=np.uint8)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    archive = tmp_path / "trajectories.npz"
    np.savez(archive, obs=observations, actions=actions)

    cfg = {
        "seed": 42,
        "data": {
            "name": "trajectory_npz",
            "path": str(archive),
            "sub_length": 3,
        },
    }
    dataset = build_dataset(cfg)

    assert isinstance(dataset, TrajectoryNPZDataset)
    assert len(dataset) == 2


def test_npz_shape_validation(tmp_path):
    archive = tmp_path / "bad.npz"
    np.savez(
        archive,
        obs=np.zeros((2, 5, 3, 8, 8), dtype=np.uint8),
        actions=np.zeros((3, 5, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="trajectory dimensions must match"):
        TrajectoryNPZDataset(archive, sub_length=3)


def test_npz_missing_key_and_window_validation(tmp_path):
    archive = tmp_path / "missing.npz"
    np.savez(
        archive,
        obs=np.zeros((1, 4, 3, 8, 8), dtype=np.uint8),
    )

    with pytest.raises(KeyError, match="action key"):
        TrajectoryNPZDataset(archive, sub_length=3)

    complete = tmp_path / "complete.npz"
    np.savez(
        complete,
        obs=np.zeros((1, 4, 3, 8, 8), dtype=np.uint8),
        actions=np.zeros((1, 4, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="exceeds trajectory length"):
        TrajectoryNPZDataset(complete, sub_length=5)

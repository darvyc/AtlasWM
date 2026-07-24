from pathlib import Path

import numpy as np
import torch

from atlaswm.data import (
    ToyTrajectoryDataset,
    TrajectoryArrayDataset,
    TrajectoryManifestDataset,
    split_by_trajectory,
)


def test_memory_mapped_array_dataset(tmp_path: Path):
    observations = np.zeros((3, 6, 3, 8, 8), dtype=np.uint8)
    observations[:, :, 0] = 255
    actions = np.arange(3 * 6 * 2, dtype=np.float32).reshape(3, 6, 2)
    states = np.zeros((3, 6, 2), dtype=np.float32)
    np.save(tmp_path / "observations.npy", observations)
    np.save(tmp_path / "actions.npy", actions)
    np.save(tmp_path / "states.npy", states)
    dataset = TrajectoryArrayDataset(tmp_path, sub_length=4)
    obs, act, state = dataset[0]
    assert isinstance(dataset.obs, np.memmap)
    assert obs.shape == (4, 3, 8, 8)
    assert obs.max() == 1 and obs.min() == -1
    assert act.shape == (4, 2)
    assert state.shape == (4, 2)


def test_split_keeps_trajectory_windows_together():
    dataset = ToyTrajectoryDataset(n_trajectories=10, traj_length=8, sub_length=3, seed=2)
    train, validation, test = split_by_trajectory(
        dataset,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=5,
    )

    def trajectories(subset):
        return {index // dataset.windows_per_trajectory for index in subset.indices}

    train_ids = trajectories(train)
    validation_ids = trajectories(validation)
    test_ids = trajectories(test)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(train) + len(validation) + len(test) == len(dataset)


def test_manifest_concatenates_shards(tmp_path: Path):
    for shard_index in range(2):
        shard = tmp_path / f"shard{shard_index}"
        shard.mkdir()
        np.save(shard / "observations.npy", np.zeros((1, 4, 3, 8, 8), dtype=np.uint8))
        np.save(shard / "actions.npy", np.zeros((1, 4, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"shards": [{"path": "shard0", "state_file": null}, {"path": "shard1", "state_file": null}]}',
        encoding="utf-8",
    )
    dataset = TrajectoryManifestDataset(manifest, sub_length=3)
    assert len(dataset) == 4
    obs, action = dataset[-1]
    assert obs.shape[0] == action.shape[0] == 3


def test_toy_environment_and_npz_adapter(tmp_path: Path):
    from atlaswm.data import ToyVisualEnv, TrajectoryNPZDataset

    environment = ToyVisualEnv(seed=3)
    observation, goal = environment.reset(
        position=np.array([0.2, 0.2], dtype=np.float32),
        goal=np.array([0.8, 0.8], dtype=np.float32),
    )
    assert observation.shape == goal.shape == (3, 64, 64)
    next_observation, reward, success, info = environment.step(torch.tensor([1.0, 1.0]))
    assert next_observation.shape == observation.shape
    assert reward <= 0 and not success and info["goal_distance"] > 0

    archive = tmp_path / "trajectories.npz"
    np.savez(
        archive,
        obs=np.zeros((2, 5, 8, 8, 3), dtype=np.uint8),
        actions=np.zeros((2, 5, 2), dtype=np.float32),
        states=np.zeros((2, 5, 2), dtype=np.float32),
    )
    dataset = TrajectoryNPZDataset(
        archive,
        sub_length=3,
        state_key="states",
        channel_last=True,
    )
    obs, action, state = dataset[1]
    assert obs.shape == (3, 3, 8, 8)
    assert action.shape == state.shape == (3, 2)


def test_manifest_split_is_trajectory_safe(tmp_path: Path):
    for shard_index in range(2):
        shard = tmp_path / f"split_shard{shard_index}"
        shard.mkdir()
        np.save(shard / "observations.npy", np.zeros((2, 5, 3, 8, 8), dtype=np.uint8))
        np.save(shard / "actions.npy", np.zeros((2, 5, 2), dtype=np.float32))
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        '{"shards": [{"path": "split_shard0", "state_file": null}, {"path": "split_shard1", "state_file": null}]}',
        encoding="utf-8",
    )
    dataset = TrajectoryManifestDataset(manifest, sub_length=3)
    train, validation, test = split_by_trajectory(
        dataset,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=9,
    )
    membership = [set(part.indices) for part in (train, validation, test)]
    for window_range in dataset.trajectory_window_ranges:
        containing = [bool(set(window_range) & group) for group in membership]
        assert sum(containing) == 1

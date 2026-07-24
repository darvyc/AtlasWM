"""Tests for CLI reproducibility metadata."""

from __future__ import annotations

import torch

from atlaswm import AtlasRegConfig, AtlasWM
from atlaswm.cli import _dataset_fingerprint, _save_checkpoint
from atlaswm.data import ToyTrajectoryDataset
from atlaswm.train import TrainState


def test_toy_dataset_fingerprint_is_deterministic():
    cfg = {
        "seed": 7,
        "data": {
            "name": "toy",
            "n_trajectories": 2,
            "traj_length": 4,
            "sub_length": 2,
        },
    }
    dataset_a = ToyTrajectoryDataset(
        n_trajectories=2,
        traj_length=4,
        sub_length=2,
        seed=7,
    )
    dataset_b = ToyTrajectoryDataset(
        n_trajectories=2,
        traj_length=4,
        sub_length=2,
        seed=7,
    )

    fingerprint_a = _dataset_fingerprint(dataset_a, cfg)
    fingerprint_b = _dataset_fingerprint(dataset_b, cfg)

    assert fingerprint_a == fingerprint_b
    assert fingerprint_a["kind"] == "toy"
    assert len(fingerprint_a["sha256"]) == 64


def test_checkpoint_contains_provenance(tmp_path):
    model = AtlasWM(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        action_dim=2,
        history_length=2,
        encoder_depth=1,
        encoder_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        reg_config=AtlasRegConfig(n_knots=5),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState(
        step=3,
        epoch=1,
        loss_history=[{"step": 3, "total": 1.0, "pred": 0.8, "reg": 0.2}],
    )
    checkpoint = tmp_path / "checkpoint.pt"

    _save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        cfg={"seed": 42, "data": {"name": "toy"}},
        state=state,
        dataset_fingerprint={"kind": "toy", "sha256": "0" * 64},
    )

    payload = torch.load(checkpoint, weights_only=False)
    assert payload["state"] == {"step": 3, "epoch": 1}
    assert payload["loss_history"] == state.loss_history
    assert payload["atlaswm_version"] == "1.0.0"
    assert payload["dataset_fingerprint"]["sha256"] == "0" * 64
    assert "python" in payload["rng_state"]
    assert "numpy" in payload["rng_state"]
    assert "torch_cpu" in payload["rng_state"]
    assert not checkpoint.with_suffix(".pt.tmp").exists()

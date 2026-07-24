from pathlib import Path

import torch
from torch.utils.data import DataLoader

from atlaswm.config import build_dataset, build_model, load_config
from atlaswm.training import (
    Trainer,
    TrainerConfig,
    TrainState,
    dataset_fingerprint,
    load_checkpoint,
    set_global_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_writes_and_restores_complete_checkpoint(tmp_path: Path):
    config = load_config(ROOT / "configs/smoke.yaml")
    set_global_seed(config["seed"], deterministic=True)
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainerConfig(
            epochs=1,
            lambda_reg=0.1,
            amp=False,
            log_every=0,
            validate_every=1,
            checkpoint_every=1,
            deterministic=True,
        ),
        output_dir=tmp_path,
        resolved_config=config,
        fingerprint=dataset_fingerprint(dataset),
    )
    state = trainer.fit(loader, validation_loader=loader)
    assert state.epoch == 1 and state.step > 0
    assert (tmp_path / "checkpoint_last.pt").is_file()
    assert (tmp_path / "checkpoint_best.pt").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()

    restored_model = build_model(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored = load_checkpoint(
        tmp_path / "checkpoint_last.pt",
        model=restored_model,
        optimizer=restored_optimizer,
    )
    assert isinstance(restored, TrainState)
    assert restored.step == state.step
    for first, second in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_epoch_sampler_reproduces_resume_order():
    from atlaswm.training import EpochRandomSampler

    dataset = list(range(12))
    uninterrupted = EpochRandomSampler(dataset, seed=19)
    uninterrupted.set_epoch(3)
    expected = list(uninterrupted)
    resumed = EpochRandomSampler(dataset, seed=19)
    resumed.set_epoch(3)
    assert list(resumed) == expected
    resumed.set_epoch(4)
    assert list(resumed) != expected

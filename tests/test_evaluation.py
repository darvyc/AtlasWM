from pathlib import Path

import torch
from torch.utils.data import DataLoader

from atlaswm.config import build_dataset, build_model, load_config
from atlaswm.evaluation import evaluate_loader, evaluate_prediction_batch, held_out_linear_probe


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_evaluation_and_probe():
    config = load_config(ROOT / "configs/smoke.yaml")
    dataset = build_dataset(config)
    model = build_model(config)
    batch = [dataset[index] for index in range(3)]
    observations = torch.stack([item[0] for item in batch])
    actions = torch.stack([item[1] for item in batch])
    metrics = evaluate_prediction_batch(model, observations, actions, max_horizon=2)
    assert "one_step_mse" in metrics
    assert "rollout_mse_h2" in metrics
    assert "effective_rank" in metrics
    with torch.no_grad():
        latent = model.encode(observations)
    probe = held_out_linear_probe(latent, torch.stack([item[2] for item in batch]))
    assert probe["linear_probe_train_samples"] > 0
    assert probe["linear_probe_test_samples"] > 0


def test_loader_evaluation():
    config = load_config(ROOT / "configs/smoke.yaml")
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=2)
    model = build_model(config)
    metrics = evaluate_loader(
        model, loader, torch.device("cpu"), max_batches=2, max_horizon=1
    )
    assert metrics["one_step_mse"] >= 0

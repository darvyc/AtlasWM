"""Train and inspect a compact AtlasWM instance on the toy environment."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from atlaswm import AtlasRegConfig, AtlasWM, ToyTrajectoryDataset
from atlaswm.evaluation import evaluate_prediction_batch, held_out_linear_probe
from atlaswm.training import TrainState, train_one_epoch


def main() -> None:
    torch.manual_seed(0)
    dataset = ToyTrajectoryDataset(
        n_trajectories=24,
        traj_length=24,
        sub_length=4,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=12, shuffle=True)
    model = AtlasWM(
        img_size=64,
        patch_size=8,
        embed_dim=32,
        action_dim=2,
        history_length=4,
        encoder_depth=2,
        encoder_heads=4,
        predictor_depth=2,
        predictor_heads=4,
        predictor_dropout=0.0,
        reg_config=AtlasRegConfig(
            rotation_mode="signed_permutation",
            kernel="two_scale",
            n_knots=17,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    state = TrainState()
    for _ in range(2):
        train_one_epoch(
            model,
            loader,
            optimizer,
            lambda_reg=0.1,
            device=torch.device("cpu"),
            state=state,
            log_every=20,
        )

    observations, actions, positions = next(iter(DataLoader(dataset, batch_size=16)))
    metrics = evaluate_prediction_batch(model, observations, actions, max_horizon=3)
    with torch.no_grad():
        latent = model.encode(observations)
    metrics.update(held_out_linear_probe(latent, positions))
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

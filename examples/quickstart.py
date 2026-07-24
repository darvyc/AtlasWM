"""Train a compact AtlasWM instance on the synthetic visual environment.

Usage:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswm import AtlasRegConfig, AtlasWM
from atlaswm.data import ToyTrajectoryDataset
from atlaswm.train import TrainState, train_one_epoch


def _held_out_linear_probe(z: torch.Tensor, positions: torch.Tensor) -> float:
    """Fit a linear probe on 80 percent of samples and report held-out R squared."""
    z = z - z.mean(dim=0, keepdim=True)
    positions = positions - positions.mean(dim=0, keepdim=True)
    split = max(1, int(0.8 * z.shape[0]))
    if split >= z.shape[0]:
        raise ValueError("linear probe requires at least two samples")

    z_train, z_test = z[:split], z[split:]
    p_train, p_test = positions[:split], positions[split:]
    weights = torch.linalg.lstsq(z_train, p_train).solution
    prediction = z_test @ weights
    residual = (prediction - p_test).square().sum()
    total = (p_test - p_test.mean(dim=0, keepdim=True)).square().sum()
    return float(1.0 - residual / total.clamp_min(1e-12))


def main() -> None:
    torch.manual_seed(0)

    regularizer_config = AtlasRegConfig(
        design="cross_polytope",
        rotate=True,
        deduplicate_antipodes=True,
        subspace_dim=1,
        target="gaussian",
        standardize_1d=False,
        estimator="biased",
        kernel="two_scale",
        lambda_1=0.5,
        lambda_2=2.0,
        alpha=0.5,
        n_knots=9,
    )
    model = AtlasWM(
        img_size=64,
        patch_size=8,
        embed_dim=64,
        action_dim=2,
        history_length=8,
        encoder_depth=3,
        encoder_heads=4,
        predictor_depth=2,
        predictor_heads=4,
        reg_config=regularizer_config,
    )
    print(f"AtlasWM parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Generating synthetic trajectories...")
    dataset = ToyTrajectoryDataset(
        n_trajectories=50,
        traj_length=32,
        sub_length=8,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    state = TrainState()

    print("Training for two epochs...")
    for _ in range(2):
        train_one_epoch(
            model,
            loader,
            optimizer,
            lambda_reg=0.1,
            device=torch.device("cpu"),
            state=state,
            grad_clip=1.0,
            log_every=20,
        )

    print("Evaluating a held-out linear state probe...")
    model.eval()
    latents = []
    positions = []
    with torch.no_grad():
        evaluation_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        for observations, _, state_positions in evaluation_loader:
            encoded = model.encode(observations)
            latents.append(encoded.reshape(-1, encoded.shape[-1]))
            positions.append(state_positions.reshape(-1, 2))

    latent_matrix = torch.cat(latents)
    position_matrix = torch.cat(positions)
    r_squared = _held_out_linear_probe(latent_matrix, position_matrix)
    print(f"Held-out linear-probe R^2: {r_squared:.3f}")
    print("This value is an integration diagnostic, not a benchmark claim.")


if __name__ == "__main__":
    main()

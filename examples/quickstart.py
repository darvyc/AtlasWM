"""Quickstart example: train a tiny AtlasWM on the toy 2D environment.

This runs in ~a minute on CPU and verifies the whole pipeline works.

Usage:
    python examples/quickstart.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswm import AtlasWM, AtlasRegConfig
from atlaswm.data import ToyTrajectoryDataset
from atlaswm.train import TrainState, train_one_epoch


def main():
    torch.manual_seed(0)

    # Tiny model for CPU
    reg_cfg = AtlasRegConfig(
        design="cross_polytope",
        subspace_dim=1,
        target="gaussian",
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
        reg_config=reg_cfg,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AtlasWM parameters: {n_params:,}")

    print("Generating toy dataset...")
    ds = ToyTrajectoryDataset(n_trajectories=50, traj_length=32, sub_length=8)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    state = TrainState()

    print("Training for 2 epochs on toy data...")
    for epoch in range(2):
        train_one_epoch(
            model,
            loader,
            opt,
            lambda_reg=0.1,
            device=torch.device("cpu"),
            state=state,
            grad_clip=1.0,
            log_every=20,
        )

    # Probe the learned latent space: can we read 2D position from z?
    print("\nProbing latent space for agent position...")
    model.eval()
    with torch.no_grad():
        all_z = []
        all_pos = []
        for obs, _, pos in loader:
            z = model.encode(obs)  # (B, T, D)
            all_z.append(z.reshape(-1, z.shape[-1]))
            all_pos.append(pos.reshape(-1, 2))
        Z = torch.cat(all_z)
        P = torch.cat(all_pos)

    # Center
    Z_c = Z - Z.mean(0, keepdim=True)
    P_c = P - P.mean(0, keepdim=True)

    # Linear probe via least-squares: find W so W z ≈ p
    # P_c ≈ Z_c @ W -> W = (Z_c^T Z_c)^{-1} Z_c^T P_c
    W, _, _, _ = torch.linalg.lstsq(Z_c, P_c)
    P_pred = Z_c @ W
    ss_res = ((P_pred - P_c) ** 2).sum()
    ss_tot = (P_c ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"Linear probe R^2 for agent position: {r2.item():.3f}")
    print(
        "(Higher is better. With sufficient training, R^2 should exceed 0.9, "
        "confirming the latent space encodes physical state.)"
    )


if __name__ == "__main__":
    main()

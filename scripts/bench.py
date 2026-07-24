"""Matched forward/backward estimator benchmark with synchronization."""

from __future__ import annotations

import argparse
import json
import time

import torch

from atlaswm.regularizer import AtlasReg, AtlasRegConfig


def measure(name: str, regularizer: AtlasReg, latent: torch.Tensor, iterations: int) -> dict:
    for _ in range(5):
        regularizer(latent).backward()
        latent.grad = None
    if latent.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        loss = regularizer(latent)
        loss.backward()
        latent.grad = None
    if latent.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "milliseconds_per_iteration": elapsed * 1000.0 / iterations,
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    latent = torch.randn(args.batch_size, args.dim, device=device, requires_grad=True)
    configurations = {
        "iid_haar_1024": AtlasRegConfig(
            design="haar",
            n_haar_projections=1024,
            rotation_mode="none",
            kernel="single",
        ),
        "atlas_haar_frame": AtlasRegConfig(
            design="cross_polytope",
            rotation_mode="haar",
            rotation_refresh_steps=16,
        ),
        "atlas_fast_frame": AtlasRegConfig(
            design="cross_polytope",
            rotation_mode="signed_permutation",
        ),
        "atlas_4d_bhep": AtlasRegConfig(
            subspace_dim=4,
            n_subspaces=4,
            hz_beta=1.0,
        ),
    }
    records = [
        measure(name, AtlasReg(args.dim, config).to(device), latent, args.iterations)
        for name, config in configurations.items()
    ]
    print(json.dumps({"device": str(device), "records": records}, indent=2))


if __name__ == "__main__":
    main()

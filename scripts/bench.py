"""Benchmark the principal AtlasReg estimator configurations.

This script measures forward plus backward time. It does not measure control
quality or establish a statistical-power advantage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswm.regularizer import AtlasReg, AtlasRegConfig


def benchmark(
    name: str,
    cfg: AtlasRegConfig,
    dim: int,
    batch_size: int,
    n_iters: int,
    device: torch.device,
) -> None:
    reg = AtlasReg(dim, cfg).to(device)
    z = torch.randn(batch_size, dim, device=device, requires_grad=True)

    for _ in range(5):
        reg(z).backward()
        z.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        loss = reg(z)
        loss.backward()
        z.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    milliseconds = elapsed / n_iters * 1000.0
    print(f"{name:<42s} {milliseconds:8.2f} ms/iter  loss={loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(
        f"Benchmark: dim={args.dim}, batch={args.batch_size}, "
        f"iters={args.n_iters}, device={device}"
    )
    print("-" * 90)

    benchmark(
        "Haar 1D quadrature, 1024 directions",
        AtlasRegConfig(
            design="haar",
            n_haar_projections=1024,
            rotate=False,
            subspace_dim=1,
            standardize_1d=False,
            kernel="single",
            lambda_=1.0,
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )
    benchmark(
        "Rotated basis 1D two-scale quadrature",
        AtlasRegConfig(
            design="cross_polytope",
            rotate=True,
            deduplicate_antipodes=True,
            subspace_dim=1,
            standardize_1d=False,
            kernel="two_scale",
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )
    benchmark(
        "Random 4D subspace, raw fixed-beta BHEP",
        AtlasRegConfig(
            subspace_dim=4,
            n_subspaces=1,
            whiten_kd=False,
            hz_beta=1.0,
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )
    benchmark(
        "Random 4D subspace, HZ-style shape test",
        AtlasRegConfig(
            subspace_dim=4,
            n_subspaces=1,
            whiten_kd=True,
            hz_beta=None,
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )


if __name__ == "__main__":
    main()

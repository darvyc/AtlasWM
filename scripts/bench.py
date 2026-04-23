"""Benchmark AtlasReg against its SIGReg-equivalent configuration.

Compares forward-pass time and gradient-flow correctness for:
  - Baseline: Haar sampling, 1024 projections, single-scale kernel (~ SIGReg)
  - Atlas:   cross-polytope + rotation, two-scale kernel (defaults)
  - Atlas-HZ: subspace_dim=4, Henze-Zirkler closed form
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
):
    reg = AtlasReg(dim, cfg).to(device)
    z = torch.randn(batch_size, dim, device=device, requires_grad=True)

    # Warm up
    for _ in range(5):
        loss = reg(z)
        loss.backward()
        z.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        loss = reg(z)
        loss.backward()
        z.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    per_iter_ms = (t1 - t0) / n_iters * 1000
    print(f"{name:<30s}  {per_iter_ms:7.2f} ms/iter   final loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(
        f"Benchmark: dim={args.dim}, batch={args.batch_size}, "
        f"iters={args.n_iters}, device={device}"
    )
    print("-" * 70)

    # SIGReg-equivalent baseline
    benchmark(
        "SIGReg-equivalent (Haar, 1D, single-scale)",
        AtlasRegConfig(
            design="haar",
            n_haar_projections=1024,
            rotate=False,
            subspace_dim=1,
            kernel="single",
            lambda_=1.0,
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )

    # AtlasReg default
    benchmark(
        "AtlasReg (cross-polytope, two-scale)",
        AtlasRegConfig(
            design="cross_polytope",
            rotate=True,
            subspace_dim=1,
            kernel="two_scale",
        ),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )

    # AtlasReg k-D (Henze-Zirkler)
    benchmark(
        "AtlasReg k=4 (Henze-Zirkler)",
        AtlasRegConfig(subspace_dim=4, hz_beta=1.0),
        args.dim,
        args.batch_size,
        args.n_iters,
        device,
    )


if __name__ == "__main__":
    main()

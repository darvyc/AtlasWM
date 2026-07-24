"""Generate machine-readable statistical verification values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from atlaswm.designs import cross_polytope
from atlaswm.statistics import (
    gaussian_bhep_discrepancy,
    gaussian_bhep_null_floor,
    spherical_projection_even_moment_coefficient,
    student_t_unit_variance_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", default="outputs/statistical_verification.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    design = cross_polytope(args.dim, dtype=torch.float64)
    second_moment = design.t() @ design / design.shape[0]
    target_second = torch.eye(args.dim, dtype=torch.float64) / args.dim
    vector = torch.arange(1, args.dim + 1, dtype=torch.float64)
    design_fourth = (design @ vector).pow(4).mean()
    sphere_fourth = (
        spherical_projection_even_moment_coefficient(args.dim, 4)
        * vector.norm().pow(4)
    )
    biased, unbiased = [], []
    for trial in range(args.trials):
        generator = torch.Generator().manual_seed(args.seed + trial)
        values = torch.randn(args.samples, 2, generator=generator)
        biased.append(gaussian_bhep_discrepancy(values))
        unbiased.append(gaussian_bhep_discrepancy(values, estimator="unbiased"))
    payload = {
        "seed": args.seed,
        "cross_polytope_second_moment_max_error": float(
            (second_moment - target_second).abs().max()
        ),
        "cross_polytope_fourth_moment": float(design_fourth),
        "spherical_fourth_moment": float(sphere_fourth),
        "gaussian_null_floor": gaussian_bhep_null_floor(args.samples, 2, 1.0),
        "biased_monte_carlo_mean": float(torch.stack(biased).mean()),
        "unbiased_monte_carlo_mean": float(torch.stack(unbiased).mean()),
        "student_t_nu5_unit_variance_scale": student_t_unit_variance_scale(5.0),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

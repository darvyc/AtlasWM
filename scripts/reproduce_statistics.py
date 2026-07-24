"""Reproduce the principal finite-sample and spherical-design identities.

The script writes a machine-readable record suitable for software-artifact
verification. It does not train AtlasWM or require an external dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from atlaswm.designs import cross_polytope
from atlaswm.statistics import (
    gaussian_bhep_discrepancy,
    gaussian_bhep_null_floor,
    spherical_projection_even_moment_coefficient,
    student_t_unit_variance_scale,
)


def _mean_and_standard_error(values: torch.Tensor) -> tuple[float, float]:
    mean = float(values.mean())
    if values.numel() < 2:
        return mean, 0.0
    standard_error = float(values.std(unbiased=True) / math.sqrt(values.numel()))
    return mean, standard_error


def run_verification(
    *,
    seed: int,
    dim: int,
    samples: int,
    trials: int,
    beta: float,
    student_t_nu: float,
) -> dict[str, Any]:
    """Return a JSON-serializable statistical verification record."""
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if trials < 1:
        raise ValueError("trials must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if student_t_nu <= 2:
        raise ValueError("student_t_nu must exceed 2 for unit variance")

    dtype = torch.float64
    design = cross_polytope(dim, dtype=dtype)
    second_moment = design.t() @ design / design.shape[0]
    expected_second_moment = torch.eye(dim, dtype=dtype) / dim
    second_moment_max_error = float(
        (second_moment - expected_second_moment).abs().max()
    )

    x = torch.arange(1, dim + 1, dtype=dtype)
    x = x / x.norm()
    design_fourth_moment = float(((design @ x) ** 4).mean())
    spherical_fourth_moment = float(
        spherical_projection_even_moment_coefficient(dim, order=4)
        * x.norm().pow(4)
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    biased_values = []
    unbiased_values = []
    for _ in range(trials):
        y = torch.randn(samples, dim, generator=generator, dtype=dtype)
        biased_values.append(
            gaussian_bhep_discrepancy(y, beta=beta, estimator="biased")
        )
        unbiased_values.append(
            gaussian_bhep_discrepancy(y, beta=beta, estimator="unbiased")
        )

    biased_tensor = torch.stack(biased_values)
    unbiased_tensor = torch.stack(unbiased_values)
    biased_mean, biased_standard_error = _mean_and_standard_error(biased_tensor)
    unbiased_mean, unbiased_standard_error = _mean_and_standard_error(
        unbiased_tensor
    )
    analytic_floor = gaussian_bhep_null_floor(samples, dim, beta)

    return {
        "schema_version": 1,
        "configuration": {
            "seed": seed,
            "dim": dim,
            "samples": samples,
            "trials": trials,
            "beta": beta,
            "student_t_nu": student_t_nu,
            "torch_version": torch.__version__,
            "dtype": str(dtype),
        },
        "cross_polytope": {
            "points": int(design.shape[0]),
            "second_moment_max_error": second_moment_max_error,
            "design_fourth_moment": design_fourth_moment,
            "spherical_fourth_moment": spherical_fourth_moment,
            "fourth_moment_gap": design_fourth_moment
            - spherical_fourth_moment,
        },
        "gaussian_bhep": {
            "analytic_biased_null_floor": analytic_floor,
            "monte_carlo_biased_mean": biased_mean,
            "monte_carlo_biased_standard_error": biased_standard_error,
            "biased_mean_minus_floor": biased_mean - analytic_floor,
            "monte_carlo_unbiased_mean": unbiased_mean,
            "monte_carlo_unbiased_standard_error": unbiased_standard_error,
        },
        "student_t": {
            "unit_variance_scale": student_t_unit_variance_scale(student_t_nu),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce AtlasWM statistical identities."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--student-t-nu", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_verification(
        seed=args.seed,
        dim=args.dim,
        samples=args.samples,
        trials=args.trials,
        beta=args.beta,
        student_t_nu=args.student_t_nu,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()

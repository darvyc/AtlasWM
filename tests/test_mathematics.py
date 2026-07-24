import math

import pytest
import torch

from atlaswm.designs import cross_polytope, haar_rotation, simplex
from atlaswm.regularizer import AtlasReg, AtlasRegConfig
from atlaswm.statistics import (
    empirical_cf_discrepancy,
    gaussian_bhep_discrepancy,
    gaussian_bhep_null_floor,
    henze_zirkler_beta,
    spherical_projection_even_moment_coefficient,
    student_t_unit_variance_scale,
)
from atlaswm.targets import StandardGaussian, StudentT


def test_cross_polytope_second_moment_and_antipodal_loss():
    dim = 8
    design = cross_polytope(dim, dtype=torch.float64)
    moment = design.t() @ design / design.shape[0]
    assert torch.allclose(moment, torch.eye(dim, dtype=torch.float64) / dim)
    torch.manual_seed(4)
    latent = torch.randn(128, dim)
    full = AtlasReg(
        dim,
        AtlasRegConfig(
            rotation_mode="none",
            deduplicate_antipodes=False,
            kernel="single",
        ),
    )(latent)
    quotient = AtlasReg(
        dim,
        AtlasRegConfig(
            rotation_mode="none",
            deduplicate_antipodes=True,
            kernel="single",
        ),
    )(latent)
    assert torch.allclose(full, quotient, atol=1e-6, rtol=1e-5)


def test_cross_polytope_is_not_four_design():
    dim = 8
    vector = torch.arange(1, dim + 1, dtype=torch.float64)
    design_value = vector.pow(4).sum() / dim
    sphere_value = (
        spherical_projection_even_moment_coefficient(dim, 4) * vector.norm().pow(4)
    )
    assert not torch.isclose(design_value, sphere_value)


def test_simplex_and_haar_rotation_geometry():
    points = simplex(5, dtype=torch.float64)
    assert points.shape == (6, 5)
    assert torch.allclose(points.norm(dim=-1), torch.ones(6, dtype=torch.float64))
    rotation = haar_rotation(7, dtype=torch.float64)
    assert torch.allclose(rotation.t() @ rotation, torch.eye(7, dtype=torch.float64), atol=1e-10)


def test_gaussian_closed_form_matches_dense_frequency_integral():
    torch.manual_seed(1)
    samples = torch.randn(96, dtype=torch.float64)
    beta = 0.8
    exact = gaussian_bhep_discrepancy(samples, beta=beta)
    nodes = torch.linspace(-10.0, 10.0, 10001, dtype=torch.float64)
    empirical = torch.exp(1j * nodes[:, None] * samples[None]).mean(dim=1)
    target = torch.exp(-0.5 * nodes.square())
    weight = torch.exp(-0.5 * (nodes / beta).square())
    integral = torch.trapezoid(weight * (empirical - target).abs().square(), nodes)
    normalized = integral / (math.sqrt(2.0 * math.pi) * beta)
    assert torch.allclose(exact, normalized, atol=3e-5, rtol=3e-4)


def test_chunked_bhep_matches_full():
    torch.manual_seed(2)
    values = torch.randn(73, 4, dtype=torch.float64)
    full = gaussian_bhep_discrepancy(values, beta=1.3)
    chunked = gaussian_bhep_discrepancy(values, beta=1.3, pair_chunk_size=11)
    assert torch.allclose(full, chunked, atol=1e-12, rtol=1e-12)


def test_null_floor_and_unbiased_expectation():
    biased_values = []
    unbiased_values = []
    for seed in range(60):
        generator = torch.Generator().manual_seed(seed)
        samples = torch.randn(64, 2, generator=generator)
        biased_values.append(gaussian_bhep_discrepancy(samples))
        unbiased_values.append(gaussian_bhep_discrepancy(samples, estimator="unbiased"))
    expected = gaussian_bhep_null_floor(64, 2, 1.0)
    assert abs(float(torch.stack(biased_values).mean()) - expected) < 0.005
    assert abs(float(torch.stack(unbiased_values).mean())) < 0.005


def test_empirical_cf_unbiased_expectation():
    nodes = torch.tensor([0.2, 0.8, 1.4], dtype=torch.float64)
    target = torch.exp(-0.5 * nodes.square())
    weights = torch.tensor([0.4, 0.7, 0.3], dtype=torch.float64)
    values = []
    for seed in range(80):
        samples = torch.randn(64, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        values.append(
            empirical_cf_discrepancy(
                samples,
                nodes,
                target,
                weights,
                estimator="unbiased",
                frequency_chunk_size=2,
            )
        )
    assert torch.stack(values).mean().abs() < 0.005


def test_henze_zirkler_and_student_t_helpers():
    expected = 2.0 ** -0.5 * (((2 * 4 + 1) * 100 / 4) ** (1 / 8))
    assert henze_zirkler_beta(100, 4) == pytest.approx(expected)
    scale = student_t_unit_variance_scale(5.0)
    assert scale * scale * 5.0 / 3.0 == pytest.approx(1.0)
    nodes = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    student = StudentT.precompute_char_fn_1d(1000.0, nodes)
    gaussian = StandardGaussian().char_fn_1d(nodes)
    assert torch.isfinite(student).all()
    assert torch.allclose(student, gaussian, rtol=0.05, atol=1e-8)

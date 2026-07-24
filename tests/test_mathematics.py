import math

import pytest
import torch

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


def test_cross_polytope_antipodal_dedup_is_exact_for_symmetric_target():
    torch.manual_seed(3)
    z = torch.randn(256, 12)
    full = AtlasReg(
        12,
        AtlasRegConfig(
            design="cross_polytope",
            rotate=False,
            deduplicate_antipodes=False,
            kernel="single",
        ),
    )(z)
    dedup = AtlasReg(
        12,
        AtlasRegConfig(
            design="cross_polytope",
            rotate=False,
            deduplicate_antipodes=True,
            kernel="single",
        ),
    )(z)
    assert torch.allclose(full, dedup, atol=1e-7, rtol=1e-6)


def test_cross_polytope_is_not_a_four_design():
    d = 8
    x = torch.arange(1, d + 1, dtype=torch.float64)
    design_fourth = x.pow(4).sum() / d
    sphere_fourth = (
        spherical_projection_even_moment_coefficient(d, 4)
        * x.norm().pow(4)
    )
    assert not torch.isclose(design_fourth, sphere_fourth)


def test_gaussian_closed_form_matches_dense_frequency_integral():
    torch.manual_seed(1)
    x = torch.randn(128, dtype=torch.float64)
    beta = 0.8
    exact = gaussian_bhep_discrepancy(x, beta=beta)

    nodes = torch.linspace(-10.0, 10.0, 12001, dtype=torch.float64)
    empirical = torch.exp(1j * nodes[:, None] * x[None, :]).mean(dim=1)
    target = torch.exp(-0.5 * nodes.square())
    weight = torch.exp(-0.5 * (nodes / beta).square())
    integral = torch.trapz(weight * (empirical - target).abs().square(), nodes)
    normalized = integral / (math.sqrt(2.0 * math.pi) * beta)
    assert torch.allclose(exact, normalized, atol=2e-5, rtol=2e-4)


def test_biased_gaussian_null_floor_matches_monte_carlo():
    n, k, beta = 64, 2, 1.0
    values = []
    for seed in range(100):
        generator = torch.Generator().manual_seed(seed)
        y = torch.randn(n, k, generator=generator)
        values.append(gaussian_bhep_discrepancy(y, beta=beta))
    empirical = torch.stack(values).mean().item()
    theoretical = gaussian_bhep_null_floor(n, k, beta)
    assert abs(empirical - theoretical) < 0.004


def test_unbiased_bhep_removes_null_floor_in_expectation():
    values = []
    for seed in range(120):
        generator = torch.Generator().manual_seed(seed)
        y = torch.randn(48, 1, generator=generator)
        values.append(
            gaussian_bhep_discrepancy(y, beta=1.0, estimator="unbiased")
        )
    assert abs(torch.stack(values).mean().item()) < 0.004


def test_quadrature_unbiased_formula_has_zero_target_expectation():
    nodes = torch.tensor([0.25, 0.75, 1.5], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.75, 0.25], dtype=torch.float64)
    target = torch.exp(-0.5 * nodes.square())
    values = []
    for seed in range(150):
        generator = torch.Generator().manual_seed(seed)
        samples = torch.randn(64, 1, generator=generator, dtype=torch.float64)
        values.append(
            empirical_cf_discrepancy(
                samples,
                nodes,
                target,
                weights,
                estimator="unbiased",
            ).squeeze()
        )
    assert abs(torch.stack(values).mean().item()) < 0.004


def test_true_target_matching_detects_point_collapse():
    torch.manual_seed(0)
    reg = AtlasReg(
        16,
        AtlasRegConfig(
            rotate=False,
            standardize_1d=False,
            kernel="single",
            n_knots=65,
        ),
    )
    healthy = reg(torch.randn(512, 16))
    collapsed = reg(torch.zeros(512, 16))
    assert collapsed > healthy


def test_unbiased_estimator_rejects_batch_standardization():
    with pytest.raises(ValueError):
        AtlasReg(
            8,
            AtlasRegConfig(estimator="unbiased", standardize_1d=True),
        )


def test_henze_zirkler_bandwidth_formula():
    n, k = 100, 4
    expected = 2.0 ** -0.5 * (((2 * k + 1) * n / 4) ** (1 / (k + 4)))
    assert henze_zirkler_beta(n, k) == pytest.approx(expected)


def test_student_t_large_nu_is_stable_and_gaussian_limit():
    pytest.importorskip("scipy")
    nodes = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    phi_t = StudentT.precompute_char_fn_1d(1000.0, nodes)
    phi_g = StandardGaussian().char_fn_1d(nodes)
    assert torch.isfinite(phi_t).all()
    assert torch.allclose(phi_t, phi_g, rtol=0.05, atol=1e-8)


def test_student_t_unit_variance_scale():
    nu = 5.0
    scale = student_t_unit_variance_scale(nu)
    assert scale * scale * nu / (nu - 2.0) == pytest.approx(1.0)


def test_regularizer_closed_form_matches_dense_quadrature_scale():
    torch.manual_seed(9)
    z = torch.randn(192, 5, dtype=torch.float64)
    common = dict(
        design="cross_polytope",
        rotate=False,
        deduplicate_antipodes=True,
        target="gaussian",
        standardize_1d=False,
        estimator="biased",
        kernel="single",
        lambda_=0.75,
    )
    closed = AtlasReg(5, AtlasRegConfig(**common, one_d_backend="closed_form"))
    quad = AtlasReg(
        5,
        AtlasRegConfig(**common, one_d_backend="quadrature", n_knots=17),
    )
    nodes = torch.linspace(0.0, 8.0, 8001, dtype=torch.float64)
    dt = nodes[1] - nodes[0]
    weights = torch.full_like(nodes, 2.0 * dt)
    weights[0] = dt
    weights[-1] = dt
    weights = weights * torch.exp(-0.5 * (nodes / common["lambda_"]).square())
    quad.quad_nodes = nodes
    quad.quad_iweights = weights
    quad.target_cf = torch.exp(-0.5 * nodes.square())
    assert torch.allclose(closed(z), quad(z), atol=3e-5, rtol=3e-4)

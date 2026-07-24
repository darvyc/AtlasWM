# AtlasWM

**End-to-end JEPA world models with structured characteristic-function distribution matching. Based on LeWorldModel by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

AtlasWM is a compact Joint-Embedding Predictive Architecture trained from raw pixels without an EMA target encoder, stop-gradient, or pretrained visual backbone. Its regularizer, AtlasReg, matches latent projections to a chosen target through empirical characteristic functions.

The implementation now distinguishes exact mathematics from finite approximation:

- The population sliced characteristic-function objective identifies a distribution.
- A finite projection and frequency rule is a training approximation, not a proof of equality in distribution.
- A rotated cross-polytope is exact for spherical polynomials through degree 3 and unbiased over random rotations for general integrands.
- Antipodal projections are redundant for symmetric targets, so the default computes `d` distinct projection lines rather than evaluating the same loss at `2d` vertices.
- Finite Gaussian batches have a known positive biased-estimator floor. An unbiased U-statistic option is available.
- Gaussian-weighted CF matching has an exact BHEP/Henze-Zirkler closed form for validation and multivariate subspaces.

See [`docs/theory.md`](docs/theory.md) for the complete derivations and guarantee table.

## What AtlasReg adds

| Axis | Options |
|---|---|
| Projection rule | Rotated cross-polytope, simplex, or iid Haar directions |
| Statistical target | Isotropic Gaussian or scaled Student-t in the 1D path |
| Estimator | Biased non-negative empirical discrepancy or unbiased U-statistic |
| Frequency integration | Fast quadrature or exact Gaussian closed form |
| Matching mode | Raw target matching or batch-standardized shape testing |
| Subspace dimension | 1D projected ECF or k-D BHEP/Henze-Zirkler discrepancy |

At latent dimension `d=192`, the symmetric-target cross-polytope path uses `192` distinct antipodal lines. The full `384`-vertex cross-polytope remains the spherical 3-design, but opposite vertices give exactly the same squared CF loss.

## Install

```bash
git clone https://github.com/darvyc/AtlasWM.git
cd AtlasWM
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+. SciPy is needed for the Student-t target.

## Quickstart

```python
import torch
from atlaswm import AtlasWM, AtlasRegConfig

model = AtlasWM(
    img_size=224,
    patch_size=14,
    embed_dim=192,
    action_dim=2,
    history_length=3,
    reg_config=AtlasRegConfig(
        design="cross_polytope",
        rotate=True,
        deduplicate_antipodes=True,
        target="gaussian",
        standardize_1d=False,
        estimator="biased",
        kernel="two_scale",
        subspace_dim=1,
    ),
)

# obs:     (B, T, 3, H, W)
# actions: (B, T, action_dim)
losses = model.training_step(obs, actions, lambda_reg=0.1)
losses["total"].backward()
```

Plan towards a visual goal:

```python
from atlaswm.planning import CEMPlanner

planner = CEMPlanner(model, horizon=5, n_samples=300, n_iters=30)
actions = planner.plan(current_obs, goal_obs)
```

The rollout implementation aligns action `a_t` with latent `z_t` when predicting `z_(t+1)`. Earlier versions shifted candidate actions by one step during planning.

## Regularizer modes

### True Gaussian target matching

Raw projections retain mean, variance, and shape information:

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=False,
    estimator="biased",
)
```

This is the default.

### Studentized projection shape testing

Projection studentization deliberately removes location and scale:

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=True,
    estimator="biased",
)
```

This tests standardized marginal shape. It does not enforce latent covariance `I`.

### Unbiased finite-sample diagnostic

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=False,
    estimator="unbiased",
)
```

The unbiased estimate has the correct population expectation but can be negative on a finite batch. It is intentionally incompatible with batch-dependent standardization and whitening.

### Exact Gaussian closed form

```python
AtlasRegConfig(
    target="gaussian",
    one_d_backend="closed_form",
    kernel="single",
    lambda_=1.0,
)
```

This integrates all frequencies analytically but costs `O(MN^2)`, so quadrature remains the practical default.

### Multivariate BHEP / Henze-Zirkler

```python
AtlasRegConfig(
    subspace_dim=4,
    n_subspaces=4,
    target="gaussian",
    whiten_kd=False,
    hz_beta=1.0,
)
```

Set `whiten_kd=True` for covariance-standardized normal-shape testing within each sampled subspace. Set `hz_beta=None` to use the classical sample-size-dependent Henze-Zirkler bandwidth.

### Student-t target

```python
from atlaswm.statistics import student_t_unit_variance_scale

nu = 5.0
AtlasRegConfig(
    target="student_t",
    student_t_nu=nu,
    student_t_scale=student_t_unit_variance_scale(nu),
)
```

A scale-one Student-t has variance `nu/(nu-2)` when `nu>2`. Heavy tails do not by themselves imply low intrinsic dimension.

## Repository layout

```text
atlaswm/
├── atlaswm/
│   ├── designs.py       # Spherical designs and Haar rotations
│   ├── statistics.py    # BHEP, HZ bandwidth, null floor, moment formulae
│   ├── targets.py       # Gaussian and numerically stable Student-t CFs
│   ├── kernels.py       # Gaussian frequency quadrature
│   ├── regularizer.py   # AtlasReg
│   ├── encoder.py       # ViT encoder
│   ├── predictor.py     # Causal action-conditioned predictor
│   ├── model.py         # End-to-end objective
│   ├── planning/        # CEM latent planner
│   └── data.py          # Synthetic trajectory scaffold
├── tests/
├── configs/
├── docs/theory.md
├── examples/
└── scripts/
```

## Testing

```bash
pytest -v tests/
```

The mathematical tests cover:

- exact cross-polytope second and third moments;
- the explicit fourth-moment limitation of a 3-design;
- equality of full and antipodally deduplicated symmetric-target losses;
- agreement between dense numerical integration and the Gaussian BHEP closed form;
- the analytic finite-sample Gaussian null floor;
- biased versus unbiased estimators;
- stable Student-t CF evaluation at large degrees of freedom;
- current-action alignment in autoregressive rollout;
- end-to-end gradient flow.

## Scope and validation status

AtlasWM is a research scaffold, not yet a reproduced benchmark result. The repository includes a synthetic image trajectory environment and unit tests, but not full PushT, OGBench, DMControl, or LeWorldModel reproduction runs. Claims of improved control performance require fixed-budget experiments with seeds, confidence intervals, and published checkpoints.

## Citation

```bibtex
@software{atlaswm2026,
  title  = {AtlasWM: JEPA World Models with Structured Distribution Matching},
  author = {{Darvy C.}},
  year   = {2026},
  url    = {https://github.com/darvyc/AtlasWM},
}
```

## License

MIT. See [LICENSE](LICENSE).

# AtlasWM

**Stable end-to-end JEPA world models via deterministic distribution matching. Based off LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

AtlasWM is a compact Joint-Embedding Predictive Architecture that trains end-to-end from raw pixels without EMAs, stop-gradients, pretrained encoders, or auxiliary losses. The anti-collapse regularizer — *AtlasReg* — replaces the stochastic 1D SIGReg of LeWM with a deterministic, multi-scale, multivariate test that costs less and detects more.

> An atlas in differential geometry is a collection of charts that together cover a manifold. AtlasReg covers the distribution of your latents with a deterministic family of charts (projections), each testing distributional match against the target.

---

## What's new vs. SIGReg / LeWM

| | SIGReg (LeWM) | AtlasReg (this repo) |
|---|---|---|
| Projections | `M = 1024` random Haar | `M = 2d` cross-polytope + random rotation |
| Moment matching | Stochastic, rate `O(1/√M)` | **Deterministic** for moments ≤ 3 |
| Subspace dimension | 1 (Epps–Pulley) | Configurable `k ∈ {1, 2, 4, ...}` (Henze–Zirkler) |
| Kernel weight | Single Gaussian `w(t) = exp(−t²/(2λ²))` | Two-scale Gaussian (body + tail) |
| Target | Isotropic `N(0, I)` | Gaussian **or** isotropic Student-t |
| Cost at d=192 | 1024 projections | **384 projections** (~2.7× reduction) |

The theoretical picture: Cramér–Wold (1936) guarantees that a high-dimensional distribution is determined by its 1D projections. Spherical *t*-designs ([Delsarte–Goethals–Seidel, 1977](https://www.sciencedirect.com/science/article/pii/S0723086977800049)) let us replace Monte Carlo sampling over the sphere with a finite set that integrates all polynomials of degree ≤ *t* **exactly**. The cross-polytope (2*d* vertices ±eᵢ) is a 3-design — combined with a per-step random rotation it gives axis-alignment-free deterministic matching of the first three moments.

See [`docs/theory.md`](docs/theory.md) for the derivations.

---

## Install

```bash
git clone https://github.com/darvyc/atlaswm.git
cd atlaswm
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch 2.0+. SciPy is optional (needed only for the Student-t target).

---

## Quickstart

Train a world model on any (`observations`, `actions`) trajectory dataset:

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
        design='cross_polytope',   # deterministic 3-design
        kernel='two_scale',         # body + tail sensitivity
        target='gaussian',          # or 'student_t'
        subspace_dim=1,             # 1 for EP, 4 for multivariate HZ
    ),
)

# obs:     (B, T, 3, H, W)
# actions: (B, T, action_dim)
loss_dict = model.training_step(obs, actions, lambda_reg=0.1)
loss_dict['total'].backward()
```

Plan actions toward a goal at inference time:

```python
from atlaswm.planning import CEMPlanner

planner = CEMPlanner(model, horizon=5, n_samples=300, n_iters=30)
actions = planner.plan(current_obs, goal_obs)
```

Or run the shipped training script:

```bash
python scripts/train.py --config configs/pusht.yaml
```

---

## Repository layout

```
atlaswm/
├── atlaswm/
│   ├── designs.py       # Spherical t-designs (cross-polytope, simplex, Haar)
│   ├── targets.py       # Gaussian and Student-t target distributions
│   ├── kernels.py       # Single- and two-scale Gaussian quadrature kernels
│   ├── regularizer.py   # AtlasReg — the anti-collapse regularizer
│   ├── encoder.py       # ViT-Tiny encoder
│   ├── predictor.py     # Causal transformer predictor with AdaLN action conditioning
│   ├── model.py         # AtlasWM end-to-end model
│   ├── planning/        # CEM-based latent-space planner
│   └── data.py          # Trajectory dataset and synthetic toy environment
├── tests/               # pytest suite — run with `pytest -v`
├── configs/             # YAML configs per environment
├── docs/theory.md       # Math derivations (Cramér-Wold, t-designs, HZ)
├── examples/            # Minimal runnable examples
└── scripts/             # Training, evaluation, benchmarking CLIs
```

---

## Four axes of generalization

The regularizer exposes four orthogonal knobs. Each can be ablated independently.

### 1. Projection scheme

```python
AtlasRegConfig(design='cross_polytope', rotate=True)   # default — 2d vertices, deterministic
AtlasRegConfig(design='simplex', rotate=True)          # d+1 vertices (2-design)
AtlasRegConfig(design='haar', n_haar_projections=1024) # baseline — Monte Carlo
```

The cross-polytope is a **spherical 3-design**: the sign symmetry kills all odd moments automatically, and the {±eᵢ} set integrates quadratic forms exactly. Random rotation per step prevents the encoder from exploiting coordinate-aligned structure.

### 2. Kernel weight

Body-vs-tail sensitivity is controlled by the weight `w(t)` in the Epps–Pulley integral. Small λ concentrates near `t=0` → low-moment sensitivity (body). Large λ probes the tails.

```python
AtlasRegConfig(kernel='single',    lambda_=1.0)                           # standard
AtlasRegConfig(kernel='two_scale', lambda_1=0.5, lambda_2=2.0, alpha=0.5) # simultaneous body+tail
```

Two-scale costs one extra kernel evaluation at quadrature nodes — zero runtime difference.

### 3. Target distribution

Cramér–Wold is target-agnostic. For environments with low intrinsic dimensionality (where isotropic Gaussian in high-d is a bad prior), Student-t with moderate ν performs better:

```python
AtlasRegConfig(target='gaussian')
AtlasRegConfig(target='student_t', student_t_nu=5.0)  # heavier tails, tighter body
```

### 4. Subspace dimension

1D projections are blind to joint structure. The Henze–Zirkler test is the natural multivariate lift, with a closed-form expression (so no quadrature needed):

```python
AtlasRegConfig(subspace_dim=1)  # Epps-Pulley (1D, scalar projections)
AtlasRegConfig(subspace_dim=4)  # Henze-Zirkler (4D subspace, joint test)
```

Per-test cost at batch size N is O(N²) for HZ regardless of k, so higher k gives strictly richer signal per FLOP.

---

## Testing

```bash
pytest -v tests/
```

The suite verifies that:
- The cross-polytope has unit-norm rows and the correct count
- Random rotations are orthogonal with probability 1
- The Epps–Pulley statistic is zero (up to `1e-6`) on standard Gaussian samples
- The Henze–Zirkler statistic agrees with its closed form
- `AtlasReg` gradients flow end-to-end

---

## Citing

If you use AtlasWM, please cite both this repository and the works it builds on:

```bibtex
@software{atlaswm2026,
  title  = {AtlasWM: Stable JEPA World Models via Deterministic Distribution Matching},
  author = {{Darvy C.}},
  year   = {2026},
  url    = {https://github.com/darvyc/atlaswm},
}
```

See [`CITATION.cff`](CITATION.cff) for references to LeWM, SIGReg, DINO-WM, PLDM, and the Cramér-Wold / Epps-Pulley / Henze-Zirkler lineage.

---

## License

MIT. See [LICENSE](LICENSE).

# AtlasWM

**End-to-end joint-embedding predictive world models with structured characteristic-function regularization.**

[![CI](https://github.com/darvyc/AtlasWM/actions/workflows/ci.yml/badge.svg)](https://github.com/darvyc/AtlasWM/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version 2.0.0](https://img.shields.io/badge/version-2.0.0-4c1.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

AtlasWM learns action-conditioned latent dynamics directly from image trajectories. It combines:

- a frame-independent Vision Transformer encoder;
- a causal action-conditioned latent predictor;
- AtlasReg, a structured characteristic-function discrepancy;
- closed-loop latent-space planning with the Cross-Entropy Method;
- reproducible training, evaluation, checkpointing and benchmark orchestration.

The implementation keeps mathematical guarantees, finite estimators, software tests and task-level evidence separate. It does not convert a population theorem into an empirical performance claim.

## System design

```text
observation frames ──> frame-independent ViT ──> latent sequence
                                                │
action sequence ────────────────────────────────┼──> causal predictor ──> future latents
                                                │
                                                ├──> AtlasReg
                                                │
goal observation ──> shared ViT ────────────────┴──> CEM planner
```

The training objective is:

```text
prediction loss + lambda_reg * distribution regularizer
```

No encoder or predictor layer computes statistics across batch elements or across trajectory time. A frame's embedding therefore does not change because unrelated samples or later frames are present in the same minibatch.

## AtlasReg

For latent law `P` and target law `Q`, the population discrepancy integrates squared characteristic-function differences over all directions and frequencies:

```text
D(P,Q) = integral over u and t of
         w(t) * |CF_P(tu) - CF_Q(tu)|^2
```

With a positive integrable frequency weight, the continuum objective equals zero exactly when `P = Q`. Training necessarily uses a finite minibatch, finite directions and either finite quadrature or a Gaussian closed form.

AtlasReg provides:

- rotated cross-polytope, regular-simplex and iid Haar directions;
- exact antipodal reduction for symmetric-target squared CF losses;
- biased non-negative and unbiased finite-sample estimators;
- single-scale and two-scale Gaussian frequency weights;
- exact Gaussian BHEP/MMD forms;
- Gaussian and one-dimensional Student-t targets;
- multivariate Gaussian subspace matching;
- chunked frequency, projection and pairwise computation;
- cached Haar frames or fast signed-permutation randomization.

See [`docs/theory.md`](docs/theory.md) for the precise claims and limitations.

## Installation

```bash
git clone https://github.com/darvyc/AtlasWM.git
cd AtlasWM
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Student-t targets require SciPy, included by the development extra or installable with:

```bash
pip install -e ".[student-t]"
```

## Minimal training step

```python
import torch

from atlaswm import AtlasRegConfig, AtlasWM

model = AtlasWM(
    img_size=64,
    patch_size=8,
    embed_dim=192,
    action_dim=2,
    history_length=8,
    encoder_depth=4,
    encoder_heads=3,
    predictor_depth=4,
    predictor_heads=8,
    reg_config=AtlasRegConfig(
        design="cross_polytope",
        rotation_mode="haar",
        rotation_refresh_steps=16,
        target="gaussian",
        kernel="two_scale",
    ),
)

observations = torch.randn(4, 8, 3, 64, 64)
actions = torch.randn(4, 8, 2)
losses = model.training_step(observations, actions, lambda_reg=0.1)
losses["total"].backward()
```

## Reproducible training

```bash
atlaswm-train --config configs/default.yaml
```

A compact CPU integration run is available through:

```bash
atlaswm-train --config configs/smoke.yaml
```

Each run writes:

```text
resolved_config.yaml
system.json
metrics.jsonl
checkpoint_last.pt
checkpoint_best.pt
evaluation.json
```

Checkpoints contain model state, optimizer state, scheduler state, full configuration, dataset fingerprint, random-number-generator states, Git commit and system metadata. Training can resume exactly on the same software and hardware stack:

```bash
atlaswm-train --config configs/default.yaml --resume outputs/default/checkpoint_last.pt
```

## Trajectory datasets

### Memory-mapped arrays

The recommended large-dataset layout is:

```text
dataset/
  observations.npy   # (N,T,C,H,W)
  actions.npy        # (N,T,A)
  states.npy         # optional (N,T,S)
```

Configure it with:

```yaml
data:
  name: trajectory_npy
  path: data/pusht/train
  observation_file: observations.npy
  action_file: actions.npy
  state_file: states.npy
  sub_length: 4
  img_size: 224
```

The arrays are opened with NumPy memory mapping and converted only for the requested trajectory window.

Compact NPZ archives remain supported. Convert them to the scalable format with:

```bash
python scripts/convert_npz.py \
  --input trajectories.npz \
  --output data/converted \
  --state-key states
```

A JSON manifest can concatenate multiple memory-mapped shards without loading the complete image corpus into RAM.

Dataset splitting occurs by complete trajectory, not by overlapping windows, preventing train-test leakage.

## Planning

```python
from atlaswm import CEMPlanner

planner = CEMPlanner(
    model,
    horizon=5,
    n_samples=512,
    n_iters=8,
    n_elites=64,
    action_low=-1.0,
    action_high=1.0,
)

action_sequence = planner.plan(current_observation, goal_observation)
```

For multi-frame context, the real actions connecting those frames are mandatory:

```python
action_sequence = planner.plan(
    current_observation,
    goal_observation,
    context_observations=context_frames,       # (T0,C,H,W)
    context_actions=historical_actions,         # (T0-1,A)
)
```

AtlasWM rejects missing historical actions rather than silently inventing zero controls.

## Evaluation

Evaluate a saved checkpoint:

```bash
atlaswm-evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/default/checkpoint_best.pt
```

The evaluation stack reports:

- one-step and multi-horizon latent prediction error;
- action sensitivity;
- effective rank and singular-value concentration;
- coordinate variance and covariance error;
- pairwise cosine statistics;
- held-out physical-state linear probes when labels are available;
- closed-loop toy-control success and final goal distance when enabled.

## Matched benchmark suite

```bash
python scripts/run_benchmark_suite.py \
  --config configs/default.yaml \
  --methods prediction_only covariance full_gaussian_mmd iid_haar_ecf atlas \
  --seeds 11 23 37 53 71 \
  --output outputs/benchmark
```

The suite creates one resolved configuration and artifact directory per method and seed, then writes:

```text
runs.jsonl
aggregate.json
aggregate.csv
```

Reported methods share the encoder, predictor, data, optimizer, training steps and evaluation pipeline. See [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for evidence requirements.

## Verification

```bash
python -m compileall atlaswm scripts
pytest --cov=atlaswm --cov-report=term-missing
python scripts/reproduce_statistics.py \
  --seed 42 \
  --dim 8 \
  --samples 128 \
  --trials 128 \
  --output outputs/statistical_verification.json
python -m build
```

Continuous integration enforces linting, compilation, tests, branch coverage, package construction and statistical verification across supported Python versions.

## Scope

AtlasWM is a research system. It is not a safety controller and does not provide collision avoidance, calibrated uncertainty, actuator modelling or deployment guarantees. Physical systems require independent safety constraints and validation.

## Research lineage

AtlasWM builds on joint-embedding predictive architectures, empirical characteristic-function testing, Cramer-Wold identification, spherical designs, Gaussian-kernel discrepancies, Henze-Zirkler testing and Cross-Entropy Method planning. Full references appear in [`paper/atlaswm.md`](paper/atlaswm.md).

## License

MIT. See [`LICENSE`](LICENSE).

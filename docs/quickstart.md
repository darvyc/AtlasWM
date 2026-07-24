# AtlasWM Quickstart

## Install

```bash
git clone https://github.com/darvyc/AtlasWM.git
cd AtlasWM
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Verify the installation

```bash
python -m compileall atlaswm scripts
pytest
python scripts/reproduce_statistics.py \
  --seed 42 \
  --dim 4 \
  --samples 32 \
  --trials 16
```

## Run the end-to-end example

```bash
python examples/quickstart.py
```

The example generates visual trajectories from the included two-dimensional environment, trains a compact AtlasWM instance, and evaluates a linear probe on the latent representation. Treat the printed metrics as an integration demonstration rather than a benchmark result.

## Train with the default configuration

```bash
atlaswm-train --config configs/default.yaml
```

The equivalent source-tree command is:

```bash
python scripts/train.py --config configs/default.yaml
```

## Dataset interface

A dataset item must provide at least `(observations, actions)`:

```text
observations: (time, channels, height, width) float tensor
actions:      (time, action_dim) float tensor
```

Additional values, such as state labels or metadata, may follow. The training loop consumes the first two elements.

The observation range depends on the encoder preprocessing. The included synthetic dataset uses values in `[-1, 1]`.

## Command-line overrides

Configuration leaves can be replaced with dotted keys.

### Multivariate Gaussian subspace objective

```bash
atlaswm-train \
  --config configs/default.yaml \
  regularizer.subspace_dim=4 \
  regularizer.n_subspaces=4 \
  regularizer.whiten_kd=true \
  regularizer.hz_beta=null
```

### Unit-variance Student-t target

For `nu = 5`, the unit-variance scale is `sqrt(3/5)`.

```bash
atlaswm-train \
  --config configs/default.yaml \
  regularizer.target=student_t \
  regularizer.student_t_nu=5.0 \
  regularizer.student_t_scale=0.7745966692
```

### iid Haar projection baseline

```bash
atlaswm-train \
  --config configs/default.yaml \
  regularizer.design=haar \
  regularizer.n_haar_projections=1024 \
  regularizer.kernel=single \
  regularizer.rotate=false
```

### Exact Gaussian closed form

```bash
atlaswm-train \
  --config configs/default.yaml \
  regularizer.target=gaussian \
  regularizer.one_d_backend=closed_form \
  regularizer.kernel=single \
  regularizer.lambda_=1.0
```

## Benchmark regularizer cost

```bash
python scripts/bench.py \
  --dim 192 \
  --batch-size 512 \
  --n-iters 200 \
  --device auto
```

Benchmark values depend on hardware, PyTorch version, precision, batch size, and synchronization policy. Report the full environment and use matched settings for every method.

## Plan with a trained model

```python
from atlaswm import CEMPlanner

planner = CEMPlanner(
    model,
    horizon=5,
    n_samples=300,
    n_iters=30,
    n_elites=30,
    action_low=-1.0,
    action_high=1.0,
)

actions = planner.plan(current_observation, goal_observation)
first_action = actions[0]
```

For receding-horizon control, execute the first action, acquire a fresh observation, and plan again. Add environment-specific safety and feasibility constraints before physical deployment.

## Reproducible experiment records

Follow:

- [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)
- [`benchmark_protocol.md`](benchmark_protocol.md)
- [`MODEL_CARD.md`](../MODEL_CARD.md)

Every reported result should identify the Git commit, configuration, dataset fingerprint, hardware, seed, and raw per-seed metrics.

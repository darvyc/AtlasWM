# Quickstart

## Install

```bash
git clone https://github.com/darvyc/atlaswm.git
cd atlaswm
pip install -e ".[dev]"
```

## Run the smoke test

```bash
python examples/quickstart.py
```

This trains a tiny AtlasWM on a built-in 2D toy environment for two epochs, then probes the latent space to show that the encoder recovered the agent's 2D position — confirming the full pipeline (encoder + predictor + regularizer) is working end-to-end.

Expected output on a CPU in under a minute:

```
AtlasWM parameters: ~900,000
Generating toy dataset...
Training for 2 epochs on toy data...
[step     20] total=0.7234  pred=0.6102  reg=1.1320
[step     40] total=0.3891  ...
...
Probing latent space for agent position...
Linear probe R^2 for agent position: 0.94
```

## Train on your own data

Drop in any `torch.utils.data.Dataset` that yields `(obs, actions, ...)` per item, where:

- `obs`: `(T, C, H, W)` float tensor, roughly in `[-1, 1]`
- `actions`: `(T, action_dim)` float tensor

Then:

```bash
python scripts/train.py --config configs/default.yaml
```

## Override config from the CLI

Any dotted key can be overridden:

```bash
# Test the multivariate Henze-Zirkler variant
python scripts/train.py --config configs/default.yaml \
    regularizer.subspace_dim=4 \
    trainer.lambda_reg=1.0

# Switch to Student-t target with ν=5
python scripts/train.py --config configs/default.yaml \
    regularizer.target=student_t \
    regularizer.student_t_nu=5.0

# Ablate back to SIGReg-equivalent
python scripts/train.py --config configs/default.yaml \
    regularizer.design=haar \
    regularizer.n_haar_projections=1024 \
    regularizer.kernel=single \
    regularizer.rotate=false
```

## Benchmark the regularizer

Compare per-iteration cost of different regularizer configurations:

```bash
python scripts/bench.py --dim 192 --batch-size 512 --n-iters 200
```

Typical output on a modern GPU:

```
SIGReg-equivalent (Haar, 1D, single-scale)    3.21 ms/iter
AtlasReg (cross-polytope, two-scale)          1.20 ms/iter    <- ~2.7x faster
AtlasReg k=4 (Henze-Zirkler)                  1.85 ms/iter
```

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
actions = planner.plan(current_obs, goal_obs)
# actions: (horizon, action_dim) — execute actions[0] in the env, then replan.
```

## Test

```bash
pytest tests/ -v
```

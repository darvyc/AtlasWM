# AtlasWM Benchmark Protocol

## Required methods

| ID | Method | Definition |
|---|---|---|
| B0 | Prediction only | prediction loss with `lambda_reg=0` |
| B1 | Covariance | mean, variance and off-diagonal covariance penalty |
| B2 | Full Gaussian MMD | full-dimensional Gaussian BHEP/MMD target matching |
| B3 | iid Haar ECF | 1D ECF matching with 1024 iid Haar directions |
| A0 | AtlasReg | rotated cross-polytope with antipodal quotient and two-scale quadrature |

## Fixed-budget conditions

The following must be equal or explicitly normalized:

- dataset and trajectory split;
- transitions seen and optimizer steps;
- encoder and predictor architecture;
- batch size and resolution;
- augmentation and action preprocessing;
- optimizer, schedule and precision;
- hyperparameter-selection budget;
- planner horizon, candidates, elites and iterations;
- evaluation seeds and episodes.

Report both step-matched and wall-clock-matched results when regularizer costs differ.

## Metrics

### Prediction

- one-step latent MSE;
- rollout MSE at horizons 1, 5, 10 and task horizon;
- action sensitivity;
- state-space or probe-space error.

### Representation

- effective rank and rank fraction;
- minimum and mean coordinate variance;
- largest singular-value fraction;
- covariance error;
- pairwise cosine mean and standard deviation;
- held-out physical-state probe.

### Control

- success rate;
- final goal distance;
- steps to success;
- planning latency;
- model rollouts per environment action.

### Compute

- examples per second;
- optimizer-step time;
- regularizer forward and backward time;
- peak memory;
- total wall-clock time.

## Statistical reporting

Use at least five final seeds. Report per-seed values, mean, sample standard deviation and 95 percent confidence intervals. Use paired comparisons when seeds and task instances are shared. Episodes from one trained seed are not independent training replicates.

## Executable suite

```bash
python scripts/run_benchmark_suite.py \
  --config configs/default.yaml \
  --methods prediction_only covariance full_gaussian_mmd iid_haar_ecf atlas \
  --seeds 11 23 37 53 71 \
  --output outputs/benchmark
```

The suite writes raw records before aggregate tables and computes paired differences against the prediction-only control on shared seeds. No aggregate claim is valid without the corresponding raw outputs.

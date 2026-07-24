# AtlasWM Model Card

## Model summary

AtlasWM is an action-conditioned latent world model trained from pixel trajectories. The system contains:

- a vision transformer encoder mapping observations to latent vectors;
- a causal transformer predicting the next latent from latent and action histories;
- AtlasReg, a characteristic-function distribution regularizer;
- a Cross-Entropy Method planner operating in the learned latent space.

The default latent target is `N(0,I)`. AtlasReg also supports a scaled univariate Student-t target in its one-dimensional projection path.

## Release

| Field | Value |
|---|---|
| Name | AtlasWM |
| Version | 1.0.0 |
| Author | Darvy Cana |
| License | MIT |
| Framework | PyTorch |
| Python | 3.10 or later |
| Repository | `darvyc/AtlasWM` |

## Intended use

AtlasWM is intended for:

- research on joint-embedding predictive architectures;
- latent dynamics modelling;
- representation-collapse analysis;
- characteristic-function distribution matching;
- structured projection estimators;
- latent-space goal-conditioned planning;
- controlled comparisons of anti-collapse objectives.

The included synthetic environment supports integration tests, debugging, and demonstrations of the complete training and planning path.

## Out-of-scope use

AtlasWM is not designed for:

- safety-critical autonomous control without an independent safety layer;
- direct deployment in vehicles, robots, medical systems, industrial plants, or weapons;
- decisions affecting legal rights, employment, credit, healthcare, or access to public services;
- unrestricted real-world planning where model errors can cause physical harm;
- claims of causal understanding based only on latent prediction accuracy.

## Inputs

Training inputs:

```text
observations: (batch, time, channels, height, width)
actions:      (batch, time, action_dim)
```

Observations are floating-point image tensors. Actions are continuous vectors in the default examples.

Planning inputs:

```text
current observation
goal observation
optional observation context
```

## Outputs

- latent embeddings;
- predicted future latent embeddings;
- structured loss dictionary containing total, prediction, and regularizer losses;
- optimized action sequences from the CEM planner.

## Training objective

AtlasWM minimizes

```text
prediction loss + lambda_reg * AtlasReg loss.
```

The prediction term alone admits constant encoder and predictor solutions. AtlasReg supplies the non-degenerate distribution constraint.

## AtlasReg modes

| Mode | Interpretation |
|---|---|
| Raw Gaussian matching | constrains projected location, scale, and shape |
| Studentized 1D matching | tests projection shape after batch location-scale removal |
| Biased estimator | non-negative empirical discrepancy with finite-sample target floor |
| Unbiased estimator | unbiased population estimate that can be negative per batch |
| Gaussian closed form | exact all-frequency BHEP objective for the chosen Gaussian weight |
| k-D BHEP | multivariate discrepancy on sampled orthonormal subspaces |
| Whitened k-D mode | covariance-standardized normal-shape testing inside each subspace |

## Data

The repository includes a procedurally generated visual trajectory dataset:

- state: two-dimensional position;
- action: two-dimensional displacement command;
- observation: rendered RGB image;
- dynamics: bounded noisy motion with a restoring drift.

External benchmark datasets are not redistributed. Their provenance, licensing, checksums, and split definitions must accompany reported results.

## Evaluation

AtlasWM evaluations should separate:

- prediction accuracy;
- control success;
- representation geometry;
- estimator fidelity;
- compute and memory cost.

The required fixed-budget protocol is defined in [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

## Limitations

### Finite estimator

The population characteristic-function objective identifies the target distribution under its stated assumptions. A finite minibatch, finite direction set, and finite quadrature rule define an approximation and do not establish equality of arbitrary distributions.

### Optimization

The regularizer penalizes degenerate distributions, but the release does not provide a global optimization theorem excluding every collapsed stationary point.

### Planning

Latent-space CEM optimizes predicted goal distance. It does not enforce collision avoidance, actuator constraints beyond box clipping, uncertainty bounds, or safety constraints unless these are added by the user.

### Generalization

Prediction accuracy on the training distribution does not guarantee accuracy under intervention, distribution shift, unseen actions, or long autoregressive horizons.

### Target choice

An isotropic Gaussian or Student-t target is a modelling choice. It may not be optimal for every environment, observation process, architecture, or downstream planner.

### Student-t path

The Student-t target is implemented for one-dimensional projected characteristic functions. Multivariate Student-t BHEP matching is not part of this release.

## Risks

- latent model errors can compound during planning;
- CEM can exploit systematic predictor errors;
- batch-dependent normalization can conceal scale collapse when interpreted incorrectly;
- finite projection rules can miss deviations concentrated outside sampled directions;
- performance comparisons can be misleading without matched compute and seed-level reporting.

## Risk mitigations

- use receding-horizon planning with frequent re-observation;
- apply environment-specific action and safety constraints;
- monitor effective rank, covariance, variance, and held-out ECF discrepancy;
- evaluate on multiple seeds and out-of-distribution conditions;
- publish per-seed records and compute accounting;
- validate critical systems independently of AtlasWM.

## Reproducibility

Follow [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The software release includes theorem-linked tests and a deterministic statistical verification script.

## Citation

```bibtex
@software{cana2026atlaswm,
  author  = {Darvy Cana},
  title   = {AtlasWM: Structured Characteristic-Function Regularization for End-to-End JEPA World Models},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/darvyc/AtlasWM}
}
```

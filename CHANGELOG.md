# Changelog

## [2.0.0] - 2026-07-24

### Architecture

- frame-independent ViT encoder with no batch-coupled latent normalization;
- batch-independent predictor output projection;
- explicit causal alignment between each action and predicted transition;
- mandatory historical actions for multi-frame planning contexts;
- bounded, smoothed CEM with action and smoothness penalties;
- terminal Euclidean and cosine goal costs.

### Statistics

- memory-bounded projection, frequency and pairwise computation;
- cached Haar frames with configurable refresh intervals;
- fast signed-permutation orthogonal randomization;
- deterministic evaluation-frame reuse;
- exact reset control for projection randomization;
- matched covariance, full-Gaussian-MMD and prediction-only baselines.

### Data and training

- memory-mapped NPY trajectory datasets;
- sharded manifest datasets;
- trajectory-level train, validation and test splitting;
- NPZ conversion utility;
- mixed-precision training;
- cosine scheduling with warm-up;
- validation and best-checkpoint selection;
- atomic checkpoints with optimizer, scheduler, RNG, configuration, system and dataset provenance;
- exact resume support.

### Evaluation

- one-step and autoregressive multi-horizon prediction metrics;
- action sensitivity;
- independent collapse diagnostics;
- held-out physical-state probes;
- closed-loop toy-control evaluation;
- matched multi-seed benchmark orchestration and aggregation.

### Quality

- configuration validation for every shipped experiment;
- executable PushT memory-mapped configuration;
- regression tests for batch independence and action-history alignment;
- coverage-enforced CI;
- package build and statistical-verification gates.

## [1.0.0] - 2026-07-24

- initial structured characteristic-function regularizer and JEPA world-model release.

# Research hardening

This document records the controls added after an adversarial review of AtlasWM. It distinguishes implemented safeguards from evidence that still requires external datasets and compute.

## Target semantics

AtlasWM supports three explicit prediction-target modes:

- `shared`: both prediction and target branches update the online encoder;
- `stop_gradient`: the target latent is detached from autograd;
- `ema`: a non-trainable exponential-moving-average encoder supplies prediction targets.

The EMA target remains in evaluation mode and is synchronised at the beginning of each training call from the online encoder produced by the previous optimizer step. `ema_decay` must lie in `[0,1)`. The default research configuration uses `ema` with decay `0.996`.

## Regularization scope

AtlasReg no longer has to operate only on the pooled batch-time marginal. The model exposes:

- `marginal`: the original pooled latent marginal;
- `per_time`: a separate cross-trajectory discrepancy at every time index, averaged over time;
- `transition`: the distribution of target latent increments relative to current online latents;
- `marginal_transition`: an equal-weight average of marginal and transition discrepancies.

These modes do not prove that the representation is sufficient, Markovian or task optimal. They make the modelling assumption explicit and experimentally testable.

## Benchmark selection

The benchmark runner now separates hyperparameter selection from final evaluation:

1. each method is trained over an equal regularization-weight grid on selection seeds;
2. the weight is selected using a declared metric and direction;
3. the chosen weight is frozen;
4. final runs use disjoint evaluation seeds.

The suite records selection candidates, raw final runs, wall-clock duration, aggregate tables and paired comparisons. It adds a random-orthogonal ECF baseline so that the cross-polytope construction is not credited for gains attributable merely to orthogonal projection sampling.

## Statistical reporting

For small numbers of independent training seeds, 95 percent intervals use Student-t critical values rather than the normal approximation. Paired reports include:

- mean and standard deviation of paired differences;
- Student-t 95 percent half-width;
- paired Hedges effect size when defined;
- fraction of positive paired differences.

Metric direction must still be interpreted by the caller. A positive difference is not automatically favourable.

## Local verification performed before publication

The hardening patch was exercised on CPU with PyTorch 2.10.0. The focused suite verifies:

- gradient isolation for stop-gradient targets;
- exact EMA interpolation and frozen target parameters;
- target encoder evaluation mode;
- per-time and transition regularizer input semantics;
- configuration construction and invalid-mode rejection;
- Student-t confidence intervals;
- nested regularization-weight selection.

An eight-step end-to-end EMA optimisation smoke run reduced fixed-batch loss from `2.3038039e-05` to `2.1768085e-06`.

## Remaining empirical burden

This patch improves causal training semantics and experimental validity. It does not manufacture evidence. Publication-level claims still require raw multi-seed results on external control datasets, downloadable checkpoints, wall-clock-matched ablations and independent reproduction. The benchmark runner is designed to produce that evidence without test-seed hyperparameter leakage.

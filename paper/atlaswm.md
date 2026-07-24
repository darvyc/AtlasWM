# AtlasWM: Structured Characteristic-Function Regularization for End-to-End JEPA World Models

**Darvy Cana**

## Abstract

AtlasWM is an action-conditioned joint-embedding predictive world model trained directly from image trajectories. A frame-independent Vision Transformer maps observations to latent vectors, a causal action-conditioned transformer predicts future latents, and AtlasReg constrains the marginal latent distribution using weighted characteristic-function discrepancies. The population objective identifies equality in distribution under a positive integrable frequency weight. Finite training uses explicit minibatch, directional and frequency approximations, including structured spherical projection rules, biased or unbiased estimators, Gaussian BHEP closed forms and multivariate subspace objectives. The system includes memory-bounded regularizer computation, memory-mapped trajectory data, trajectory-safe splits, resumable mixed-precision training, matched baselines, multi-horizon evaluation and latent CEM planning with exact action-history alignment. Mathematical identities, software correctness and task-level evidence are treated as distinct evidence classes.

## 1. Model

For observation `o_t`, action `a_t`, encoder `f` and predictor `g`:

```text
z_t = f(o_t)
z_hat_(t+1) = g(z_<=t, a_<=t)
```

The objective is:

```text
L = mean ||z_hat_(t+1) - z_(t+1)||^2 + lambda * AtlasReg(Z)
```

Both encoder branches remain trainable. AtlasReg supplies the non-degenerate marginal distribution constraint.

## 2. Causal representation boundary

Images are encoded independently. Flattening batch and trajectory dimensions is used only for throughput. No BatchNorm or cross-sample normalization occurs in either latent head. The embedding of one frame therefore cannot depend on the other frames or samples present in the minibatch.

The action at index `t` conditions the token used to predict `z_(t+1)`. Autoregressive rollout requires the real `T0-1` actions connecting `T0` context states. This preserves the same transition semantics in training and planning.

## 3. AtlasReg

Let `P` and `Q` have characteristic functions `phi_P` and `phi_Q`. AtlasReg begins from:

```text
D_w(P,Q) = integral over unit u and real t of
           w(t) |phi_P(tu)-phi_Q(tu)|^2
```

For positive integrable `w`, the population discrepancy is identifying. The implementation replaces the population, sphere and usually the frequency integral with finite objects. Finite zero loss is not equality proof.

### 3.1 Structured directions

The cross-polytope is a spherical 3-design. Symmetric targets make antipodal lines redundant. Haar rotation provides unbiased directional integration in rotation expectation; configurable frame reuse amortizes dense QR. Signed-permutation randomization offers a faster orthogonal but non-Haar alternative.

### 3.2 Finite estimators

The default V-statistic is non-negative and has a known target floor. The U-statistic is unbiased for the population discrepancy but can be negative. It is not used after batch-dependent standardization or whitening.

### 3.3 Gaussian BHEP

Gaussian frequency weighting yields an exact Gaussian-kernel MMD expression against `N(0,I)`. The implementation supports exact pairwise chunking to bound temporary memory.

## 4. Data and optimization

Large trajectories use separate memory-mapped NPY arrays. JSON manifests concatenate shards lazily. Data splits operate on complete trajectories so overlapping windows cannot cross split boundaries.

Training supports mixed precision, gradient clipping, warm-up and cosine scheduling, validation, best-model selection, atomic checkpoints and exact resume on a compatible stack. Every checkpoint includes configuration, dataset fingerprint, RNG states, Git commit and system metadata.

## 5. Evaluation

The evaluation contract contains:

- one-step and autoregressive multi-horizon latent error;
- action sensitivity;
- effective rank, variance, covariance and cosine diagnostics;
- held-out physical-state probes;
- closed-loop goal-conditioned control;
- throughput and regularizer microbenchmarks.

Matched multi-seed comparisons include prediction-only, covariance, full Gaussian MMD, iid Haar ECF and AtlasReg configurations.

## 6. Planning

CEM optimizes bounded action sequences in latent space. It retains the best sampled sequence, smooths elite distribution updates and optionally penalizes action magnitude and temporal roughness. Planning remains model-based optimization without uncertainty or physical safety guarantees.

## 7. Limitations

1. A finite projection estimator can miss unsampled distributional deviations.
2. Distribution matching does not guarantee task-relevant information.
3. Isotropic targets are modelling choices.
4. Long-horizon autoregression can accumulate error.
5. CEM can exploit predictor error.
6. Physical deployment requires external safety constraints.
7. Empirical superiority requires the matched benchmark protocol and raw outputs.

## References

1. H. Cramer and H. Wold. Some Theorems on Distribution Functions. 1936.
2. P. Delsarte, J. M. Goethals and J. J. Seidel. Spherical Codes and Designs. 1977.
3. T. W. Epps and L. B. Pulley. A Test for Normality Based on the Empirical Characteristic Function. 1983.
4. L. Baringhaus and N. Henze. A Consistent Test for Multivariate Normality Based on the Empirical Characteristic Function. 1988.
5. N. Henze and B. Zirkler. A Class of Invariant Consistent Tests for Multivariate Normality. 1990.
6. F. Mezzadri. How to Generate Random Matrices from the Classical Compact Groups. 2007.
7. R. Balestriero and Y. LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 2025.
8. L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun and R. Balestriero. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. 2026.
9. R. Y. Rubinstein and D. P. Kroese. The Cross-Entropy Method. 2004.

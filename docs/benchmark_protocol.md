# AtlasWM Benchmark Protocol

This protocol defines controlled comparisons for AtlasReg and AtlasWM. Its purpose is to separate representation quality, predictive accuracy, planning performance, and computational cost.

## 1. Research questions

The benchmark evaluates five questions:

1. Does structured projection sampling improve the efficiency or stability of latent distribution matching relative to iid Haar projections?
2. Does AtlasReg reduce representation collapse while preserving predictive information?
3. Which objective configuration produces the strongest downstream planning performance at a fixed compute budget?
4. How do projection count, frequency weighting, estimator choice, and subspace dimension affect optimization?
5. What computational cost is added by each regularizer configuration?

## 2. Environments

Use environments spanning low-dimensional control, image-based manipulation, and longer-horizon dynamics.

Recommended suite:

| Group | Environments | Primary purpose |
|---|---|---|
| Synthetic | AtlasWM toy trajectories | integration, debugging, controlled latent geometry |
| Planar manipulation | PushT | image-based goal-conditioned planning |
| Continuous control | DMControl visual tasks | action-conditioned dynamics and long-horizon prediction |
| Offline goal-conditioned control | OGBench visual or state-rendered tasks | planning under dataset coverage constraints |

Every benchmark report must state the exact environment version, observation pipeline, action scaling, horizon, dataset source, and evaluation episode seeds.

## 3. Compared methods

All methods use the same encoder, predictor, optimizer, data, and planner unless the method definition requires otherwise.

### Required baselines

| ID | Method | Definition |
|---|---|---|
| B0 | Prediction only | JEPA prediction loss without a latent distribution regularizer |
| B1 | iid Haar ECF | 1D Gaussian ECF loss with iid Haar directions |
| B2 | Covariance penalty | mean, variance, and off-diagonal covariance regularization |
| B3 | Gaussian MMD | full-latent Gaussian-kernel MMD against `N(0,I)` |
| A0 | AtlasReg default | rotated cross-polytope, antipodal quotient, two-scale quadrature |
| A1 | AtlasReg closed form | rotated cross-polytope, exact Gaussian BHEP backend |
| A2 | AtlasReg k-D | random k-frames with multivariate BHEP |

### Optional baselines

- LeWorldModel or SIGReg reference implementation;
- VICReg-style variance and covariance objective;
- sliced Wasserstein matching;
- random Fourier feature MMD;
- pretrained visual world-model baseline where compute accounting is explicit.

## 4. Fixed-budget rules

A comparison is valid only when the following quantities are equal or explicitly normalized:

- training transitions seen;
- optimizer steps;
- encoder and predictor parameter count;
- batch size;
- image resolution;
- augmentation policy;
- optimizer and schedule;
- mixed-precision policy;
- planner samples, elites, horizon, and iterations;
- evaluation episodes;
- hyperparameter-selection budget.

Report both step-matched and wall-clock-matched results when regularizer costs differ materially.

## 5. Hyperparameter selection

Use a separate validation seed set. Do not tune on test episodes.

Recommended search spaces:

```text
lambda_reg:           {0.01, 0.03, 0.1, 0.3, 1.0}
projection count:     {d/2, d, 2d, 512, 1024}
frequency knots:      {9, 17, 33, 65}
lambda_1:             {0.25, 0.5, 1.0}
lambda_2:             {1.0, 2.0, 4.0}
alpha:                {0.25, 0.5, 0.75}
subspace_dim:         {2, 4, 8, 16}
n_subspaces:          {1, 2, 4, 8}
student_t_nu:         {3, 5, 10, 30}
```

Use the same number of validation trials for each method. Report the complete search space and the selection rule.

## 6. Primary metrics

### Control and planning

- success rate;
- final goal distance;
- area under the success-versus-planning-budget curve;
- planning latency per environment action;
- number of model rollouts per action.

### Predictive modelling

- one-step latent MSE;
- multi-step latent MSE at horizons 1, 5, 10, and the task horizon;
- decoded or probe-space state error when ground-truth state is available;
- action sensitivity of predicted latents.

### Representation geometry

- effective rank of the centered latent matrix;
- minimum and median singular values;
- covariance error `||Cov(Z)-I||_F` for raw Gaussian target matching;
- mean norm and variance profile;
- held-out ECF discrepancy on iid Haar directions;
- nearest-neighbour state consistency;
- linear-probe error for physical state variables.

### Compute

- training examples per second;
- optimizer-step time;
- regularizer forward and backward time;
- peak accelerator memory;
- total training wall-clock time;
- planning latency and memory.

## 7. Collapse diagnostics

Report the following during training:

```text
latent mean norm
average coordinate variance
minimum coordinate variance
effective rank
largest singular-value fraction
pairwise cosine-similarity mean and variance
prediction loss
regularizer loss
```

A run is classified as numerically collapsed when at least two independent geometry diagnostics cross predeclared thresholds for a sustained interval. Thresholds must be fixed before final evaluation.

Recommended diagnostic thresholds for investigation, not universal definitions:

```text
effective rank / d < 0.05
mean coordinate variance < 1e-4
largest singular-value fraction > 0.95
pairwise cosine similarity > 0.99
```

## 8. AtlasReg ablations

The minimum ablation matrix includes:

1. cross-polytope versus iid Haar;
2. fixed versus fresh random rotation;
3. full `2d` vertices versus `d` antipodal lines;
4. single-scale versus two-scale frequency weight;
5. biased versus unbiased estimator;
6. quadrature versus exact Gaussian closed form;
7. raw target matching versus studentized shape testing;
8. 1D projections versus k-dimensional subspaces;
9. fixed BHEP bandwidth versus classical Henze-Zirkler bandwidth;
10. Gaussian versus unit-variance Student-t target.

Each ablation changes one factor while holding the remaining configuration fixed.

## 9. Statistical analysis

Use at least five final-reporting seeds. Report all per-seed values.

For each metric:

- mean and sample standard deviation;
- 95 percent confidence interval;
- paired difference against the designated baseline;
- paired bootstrap confidence interval;
- standardized effect size where meaningful.

For success rates, bootstrap over both seeds and episodes or use a hierarchical binomial model. Avoid treating multiple episodes from one trained seed as independent training replicates.

Correct for multiple comparisons when drawing conclusions from a large ablation matrix. A false-discovery-rate procedure is suitable for exploratory ablations.

## 10. Reporting table

A primary result table should contain:

| Method | Success | Goal distance | H5 prediction error | Effective rank | Train ms/step | Plan ms/action | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|

Every aggregate cell must be traceable to raw seed-level records.

## 11. Reproducible outputs

Each run writes:

```text
config.yaml
system.json
metrics.jsonl
checkpoint.pt
latent_diagnostics.npz
evaluation_episodes.jsonl
```

The aggregate report writes:

```text
aggregate.csv
paired_comparisons.csv
confidence_intervals.csv
figures/
```

## 12. Claim policy

The evidence supports a claim only when the corresponding protocol is satisfied.

| Claim | Required evidence |
|---|---|
| Lower estimator cost | matched hardware microbenchmark with warm-up and synchronization |
| Lower estimator variance | repeated projection estimates on fixed latent batches |
| Better collapse resistance | multi-seed geometry diagnostics and predeclared collapse rule |
| Better prediction | held-out multi-horizon errors under matched training budget |
| Better control | matched planner and evaluation episodes with seed-level confidence intervals |
| Better sample efficiency | learning curves against transitions seen, not epochs alone |

Mathematical identities, software correctness, and empirical performance are reported as separate evidence classes.

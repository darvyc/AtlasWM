# Mathematical Foundations of AtlasReg

## Population objective

Let `P` be the latent distribution on `R^d`, `Q` a target distribution, `u` a unit direction and `t` a scalar frequency. Define:

```text
D_w(P,Q) = integral_S integral_R
           w(t) |phi_P(tu) - phi_Q(tu)|^2 dt dsigma(u)
```

If `w` is integrable and positive almost everywhere, then `D_w(P,Q) >= 0` and `D_w(P,Q)=0` exactly when `P=Q`. The argument uses non-negativity, continuity of characteristic functions and uniqueness of characteristic functions.

## Finite implementation

AtlasReg replaces:

1. `P` with a minibatch empirical measure;
2. the sphere with finite directions or subspaces;
3. the frequency integral with quadrature, except for the Gaussian closed form.

A zero finite loss does not prove equality of arbitrary distributions.

## Estimators

The biased empirical-measure estimator is non-negative but has a positive expected target floor. The unbiased U-statistic removes this expectation but can be negative on one batch. Batch-dependent studentization or whitening destroys the iid structure required by the unbiased formula and is therefore rejected.

## Gaussian closed form

For samples `y_i` in `R^k`, Gaussian bandwidth `beta` and target `N(0,I)`, the BHEP discrepancy is:

```text
mean_ij exp(-beta^2 ||y_i-y_j||^2 / 2)
- 2(1+beta^2)^(-k/2) mean_i exp(-beta^2 ||y_i||^2 / (2(1+beta^2)))
+ (1+2beta^2)^(-k/2)
```

This is Gaussian-kernel MMD squared against the Gaussian target. Pairwise chunking changes only floating-point summation order.

## Projection rules

The full cross-polytope `{+e_i,-e_i}` is a spherical 3-design. It integrates spherical polynomials through degree three exactly. The characteristic-function integrand is not a cubic polynomial, so this does not make the full loss exact.

For symmetric targets, the squared CF discrepancy is identical for `u` and `-u`; one representative per antipodal line gives exactly the same averaged loss.

A Haar-rotated orthonormal basis is unbiased for the spherical average across rotations. Reusing one sampled Haar frame for several optimization steps amortizes QR cost but introduces temporal dependence between estimator draws. The optional signed-permutation mode is exactly orthogonal and inexpensive but is not Haar-unbiased for arbitrary directional integrands.

## Target matching and shape testing

Raw projection matching constrains projected location, scale and shape. Projection-wise studentization removes location and scale and tests only standardized shape. Subspace whitening similarly removes mean and covariance within each sampled subspace.

## Collapse boundary

Prediction loss alone admits a constant encoder and predictor. A non-degenerate distribution target penalizes that solution, but there is no global optimization theorem excluding every collapsed stationary point. Aggregate distribution matching also does not prove that the representation retains task-relevant information; probes, prediction and control evaluation remain necessary.

## Complexity

For `N` samples, `M` directions, `T` frequency nodes, `S` subspaces and subspace dimension `k`:

```text
1D quadrature:        O(TNM)
1D Gaussian closed:   O(MN^2)
k-D BHEP:             O(SNkd + SN^2k)
Haar QR refresh:      O(d^3)
```

Chunking bounds temporary memory without approximating the stated estimator.

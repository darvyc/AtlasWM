# Mathematical foundations of AtlasReg

This document states exactly what AtlasReg optimizes, what the finite implementation approximates, and which claims do and do not follow from Cramer-Wold, spherical designs, Epps-Pulley, and Henze-Zirkler.

## 1. Notation

Let `P` be the distribution of a latent vector `Z in R^d` and let `Q` be the target distribution. Their characteristic functions are

```text
phi_P(xi) = E[exp(i xi^T Z)],
phi_Q(xi) = E[exp(i xi^T Y)],  Y ~ Q.
```

For a direction `u in S^(d-1)`, the one-dimensional projected characteristic function is

```text
phi_{P,u}(t) = phi_P(tu) = E[exp(i t u^T Z)].
```

Throughout, `sigma` denotes normalized Haar probability measure on the unit sphere.

## 2. The population sliced characteristic-function metric

The ideal objective is

```text
D_w(P,Q)
  = integral_{S^(d-1)} integral_R
      w(t) |phi_P(tu) - phi_Q(tu)|^2 dt dsigma(u),
```

where `w(t) > 0` for almost every `t` and `w` is integrable.

### Proposition 1: identifiability

Under those conditions,

```text
D_w(P,Q) >= 0,
D_w(P,Q) = 0  if and only if  P = Q.
```

Reason:

1. A zero integral implies equality of the projected characteristic functions for almost every `(u,t)`.
2. Characteristic functions are continuous, so equality extends from an almost-everywhere dense set to all `tu in R^d`.
3. Equality of characteristic functions on `R^d` determines equality in distribution.

This is the precise role of the Cramer-Wold theorem. It applies to the full continuum of directions and frequencies. It does not say that any fixed finite set of projections identifies an arbitrary distribution.

## 3. What the implementation actually computes

AtlasReg replaces both integrals by finite rules:

```text
D_hat(P_N,Q)
  = (1/M) sum_{m=1}^M sum_{r=1}^T
      a_r |phi_N(t_r u_m) - phi_Q(t_r u_m)|^2,
```

where

```text
phi_N(xi) = (1/N) sum_{n=1}^N exp(i xi^T z_n).
```

There are three distinct approximation layers:

1. `P` is replaced by the empirical batch distribution `P_N`.
2. The sphere is replaced by `M` directions.
3. The frequency integral is replaced by `T` quadrature nodes over a finite interval.

Consequently, a zero finite loss does not prove `P = Q`. Distinct distributions can agree at finitely many directions and frequencies. The finite loss is a training discrepancy, not a formal equality test.

For a Gaussian target and Gaussian frequency weight, AtlasWM also exposes an exact all-frequency closed form. It removes frequency quadrature, but a finite direction set and finite batch remain.

## 4. Biased and unbiased finite-sample estimators

At a fixed frequency and direction, write

```text
phi_N = (1/N) sum_n exp(i t h_n),
h_n = u^T z_n.
```

The non-negative empirical-measure estimator is

```text
V_N(t) = |phi_N(t) - phi_Q(t)|^2.
```

It is biased upward at the target. If `h_n` are iid from `Q`, then

```text
E[|phi_N(t)|^2]
  = (1 - 1/N) |phi_Q(t)|^2 + 1/N,
```

and therefore

```text
E[V_N(t)] = (1/N) (1 - |phi_Q(t)|^2).
```

This explains why a finite Gaussian batch does not and should not produce exactly zero loss.

An unbiased estimator replaces the empirical squared modulus by the U-statistic

```text
U_phi(t)
  = 1/[N(N-1)] sum_{i != j} exp(i t(h_i-h_j))
  = [N |phi_N(t)|^2 - 1] / (N-1).
```

Thus

```text
U_N(t)
  = U_phi(t)
    - 2 Re(phi_N(t) conjugate(phi_Q(t)))
    + |phi_Q(t)|^2
```

has expectation equal to the population squared discrepancy. Unlike the biased estimator, `U_N` can be negative on an individual batch. AtlasReg therefore defaults to the biased, non-negative estimator and offers the unbiased estimator for diagnostics and research.

The unbiased iid formula is not valid after batch-dependent standardization or whitening because the transformed observations are no longer independent.

## 5. Gaussian weight: exact BHEP closed form

For a `k`-dimensional sample `y_1,...,y_N` and target `N(0,I_k)`, choose the normalized Gaussian kernel

```text
K_beta(x,y) = exp[-beta^2 ||x-y||^2 / 2].
```

The Gaussian-weighted characteristic-function discrepancy has the exact form

```text
BHEP_beta(Y)
  = (1/N^2) sum_{i,j} exp[-beta^2 ||y_i-y_j||^2 / 2]

    - 2 (1+beta^2)^(-k/2) (1/N) sum_i
        exp[-beta^2 ||y_i||^2 / (2(1+beta^2))]

    + (1+2 beta^2)^(-k/2).
```

This follows by expanding the squared characteristic-function difference and applying the Fourier transform of a Gaussian to each term.

### Relation to MMD

The same expression is

```text
MMD^2_K(P_N, N(0,I_k))
```

for the Gaussian kernel `K_beta`. The biased form is therefore non-negative because it is a squared norm in a reproducing-kernel Hilbert space.

### Finite-sample Gaussian floor

If `Y_i ~ N(0,I_k)` iid, then

```text
E[BHEP_beta(Y)]
  = (1/N) [1 - (1+2 beta^2)^(-k/2)].
```

AtlasWM exposes this quantity as `gaussian_bhep_null_floor`.

### Henze-Zirkler bandwidth

The classical Henze-Zirkler normality test uses the BHEP family with

```text
beta_N
  = 2^(-1/2) [((2k+1)N)/4]^(1/(k+4)).
```

AtlasReg treats `beta` principally as a trainable-objective bandwidth. A fixed `beta` is therefore a BHEP discrepancy. Setting `hz_beta=None` selects the classical sample-size-dependent Henze-Zirkler rule.

The classical test statistic is usually multiplied by `N`; AtlasReg returns the unscaled discrepancy so its magnitude does not grow linearly with batch size.

## 6. Spherical designs: exact claim and limitation

A finite set `D = {u_1,...,u_M}` is a spherical `t`-design if

```text
(1/M) sum_m p(u_m)
  = integral_{S^(d-1)} p(u) dsigma(u)
```

for every polynomial `p` of total degree at most `t`.

### Cross-polytope

The vertices `{+e_i,-e_i}_{i=1}^d` form a spherical 3-design:

```text
average_D[u] = 0,
average_D[u u^T] = I/d,
average_D[u_i u_j u_l] = 0.
```

This gives exact spherical integration for linear, quadratic, and cubic polynomials in the direction `u`, after any orthogonal rotation.

### What it does not imply

The Epps-Pulley integrand contains exponentials and is not a degree-3 polynomial in `u`. A 3-design therefore does not integrate the full characteristic-function loss exactly.

The first missed even order is four. For a fixed vector `x`,

```text
E_{u~sphere}[(u^T x)^4]
  = 3 ||x||^4 / [d(d+2)],
```

whereas the unrotated cross-polytope gives

```text
average_{u in cross-polytope}[(u^T x)^4]
  = (1/d) sum_i x_i^4.
```

These are not equal in general. The cross-polytope provides exact low-degree cubature, not exact distribution matching.

More generally, for `m >= 0`,

```text
E[(u^T x)^(2m)]
  = [(2m-1)!! / (d(d+2)...(d+2m-2))] ||x||^(2m).
```

The coefficient is exposed by `spherical_projection_even_moment_coefficient`.

## 7. Why random rotation is useful

Let `R` be Haar distributed on `O(d)` and let `f` be integrable on the sphere. Then

```text
E_R[(1/d) sum_{i=1}^d f(R e_i)]
  = integral f(u) dsigma(u).
```

Each column `R e_i` is marginally uniform on the sphere. Linearity of expectation proves the result; independence of the columns is not required.

Therefore a randomly rotated orthonormal basis is an unbiased estimator of the spherical average for any integrable `f`. Its directions are dependent, so its variance is not the iid Monte Carlo variance `Var(f)/d`. It can be lower or higher depending on the angular structure of `f`.

The correct description is:

```text
structured and conditionally deterministic within a step,
stochastic across random rotations,
exact for degree <= 3 spherical polynomials,
unbiased in rotation expectation for general integrands.
```

It is not a fully deterministic estimator when a fresh random rotation is sampled each step.

## 8. Antipodal redundancy

For a symmetric target, the discrepancy for `u` and `-u` is identical:

```text
phi_{P,-u}(t) = conjugate(phi_{P,u}(t)),
phi_Q(t) is real and even,
|phi_{P,-u}(t)-phi_Q(t)|^2
  = |phi_{P,u}(t)-phi_Q(t)|^2.
```

The `2d` cross-polytope vertices therefore represent only `d` distinct projection lines for this loss. AtlasReg keeps one representative from each antipodal pair by default, producing exactly the same averaged loss with half the projection work.

The full `2d` set remains the object that is a spherical 3-design. The `d`-direction computation is an exact quotient of an even loss on real projective space.

## 9. Target matching versus shape testing

There are two different objectives.

### Target matching

Use raw projections:

```text
h_n = u^T z_n.
```

Matching them to `N(0,1)` constrains location, scale, and shape. Across all directions this is genuine matching to `N(0,I_d)`.

### Shape testing

Studentize each projected batch:

```text
h_tilde = (h - batch_mean(h)) / batch_std(h).
```

This removes location and scale information. The resulting objective tests whether each standardized projection has the target shape. It does not enforce latent mean zero, covariance identity, or any fixed embedding scale.

AtlasReg defaults to target matching. `standardize_1d=True` and `whiten_kd=True` are explicit affine-invariant shape-test modes.

## 10. Collapse and the JEPA objective

The predictive term is

```text
L_pred(theta,psi)
  = E ||g_psi(f_theta(o_<=t),a_<=t) - f_theta(o_(t+1))||^2.
```

A constant encoder `f_theta(o)=c` together with a predictor that outputs `c` makes `L_pred=0`. Thus the prediction loss alone admits complete representation collapse as a global optimum.

The distribution regularizer is not auxiliary decoration. It is the term that makes a constant empirical distribution disagree with a non-degenerate target.

A further optimization caveat remains: for symmetric discrepancies, exact point collapse at the target mean can be a stationary configuration even when its loss is positive. Random initialization, BatchNorm, minibatch noise, and the surrounding prediction dynamics normally prevent reaching that exact measure-zero state, but there is no theorem here guaranteeing global avoidance of collapse.

## 11. k-dimensional subspaces

For `k > 1`, AtlasReg samples an orthonormal frame `U in R^(k x d)` and forms

```text
Y = Z U^T.
```

It then applies the exact BHEP Gaussian discrepancy in `R^k`. Averaging over `n_subspaces > 1` reduces the variance of using one random subspace.

A finite collection of k-dimensional subspaces is not automatically stronger than a larger finite collection of one-dimensional projections. The population statement is only that matching every k-dimensional projection implies matching every one-dimensional projection and hence the joint law.

The k-dimensional path does not use the cross-polytope design. Projection design and subspace dimension are therefore not completely orthogonal implementation knobs.

## 12. Student-t target

For a scale-one univariate Student-t variable with `nu` degrees of freedom,

```text
phi_nu(t)
  = 2^(1-nu/2) / Gamma(nu/2)
    * (sqrt(nu)|t|)^(nu/2)
    * K_(nu/2)(sqrt(nu)|t|).
```

For scale `s`, replace `t` by `s t`.

When `nu > 2`, scale one has variance

```text
Var(T_nu) = nu/(nu-2).
```

A unit-variance Student-t target therefore uses

```text
s = sqrt((nu-2)/nu).
```

This scale is available through `student_t_unit_variance_scale`.

The direct Bessel formula can overflow numerically for large `nu` even though a characteristic function is bounded by one. AtlasWM evaluates the logarithmic scaled-Bessel form and falls back to a stable Fourier integral if necessary.

Heavy tails do not create low intrinsic dimension. An isotropic Student-t target remains full-dimensional. Any advantage on low-dimensional environments is an empirical modelling hypothesis, not a consequence of Cramer-Wold.

## 13. Complexity

Let `N` be batch samples, `M` directions, `T` frequency knots, and `S` sampled subspaces.

```text
1D quadrature:       O(T N M) time, O(T N M) temporary memory in the direct form.
1D exact closed form: O(M N^2) time.
k-D BHEP:            O(S N^2 k + S N k d) time.
```

For a symmetric target, antipodal deduplication changes the cross-polytope from `M=2d` evaluated vertices to `M=d` distinct lines without changing the loss.

## 14. Guarantee table

| Statement | Status |
|---|---|
| Full spherical and frequency CF objective identifies the distribution | Proven under positive integrable weight |
| Finite directions and finite frequency knots identify every distribution | False |
| Cross-polytope integrates spherical polynomials through degree 3 exactly | True |
| Cross-polytope integrates the full EP loss exactly | False |
| Haar-rotated basis is unbiased for the spherical average | True |
| Fresh random rotation is deterministic | False |
| Antipodal pairs add information for symmetric-target squared CF loss | False |
| Finite Gaussian sample has exactly zero biased loss | False |
| Prediction loss alone excludes constant collapse | False |
| Raw-projection matching constrains mean, covariance, and shape | True in the population full-projection objective |
| Studentized projection testing fixes target scale | False |

## 15. Recommended research configurations

True Gaussian target matching:

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=False,
    estimator="biased",
    design="cross_polytope",
    deduplicate_antipodes=True,
    rotate=True,
)
```

Affine-invariant Gaussian shape testing:

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=True,
    estimator="biased",
)
```

Unbiased diagnostic estimate:

```python
AtlasRegConfig(
    target="gaussian",
    standardize_1d=False,
    estimator="unbiased",
)
```

Exact all-frequency one-dimensional Gaussian discrepancy for validation:

```python
AtlasRegConfig(
    target="gaussian",
    one_d_backend="closed_form",
    kernel="single",
    lambda_=1.0,
)
```

Canonical HZ bandwidth on random k-dimensional subspaces:

```python
AtlasRegConfig(
    subspace_dim=4,
    n_subspaces=4,
    target="gaussian",
    whiten_kd=True,
    hz_beta=None,
)
```

## References

- Cramer, H. and Wold, H. (1936), *Some Theorems on Distribution Functions*.
- Delsarte, P., Goethals, J. M. and Seidel, J. J. (1977), *Spherical Codes and Designs*.
- Murota, K. and Takeuchi, K. (1981), *The studentized empirical characteristic function and its application to test for the shape of distribution*.
- Epps, T. W. and Pulley, L. B. (1983), *A test for normality based on the empirical characteristic function*.
- Henze, N. and Zirkler, B. (1990), *A class of invariant consistent tests for multivariate normality*.
- Mezzadri, F. (2007), *How to generate random matrices from the classical compact groups*.

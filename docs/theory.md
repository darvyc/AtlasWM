# Theory of AtlasReg

This document derives the four generalizations AtlasReg makes over SIGReg
and explains why each is principled.

## 1. Cramer-Wold as the backbone

**Theorem (Cramer–Wold, 1936).** *Two probability measures `P, Q` on `R^d`
are equal if and only if every one-dimensional marginal matches:*

```
P = Q  <==>  (u^T X) ~_{d} (u^T Y)  for all u in S^(d-1).
```

This is the lever every CF-based regularizer pulls on. Instead of testing
whether `Z ~ N(0, I)` directly in high dimensions — which is hard — we
test whether `u^T Z` is univariate standard normal, for a large enough
collection of directions `u`. By Cramer-Wold, matching all 1D projections
implies matching the joint.

Crucially, the theorem is **target-agnostic**: it works for any target
distribution, not just Gaussian. This underlies the optional Student-t
target.

## 2. Spherical t-designs: from Monte Carlo to exact moments

A **spherical t-design** on `S^(d-1)` is a finite set `{u_1, ..., u_M}`
such that for every polynomial `p` with `deg(p) <= t`,

```
(1/M) sum_i p(u_i)  =  integral_{S^(d-1)} p(u) dsigma(u)
```

where `dsigma` is Haar measure on the sphere.

**Why this matters for SIGReg-style regularizers.** The Epps-Pulley
statistic, when averaged over a direction `u`, can be expanded as a sum
of polynomial moments of `(u^T Z)`. Those are polynomial functions of
`u`. A t-design gives us exact integration of those moments for any
polynomial of degree `<= t`, whereas Monte Carlo over Haar samples gives
variance `O(1/sqrt(M))`.

**Cross-polytope is a 3-design.** The 2d vertices `{+-e_i}` have three
critical properties:

1. *Unit norm:* `||+-e_i|| = 1`.
2. *Isotropy (2-design):*
   ```
   (1/2d) sum_i (+-e_i)(+-e_i)^T = (1/d) I.
   ```
3. *Odd-moment annihilation:* For any odd polynomial,
   ```
   sum_i p(+-e_i) = 0
   ```
   by the sign-flip symmetry `u -> -u` in the point set.

Together, properties 2 and 3 upgrade the cross-polytope from a 2-design
to a 3-design with zero extra work.

**Random rotation per step.** The cross-polytope is axis-aligned. If we
used the same design at every step, the encoder could learn a non-Gaussian
distribution whose deviations from Gaussianity lie in 45-degree rotations
of the design axes, and we'd never detect it. Composing with a fresh
random rotation `R_t` each step (`U_t = R_t D`) preserves the design
property step-by-step while sweeping over all directions in expectation.

**Compute savings.** For `d = 192`, the paper uses `M = 1024` Haar
samples. Cross-polytope uses `M = 2d = 384`, a factor of `1024/384 ≈ 2.7x`
reduction — and the remaining projections each carry exact (not
stochastic) low-moment information.

## 3. Epps-Pulley test statistic

Given `N` samples of a 1D projection `h = Zu` and a target CF `phi_0`,
the Epps-Pulley (EP) statistic is

```
T(h) = integral_{-inf}^{inf} w(t) |phi_N(t) - phi_0(t)|^2 dt,
```

where `phi_N(t) = (1/N) sum_n exp(i t h_n)` is the empirical CF.

**Weight function controls frequency sensitivity.** Expand `phi(t)` near
`t = 0`:
```
phi(t) = 1 + i t mu - (t^2/2) sigma^2 - i (t^3/6) m_3 + (t^4/24) m_4 - ...
```

Near `t = 0`, mismatches in `|phi_N - phi_0|^2` reflect mismatches in
low-order moments. At large `|t|`, Gaussian CF decays as `exp(-t^2/2)`,
so large-`t` behavior is sensitive to fine structure in the tails.

Weighting with a Gaussian `w(t) = exp(-t^2/(2 lambda^2))`:
- Small `lambda` concentrates weight near `t = 0` -> body-sensitive.
- Large `lambda` spreads weight -> tail-sensitive.

**What EP detects after standardization.** AtlasReg standardizes each 1D
projection to zero mean, unit variance before computing EP. This is
deliberate: it decouples distribution *shape* testing from the overall
scale, letting the encoder freely choose its embedding scale. One
consequence: pure collapse to a single point is *not* penalized by EP
alone, because after standardization the collapsed distribution becomes
a degenerate delta at 0 divided by ~0 (in practice, numerical noise). EP
fights *dimensional collapse* — where some subspace of the latent
collapses while others retain variance — because projections onto the
collapsed subspace become non-Gaussian (spiky). The prediction loss
`L_pred` is what directly penalizes pure collapse: a constant latent
makes next-state prediction no better than the mean predictor. Together,
`L_pred + lambda * AtlasReg` covers both failure modes.

**Two-scale weight.** Since body and tail sensitivity are both useful,
take
```
w(t) = alpha exp(-t^2/(2 l1^2)) + (1 - alpha) exp(-t^2/(2 l2^2))
```
with `l1 < l2`. Computationally, this is identical to single-kernel
(one kernel evaluation at each quadrature node, precomputed).

## 4. From 1D to k-D: Henze-Zirkler

One-dimensional projections are blind to **joint** structure. Classic
counterexample: let `X ~ N(0, 1)` and `Y = sign(U) |X|` with `U`
independent uniform. Both `X` and `Y` are marginally standard normal,
but `(X, Y)` is not jointly Gaussian — it has a kink along the axes.
No 1D regularizer of `(X, Y)` detects this.

**Henze-Zirkler (1990)** extends Epps-Pulley to `R^k`, comparing the
empirical k-D CF against standard multivariate Gaussian. For a
k-dimensional sample `{Y_i}_{i=1}^N` whitened to zero mean and identity
covariance, the HZ statistic with smoothing parameter `beta > 0` has a
**closed form**:
```
T_beta(Y) =   (1/N^2) sum_{i,j} exp(-beta^2/2 * |Y_i - Y_j|^2)
            - 2 (1 + beta^2)^(-k/2) (1/N) sum_i exp(-beta^2/(2(1+beta^2)) * |Y_i|^2)
            + (1 + 2 beta^2)^(-k/2).
```

This is differentiable and avoids quadrature entirely. Cost is `O(N^2)`
pairwise — the same as for any kernel-based two-sample test.

**Cramer-Wold still applies.** Testing k-D projections is *strictly
stronger* than testing 1D projections (every 1D projection is a special
k-D projection with k=1). So moving from `k=1` to `k>1` only adds
discriminative power.

In AtlasReg, we sample a fresh k-frame (k orthonormal vectors in `R^d`)
per call via QR of a Gaussian matrix, which gives a uniform sample on
the Stiefel manifold `V_k(R^d)`. This is the k-D analogue of the
per-call rotation in the 1D path.

## 5. Student-t target for heavy-tail-robust matching

In environments with low intrinsic dimensionality embedded in a
high-dimensional latent space (e.g., TwoRoom: 2D state in 192D latent),
forcing isotropic Gaussian is a mismatched prior. The encoder is asked
to spread information evenly over `d` axes when the data only occupies
`~2` of them. The result is either failure (as reported in the LeWM
paper for TwoRoom) or wasteful latent use.

**Student-t relaxation.** An isotropic Student-t with `nu` degrees of
freedom has:
- Heavier tails than Gaussian (controlled by `nu`). Outlier observations
  can produce outlier latents without being penalized.
- A more concentrated body than Gaussian for small `nu`.
- Smooth interpolation: as `nu -> infinity`, `t_nu -> N(0, 1)`.

**CF of isotropic `t_nu`:**
```
phi_0(t) = K_{nu/2}(sqrt(nu) |t|) * (sqrt(nu)|t|)^(nu/2) / (Gamma(nu/2) 2^(nu/2 - 1))
```
where `K_{nu/2}` is the modified Bessel function of the second kind.
We precompute `phi_0` at the quadrature nodes once at initialization
using SciPy. Training cost is unchanged.

**Cramer-Wold still applies.** The theorem doesn't care that the target
is `t_nu` instead of Gaussian — it only requires the target be
identifiable from its projections, which `t_nu` is.

## 6. The composition

Putting it all together, AtlasReg with all four generalizations enabled
computes:

```
1. Sample a k-frame U_t ~ Stiefel(k, d) per step (or apply random
   rotation to a fixed design for k=1).
2. Project Y = Z U_t^T, whiten.
3. For k=1: evaluate phi_N(t_i) at two-scale quadrature nodes against
   precomputed target CF; integrate.
   For k>=2: use Henze-Zirkler closed form.
4. Average over projections (or directly return HZ statistic).
```

The result is a single scalar added to the prediction loss with weight
`lambda`, with one effective hyperparameter to tune (`lambda`).

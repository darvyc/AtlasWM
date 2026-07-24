# AtlasWM: Structured Characteristic-Function Regularization for End-to-End JEPA World Models

**Darvy Cana**

## Abstract

Joint-Embedding Predictive Architectures learn representations by predicting future embeddings rather than reconstructing observations. End-to-end training is attractive for world modelling because it avoids a separately pretrained visual backbone, but the joint encoder-predictor objective admits constant representations. AtlasWM combines an action-conditioned JEPA with AtlasReg, a structured characteristic-function discrepancy that constrains the latent distribution to a prescribed target. The population objective integrates squared characteristic-function differences over directions and frequencies and is identifying under a positive integrable frequency weight. Its finite implementation uses structured spherical projection rules, empirical V-statistic or U-statistic estimators, and either frequency quadrature or an exact Gaussian Baringhaus-Henze-Epps-Pulley form. A Haar-rotated cross-polytope supplies exact spherical cubature for polynomials through degree three and an unbiased spherical average across rotations. Symmetric targets permit an exact antipodal quotient, reducing the `2d` cross-polytope vertices to `d` distinct projection lines. AtlasWM includes one-dimensional Gaussian and Student-t targets, multivariate Gaussian subspace matching, a causal action-conditioned transformer, and latent Cross-Entropy Method planning. The accompanying implementation provides theorem-linked tests, a deterministic statistical verification artifact, and a fixed-budget experimental protocol.

## 1. Introduction

World models compress observations into latent states and predict how those states evolve under actions. A useful latent space should retain task-relevant dynamics, remain numerically stable, and support planning. Joint-Embedding Predictive Architectures, or JEPAs, pursue this objective by comparing a predicted latent with the latent of a future observation.

Let `f_theta` be an encoder and `g_psi` an action-conditioned predictor. A basic predictive objective is

$$
\mathcal{L}_{\mathrm{pred}}(\theta,\psi)
=
\mathbb{E}
\left[
\left\|
 g_\psi(f_\theta(o_{\leq t}),a_{\leq t})
 -f_\theta(o_{t+1})
\right\|_2^2
\right].
$$

This objective admits the constant solution

$$
f_\theta(o)=c,
\qquad
g_\psi(c,a)=c,
$$

which achieves zero prediction loss while discarding all information. A non-degenerate latent-distribution constraint is therefore part of the learning problem rather than an optional diagnostic.

AtlasWM adopts a direct distributional constraint. AtlasReg compares the latent distribution with a target distribution by integrating differences between characteristic functions. Characteristic functions uniquely identify probability distributions, admit differentiable empirical estimators, and support exact Gaussian kernel forms.

### 1.1 Contributions

This work contributes:

1. an identifying population sliced characteristic-function objective for latent distribution matching;
2. a precise decomposition of minibatch, directional, and frequency approximations;
3. structured projection rules based on spherical designs and Haar rotations;
4. an exact antipodal quotient for symmetric-target squared characteristic-function losses;
5. biased non-negative and unbiased finite-sample estimators;
6. exact Gaussian BHEP forms and the classical Henze-Zirkler bandwidth;
7. Gaussian and scaled Student-t targets;
8. a complete action-conditioned JEPA and latent CEM planning implementation;
9. theorem-linked software tests and a reproducible benchmark protocol.

## 2. Related work

### 2.1 Joint-embedding predictive architectures

JEPAs learn representations by predicting embeddings across views, time, or masked context. LeJEPA formalizes isotropic Gaussian embeddings as a favourable representation target and introduces Sketched Isotropic Gaussian Regularization. LeWorldModel applies the end-to-end JEPA formulation to action-conditioned visual dynamics and latent planning.

AtlasWM follows the end-to-end world-model setting and focuses on the statistical estimator used to constrain latent distributions.

### 2.2 Empirical characteristic-function tests

Epps and Pulley construct a normality statistic from a weighted integral of the squared difference between empirical and Gaussian characteristic functions. Baringhaus and Henze extend the construction to multivariate normality. Henze and Zirkler study affine-invariant, consistent multivariate normality tests based on standardized residuals and a sample-size-dependent bandwidth.

The Gaussian-weighted characteristic-function discrepancy is equivalent to a Gaussian-kernel maximum mean discrepancy against a Gaussian target. This relation supplies an exact closed form and clarifies the distinction between biased V-statistics and unbiased U-statistics.

### 2.3 Spherical designs and random rotations

Cramer-Wold identification relates equality of multivariate distributions to equality of all one-dimensional projections. Spherical designs replace spherical integrals of low-degree polynomials with exact finite averages. The cross-polytope vertices form a spherical 3-design. Haar-distributed orthogonal matrices provide rotationally invariant frames whose columns are marginally uniform on the sphere.

## 3. AtlasWM

### 3.1 Encoder

The encoder maps each image observation to a latent vector:

$$
z_t=f_\theta(o_t)\in\mathbb{R}^d.
$$

The reference architecture is a vision transformer with patch embeddings, a classification token, learned positional embeddings, pre-normalized transformer blocks, and a projection head.

### 3.2 Action-conditioned predictor

A causal transformer receives latent and action histories. Action embeddings modulate transformer blocks through adaptive layer normalization. The action at time `t` conditions the latent at time `t` to predict the next latent:

$$
\widehat z_{t+1}
=g_\psi(z_{\leq t},a_{\leq t}).
$$

The modulation projections are zero-initialized so action conditioning enters smoothly from an identity-like initial block.

### 3.3 Objective

AtlasWM minimizes

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{AtlasReg}}.
$$

The prediction term captures dynamics. AtlasReg constrains the marginal latent distribution.

### 3.4 Planning

Given a current latent context and goal latent, the planner samples candidate action sequences, rolls them through the predictor, ranks final predicted latents by squared goal distance, refits a Gaussian distribution to elite sequences, and repeats. The first action can be executed in a receding-horizon loop.

## 4. Population characteristic-function objective

Let `P` be the distribution of `Z in R^d` and `Q` the target distribution. Their characteristic functions are

$$
\varphi_P(\xi)
=
\mathbb{E}_{Z\sim P}[e^{i\xi^\top Z}],
\qquad
\varphi_Q(\xi)
=
\mathbb{E}_{Y\sim Q}[e^{i\xi^\top Y}].
$$

For a direction `u` on the unit sphere and frequency `t`, define the projected characteristic function

$$
\varphi_{P,u}(t)=\varphi_P(tu).
$$

AtlasReg begins with the population objective

$$
\mathcal{D}_w(P,Q)
=
\int_{\mathbb{S}^{d-1}}
\int_{\mathbb{R}}
 w(t)
 \left|
 \varphi_P(tu)-\varphi_Q(tu)
 \right|^2
 dt\,d\sigma(u),
$$

where `sigma` is normalized Haar measure on the sphere and `w` is integrable and positive almost everywhere.

### Proposition 1: Identifiability

Under these conditions,

$$
\mathcal{D}_w(P,Q)\geq 0,
\qquad
\mathcal{D}_w(P,Q)=0
\Longleftrightarrow
P=Q.
$$

#### Proof sketch

A zero non-negative integral implies equality of projected characteristic functions almost everywhere in direction-frequency space. Characteristic functions are continuous, so equality extends to every point `tu` in `R^d`. Equality of characteristic functions determines equality in distribution.

The proposition concerns the continuum objective. Finite directions and finite frequencies define a training discrepancy and do not identify every arbitrary distribution.

## 5. Finite estimator

Given latent samples `z_1,...,z_N`, directions `u_1,...,u_M`, frequencies `t_1,...,t_T`, and integration weights `a_1,...,a_T`, the quadrature estimator is

$$
\widehat{\mathcal{D}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\sum_{r=1}^{T}
 a_r
 \left|
 \widehat\varphi_N(t_r u_m)
 -\varphi_Q(t_r u_m)
 \right|^2,
$$

with

$$
\widehat\varphi_N(\xi)
=
\frac{1}{N}
\sum_{n=1}^{N}e^{i\xi^\top z_n}.
$$

The estimator contains three approximations:

1. the population distribution is represented by a minibatch;
2. the sphere is represented by finite directions;
3. the frequency integral is represented by finite quadrature.

### 5.1 Biased V-statistic

At a fixed projected frequency, the non-negative estimator is

$$
V_N(t)
=
\left|
\widehat\varphi_N(t)-\varphi_Q(t)
\right|^2.
$$

When samples are iid from the target,

$$
\mathbb{E}[V_N(t)]
=
\frac{1}{N}
\left(1-|\varphi_Q(t)|^2\right).
$$

A finite target sample therefore has a positive expected biased loss.

### 5.2 Unbiased U-statistic

An unbiased estimate of the empirical characteristic-function squared modulus is

$$
U_\varphi(t)
=
\frac{N|\widehat\varphi_N(t)|^2-1}{N-1}.
$$

Substitution yields an estimator whose expectation equals the population squared discrepancy. It can be negative on an individual batch. The iid derivation is not applicable after batch-dependent studentization or whitening.

## 6. Structured projections

### 6.1 Cross-polytope design

The full direction set

$$
\mathcal{C}_d
=
\{+e_1,-e_1,\ldots,+e_d,-e_d\}
$$

forms a spherical 3-design. It integrates linear, quadratic, and cubic spherical polynomials exactly:

$$
\frac{1}{2d}\sum_{u\in\mathcal{C}_d}u=0,
$$

$$
\frac{1}{2d}\sum_{u\in\mathcal{C}_d}uu^\top=\frac{I_d}{d},
$$

and every third-order monomial average is zero.

This guarantee does not extend to the complete characteristic-function integrand. The first missed even moment is order four. For a fixed vector `x`,

$$
\mathbb{E}_{u\sim\sigma}[(u^\top x)^4]
=
\frac{3\|x\|^4}{d(d+2)},
$$

while the unrotated cross-polytope gives

$$
\frac{1}{d}\sum_{j=1}^{d}x_j^4.
$$

### 6.2 Haar rotation

Let `R` be Haar distributed on the orthogonal group. For any integrable `f` on the sphere,

$$
\mathbb{E}_R
\left[
\frac{1}{d}\sum_{j=1}^{d}f(Re_j)
\right]
=
\int_{\mathbb{S}^{d-1}}f(u)\,d\sigma(u).
$$

The directions within a frame are dependent, but each column is marginally uniform. The resulting estimator is structured within a step and unbiased across rotations.

### 6.3 Antipodal quotient

For a symmetric target, `varphi_Q` is real and even. Since

$$
\varphi_{P,-u}(t)
=
\overline{\varphi_{P,u}(t)},
$$

it follows that

$$
\left|
\varphi_{P,-u}(t)-\varphi_Q(t)
\right|^2
=
\left|
\varphi_{P,u}(t)-\varphi_Q(t)
\right|^2.
$$

The `2d` vertices therefore represent `d` distinct lines for this loss. AtlasReg evaluates one representative of each antipodal pair.

## 7. Gaussian closed form

For a sample `y_1,...,y_N` in `R^k` and target `N(0,I_k)`, the Gaussian-weighted discrepancy is

$$
\begin{aligned}
\mathrm{BHEP}_\beta(Y)
={}&
\frac{1}{N^2}
\sum_{i,j}
\exp\left(-\frac{\beta^2}{2}\|y_i-y_j\|^2\right)
\\
&-2(1+\beta^2)^{-k/2}
\frac{1}{N}
\sum_i
\exp\left(
-\frac{\beta^2}{2(1+\beta^2)}\|y_i\|^2
\right)
\\
&+(1+2\beta^2)^{-k/2}.
\end{aligned}
$$

This is the squared maximum mean discrepancy between the empirical measure and the Gaussian target for the corresponding Gaussian kernel.

If samples are iid standard Gaussian,

$$
\mathbb{E}[\mathrm{BHEP}_\beta]
=
\frac{1}{N}
\left[
1-(1+2\beta^2)^{-k/2}
\right].
$$

The classical Henze-Zirkler bandwidth is

$$
\beta_N
=
2^{-1/2}
\left[
\frac{(2k+1)N}{4}
\right]^{1/(k+4)}.
$$

A fixed `beta` defines a fixed-bandwidth BHEP objective. The sample-size-dependent bandwidth reproduces the classical normality-test convention.

## 8. Target and shape objectives

### 8.1 Raw target matching

Raw projections

$$
h_n=u^\top z_n
$$

retain location and scale. Matching every projection to `N(0,1)` constrains the full population law to `N(0,I_d)`.

### 8.2 Studentized shape testing

Projection-wise studentization uses

$$
\widetilde h
=
\frac{h-\overline h}{s_h}.
$$

This removes batch location and scale and tests standardized marginal shape. It does not enforce a particular latent mean or covariance.

### 8.3 Multivariate subspaces

For an orthonormal frame `U in R^{k x d}`,

$$
Y=ZU^\top.
$$

AtlasReg applies the exact Gaussian BHEP discrepancy in the sampled subspace. Optional whitening produces a covariance-standardized normal-shape objective within each subspace. Averaging over multiple frames reduces subspace-sampling variance.

### 8.4 Student-t target

For degrees of freedom `nu` and scale `s`, the univariate Student-t characteristic function is

$$
\varphi_{\nu,s}(t)
=
\frac{2^{1-\nu/2}}{\Gamma(\nu/2)}
\left(\sqrt{\nu}s|t|\right)^{\nu/2}
K_{\nu/2}(\sqrt{\nu}s|t|).
$$

For `nu > 2`, the unit-variance scale is

$$
s=\sqrt{\frac{\nu-2}{\nu}}.
$$

The implementation uses a logarithmic scaled-Bessel evaluation and a numerical Fourier-integral fallback for difficult parameter ranges.

## 9. Algorithm

### Algorithm 1: AtlasWM training step

```text
Input: observation trajectories O, action trajectories A,
       encoder f, predictor g, regularizer configuration,
       regularizer weight lambda

1. Encode every frame:
       Z <- f(O)
2. Predict next latents with causal action conditioning:
       Z_hat[t+1] <- g(Z[<=t], A[<=t])
3. Compute prediction loss:
       L_pred <- mean squared error(Z_hat[:, :-1], Z[:, 1:])
4. Build structured projection directions or k-frames
5. Project latent samples
6. Optionally studentize or whiten
7. Compute quadrature ECF or exact Gaussian BHEP discrepancy
8. Average over directions or subspaces:
       L_reg <- AtlasReg(Z)
9. Return:
       L <- L_pred + lambda * L_reg
```

## 10. Complexity

Let `N` be the number of latent samples, `M` the number of one-dimensional directions, `T` the number of quadrature frequencies, `S` the number of subspaces, and `k` the subspace dimension.

| Component | Time complexity |
|---|---:|
| 1D quadrature | `O(TNM)` |
| 1D exact Gaussian closed form | `O(MN^2)` |
| k-D BHEP | `O(SNkd + SN^2k)` |
| Haar rotation by dense QR | `O(d^3)` |

For symmetric targets, the cross-polytope evaluation uses `M=d` distinct lines rather than `2d` vertices.

## 11. Verification

The software artifact tests:

- unit norms and moment identities of spherical designs;
- exact second and third cross-polytope moments;
- the fourth-moment limitation;
- antipodal loss equality;
- numerical quadrature against the Gaussian closed form;
- the analytic finite-sample Gaussian floor;
- biased and unbiased estimator behaviour;
- Student-t characteristic-function stability;
- differentiability of AtlasReg;
- action alignment in autoregressive rollout;
- end-to-end gradient flow.

The command

```bash
python scripts/reproduce_statistics.py --seed 42 --output outputs/statistical_verification.json
```

produces a machine-readable verification record.

## 12. Empirical protocol

Control-performance evaluation follows a fixed-budget, multi-seed protocol. Required baselines include prediction-only training, iid Haar ECF matching, covariance regularization, full-latent Gaussian MMD, and three AtlasReg configurations. Primary metrics cover planning success, goal distance, multi-horizon prediction, effective latent rank, held-out ECF discrepancy, training throughput, planning latency, and peak memory.

The complete protocol is specified in `docs/benchmark_protocol.md`. Raw per-seed measurements are required for every aggregate claim.

## 13. Limitations

1. The finite estimator does not identify every arbitrary distribution from finitely many directions and frequencies.
2. The optimization objective does not supply a global theorem excluding every collapsed stationary point.
3. An isotropic target is a modelling choice and may be mismatched to a specific environment or downstream planner.
4. Student-t matching is implemented in the one-dimensional projection path.
5. Latent CEM planning can exploit predictor error and requires external safety constraints for physical deployment.
6. Mathematical correctness does not imply empirical superiority. Task-level conclusions require the declared benchmark protocol.

## 14. Conclusion

AtlasWM provides a direct statistical formulation of latent distribution regularization for end-to-end predictive world models. AtlasReg connects Cramer-Wold identification, empirical characteristic functions, spherical designs, Gaussian-kernel discrepancies, and multivariate normality testing in a differentiable training objective. The release treats population guarantees, finite estimators, software validation, and task-level evidence as distinct layers. This structure supports precise mathematical claims and controlled empirical evaluation.

## References

1. H. Cramer and H. Wold. Some Theorems on Distribution Functions. *Journal of the London Mathematical Society*, 11(4), 1936.
2. P. Delsarte, J. M. Goethals, and J. J. Seidel. Spherical Codes and Designs. *Geometriae Dedicata*, 6:363-388, 1977. DOI: 10.1007/BF03187604.
3. T. W. Epps and L. B. Pulley. A Test for Normality Based on the Empirical Characteristic Function. *Biometrika*, 70(3):723-726, 1983. DOI: 10.1093/biomet/70.3.723.
4. L. Baringhaus and N. Henze. A Consistent Test for Multivariate Normality Based on the Empirical Characteristic Function. *Metrika*, 35:339-348, 1988. DOI: 10.1007/BF02613322.
5. N. Henze and B. Zirkler. A Class of Invariant Consistent Tests for Multivariate Normality. *Communications in Statistics - Theory and Methods*, 19(10):3595-3617, 1990. DOI: 10.1080/03610929008830400.
6. F. Mezzadri. How to Generate Random Matrices from the Classical Compact Groups. *Notices of the AMS*, 54(5):592-604, 2007.
7. R. Balestriero and Y. LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544, 2025.
8. L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, and R. Balestriero. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312, 2026.

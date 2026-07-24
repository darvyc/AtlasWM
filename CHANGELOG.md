# Changelog

All notable changes to AtlasWM are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Replaced claims of fully deterministic distribution matching with precise
  population, cubature, random-rotation, and finite-estimator statements.
- Made raw target matching the default and exposed batch-standardized shape
  testing explicitly.
- Deduplicated antipodal projections for symmetric targets.
- Corrected autoregressive action alignment during planning rollouts.
- Restored the intended zero initialization of AdaLN action modulation.

### Added
- Biased and unbiased empirical characteristic-function estimators.
- Exact Gaussian BHEP closed forms, the finite-sample null floor, and the
  classical Henze-Zirkler bandwidth rule.
- Stable large-degree-of-freedom Student-t characteristic-function evaluation.
- Mathematical tests for design limitations, estimator bias, analytic closed
  forms, collapse detection, and rollout alignment.

## [0.1.0] - 2026-04-22

### Added
- Initial research scaffold.
- `AtlasReg` with cross-polytope, simplex, Haar, Gaussian, Student-t, and
  k-dimensional Gaussian paths.
- `AtlasWM`, `CEMPlanner`, toy trajectories, configs, scripts, and tests.

[0.1.0]: https://github.com/darvyc/AtlasWM/releases/tag/v0.1.0

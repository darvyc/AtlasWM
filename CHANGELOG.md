# Changelog

All notable AtlasWM releases are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-24

### Added

- identifying population sliced characteristic-function objective;
- biased non-negative and unbiased finite-sample estimators;
- exact Gaussian BHEP discrepancy and analytic target null floor;
- classical Henze-Zirkler bandwidth selection;
- antipodal quotient for symmetric-target cross-polytope projections;
- numerically stable scaled Student-t characteristic functions;
- explicit raw-target, studentized, and whitened matching modes;
- multiple k-dimensional subspace evaluation;
- action-aligned autoregressive latent rollout;
- validated generic NPZ trajectory dataset interface;
- deterministic dataset fingerprints and provenance-rich atomic checkpoints;
- statistical verification script with machine-readable output;
- technical manuscript, model card, reproducibility standard, and benchmark protocol;
- Python 3.10, 3.11, and 3.12 continuous integration;
- source and wheel package build verification;
- security policy, dependency update policy, and release-oriented contribution standard.

### Mathematical specification

- The continuum characteristic-function objective identifies equality in distribution under a positive integrable frequency weight.
- The cross-polytope guarantee is stated as exact spherical cubature through degree three.
- Haar-rotated orthonormal frames are characterized as unbiased spherical estimators across rotations.
- The degree-four cross-polytope limitation is given explicitly.
- Finite direction, frequency, and batch approximations are separated from population claims.
- Finite-sample Gaussian V-statistic bias is derived analytically.
- Prediction-only constant collapse is stated as an admissible global optimum.

### Software

- Public package version set to `1.0.0`.
- Package metadata, citation metadata, documentation links, and research classifiers completed.
- The `atlaswm-train` command provides the installed training entry point.
- Toy and NPZ trajectory datasets share the same training interface.
- Checkpoints record model and optimizer state, resolved configuration, loss history, random-number-generator state, package version, Git commit, and dataset fingerprint.
- CEM rollout conditions each predicted transition on the corresponding action.
- AdaLN action modulation retains zero initialization.
- The quickstart uses a held-out linear probe and labels it as an integration diagnostic.

## [0.1.0] - 2026-04-22

### Added

- initial AtlasWM implementation;
- AtlasReg with cross-polytope, simplex, Haar, Gaussian, Student-t, and k-dimensional Gaussian paths;
- vision transformer encoder and causal action-conditioned predictor;
- latent CEM planner;
- synthetic visual trajectory environment;
- training, benchmarking, configuration, and test modules.

[1.0.0]: https://github.com/darvyc/AtlasWM/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/darvyc/AtlasWM/releases/tag/v0.1.0

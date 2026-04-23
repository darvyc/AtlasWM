# Changelog

All notable changes to AtlasWM are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-22

### Added
- Initial release.
- `AtlasReg` anti-collapse regularizer generalizing SIGReg along four axes:
  - Spherical 2- and 3-designs (cross-polytope, simplex) as deterministic
    alternatives to Haar sampling.
  - Per-step random rotation for axis-alignment immunity.
  - Two-scale Gaussian kernel weight for joint body + tail sensitivity.
  - Optional isotropic Student-t target via SciPy precomputation.
  - k-D subspace testing via the closed-form Henze-Zirkler statistic.
- `AtlasWM` end-to-end model: ViT-Tiny encoder + causal transformer predictor
  with AdaLN action conditioning.
- `CEMPlanner` for latent-space model-predictive control.
- Synthetic 2D toy environment (`ToyTrajectoryDataset`) for smoke testing.
- Training CLI (`scripts/train.py`) and benchmark (`scripts/bench.py`).
- Configs for default, Henze-Zirkler k=4, and PushT-style setups.
- Full pytest suite covering designs, targets, regularizer, and model integration.
- Theory documentation in `docs/theory.md`.

[0.1.0]: https://github.com/darvyc/atlaswm/releases/tag/v0.1.0

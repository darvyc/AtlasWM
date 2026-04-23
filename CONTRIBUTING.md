# Contributing to AtlasWM

Contributions are very welcome. This document describes how to propose changes.

## Development setup

```bash
git clone https://github.com/darvyc/atlaswm.git
cd atlaswm
pip install -e ".[dev]"
```

## Before submitting a PR

1. **Run the tests**:
   ```bash
   pytest tests/ -v
   ```
2. **Run the linter**:
   ```bash
   ruff check atlaswm/ tests/
   ruff format atlaswm/ tests/
   ```
3. **If you added a new public class or function**, add a docstring and a test.
4. **If you changed the math**, update `docs/theory.md` alongside the code.

## Style

- Python 3.10+ type hints on all public APIs.
- Docstrings follow Google style.
- Keep modules focused: one idea per file where possible.
- No external deps in `atlaswm/` core beyond `torch` and `numpy`. SciPy is optional.

## What we're looking for

- **New designs.** Known spherical `t`-designs for specific `(d, t)` — e.g. the `(d=3, t=3)` icosahedron or `(d=4, t=5)` 24-cell.
- **New targets.** Alternative heavy-tailed or bounded-support targets (e.g. uniform-on-ball, Laplace, elliptical families).
- **Environment integrations.** Datasets + configs for standard benchmarks (PushT, OGBench, DMControl).
- **Empirical ablation studies.** Head-to-head evaluations on a fixed compute budget.
- **Documentation.** Worked examples, tutorials, explanations of the math.

## Non-goals

- General self-supervised pre-training frameworks. AtlasWM is focused on action-conditioned latent world modeling.
- Reward-based RL algorithms. AtlasWM is reward-free by design.

## Reporting bugs

Please include:

- Python and PyTorch versions (`python -c "import sys, torch; print(sys.version); print(torch.__version__)"`)
- The config you used (or a minimal reproducer)
- The actual vs. expected behavior
- The full traceback if applicable

# Contributing to AtlasWM

AtlasWM welcomes contributions to its mathematics, software, experiments, documentation, and environment integrations.

## Development setup

```bash
git clone https://github.com/darvyc/AtlasWM.git
cd AtlasWM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Contribution workflow

1. Create a focused branch from `main`.
2. Add or update tests with every behavioural change.
3. Update mathematical documentation when an objective, estimator, target, or guarantee changes.
4. Run the complete verification commands.
5. Open a pull request describing the claim, implementation, evidence, and compatibility impact.

## Required checks

```bash
python -m compileall atlaswm scripts
ruff check atlaswm tests scripts
ruff format --check atlaswm tests scripts
pytest --cov=atlaswm --cov-report=term-missing
python scripts/reproduce_statistics.py --trials 16 --samples 32 --dim 4
python -m build
```

A pull request should not weaken existing mathematical tests or replace a precise statement with a broader unsupported claim.

## Code standards

- Support Python 3.10 or later.
- Type all public APIs.
- Use Google-style docstrings for public classes and functions.
- Keep modules focused on one mathematical or architectural responsibility.
- Preserve differentiability unless a function is explicitly diagnostic.
- Validate public configuration values at construction time.
- Include informative error messages for invalid shapes and parameter ranges.
- Keep the core dependency set limited to PyTorch, NumPy, tqdm, and PyYAML. SciPy is optional for Student-t evaluation.

## Mathematical contributions

A mathematical contribution must include:

- a precise definition of the object being computed;
- assumptions and quantifiers;
- a proof, derivation, or primary-source citation;
- the distinction between a population statement and a finite estimator;
- numerical-stability considerations;
- tests against an independent calculation, limiting case, or counterexample.

Examples include:

- new spherical designs or cubature rules;
- variance analysis for structured rotations;
- new characteristic-function targets;
- unbiased or lower-variance estimators;
- closed-form kernels;
- concentration or approximation bounds;
- collapse and stationary-point analysis.

## Experimental contributions

Empirical pull requests must follow [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Include:

- committed configurations;
- dataset provenance and checksums;
- matched compute and planner budgets;
- at least five final-reporting seeds;
- raw per-seed metrics;
- confidence intervals;
- system and timing information;
- commands that regenerate tables and figures.

Do not report only the strongest seed or select checkpoints using test metrics.

## Documentation contributions

Documentation should:

- use declarative, current-state language;
- define notation before using it;
- distinguish theorem, estimator, implementation, hypothesis, and empirical result;
- provide executable examples;
- link claims to source code, tests, or primary references.

## Pull request template

A strong pull request answers:

```text
What claim or capability does this add?
What assumptions does it require?
Which files implement it?
Which tests validate it?
What is the time and memory cost?
Does it change public APIs or default behaviour?
Which documentation and citations support it?
```

## Reporting bugs

Include:

- AtlasWM version and Git commit SHA;
- Python, PyTorch, CUDA, and operating-system versions;
- exact configuration;
- minimal reproduction command;
- expected and observed behaviour;
- complete traceback;
- input tensor shapes and dtypes;
- whether the issue reproduces on CPU.

## Research scope

High-value contributions include:

- PushT, OGBench, and DMControl integrations;
- fixed-budget baseline comparisons;
- representation-geometry diagnostics;
- planning under uncertainty;
- scalable projection and kernel implementations;
- published checkpoints and raw evaluation records;
- proofs and counterexamples that sharpen AtlasReg's guarantees.

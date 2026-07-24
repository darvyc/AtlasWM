# Contributing to AtlasWM

AtlasWM accepts contributions to mathematics, software, datasets, experiments, documentation and environment integrations.

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
2. Add regression tests for every behavioural change.
3. Update theory and claim boundaries when an estimator or guarantee changes.
4. Run the complete verification commands.
5. Open a pull request describing capability, assumptions, evidence and compatibility impact.

## Required checks

```bash
ruff check atlaswm scripts tests
python -m compileall atlaswm scripts
pytest --cov=atlaswm --cov-report=term-missing
python scripts/reproduce_statistics.py --trials 16 --samples 32 --dim 4
python -m build
```

## Code standards

- support Python 3.10 or later;
- type public APIs;
- keep modules focused on one architectural or mathematical responsibility;
- preserve differentiability unless a function is explicitly diagnostic;
- validate public configuration values and tensor shapes;
- avoid batch-coupled operations in causal representation paths;
- maintain training/planning action alignment;
- keep core dependencies limited to PyTorch, NumPy and PyYAML;
- treat SciPy as optional for Student-t evaluation.

## Mathematical contributions

Include a precise definition, assumptions, derivation or primary citation, population-versus-finite distinction, numerical analysis and a test against an independent calculation, limiting case or counterexample.

## Experimental contributions

Follow `docs/benchmark_protocol.md` and `REPRODUCIBILITY.md`. Include committed configurations, dataset provenance and checksums, matched budgets, final seed-level records, confidence intervals, system information and regeneration commands. Do not select the strongest seed or tune on test metrics.

## Pull request evidence

A pull request should state:

```text
What capability or claim is added?
What assumptions does it require?
Which files implement it?
Which tests validate it?
What is the time and memory cost?
Does it change public APIs or defaults?
Which documentation and primary references support it?
```

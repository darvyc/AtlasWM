# Reproducibility Standard

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pip freeze > outputs/pip-freeze.txt
```

Every reported result must record the Git commit, Python, PyTorch, NumPy, SciPy, CUDA, cuDNN, operating system, accelerator, CPU, RAM, optimizer steps, examples processed, wall-clock time and peak memory.

## Verification

```bash
python -m compileall atlaswm scripts
pytest --cov=atlaswm --cov-report=term-missing
python scripts/reproduce_statistics.py --seed 42 --output outputs/statistical_verification.json
python -m build
```

## Randomness

AtlasWM seeds Python, NumPy, CPU PyTorch and CUDA PyTorch. Strict mode additionally enables deterministic algorithms and disables cuDNN benchmarking.

Determinism is defined for a fixed software, hardware and execution stack. Identical seeds do not imply bitwise identity across different PyTorch versions or devices.

## Dataset provenance

Each dataset record must include:

- canonical name and version;
- source and licence;
- raw checksums;
- preprocessing code and parameters;
- trajectory-level split definitions;
- trajectory, frame and transition counts;
- image and action normalization;
- exclusions and corruption handling.

AtlasWM stores a deterministic fingerprint of the resolved dataset description inside each checkpoint.

## Seed policy

Use disjoint seed sets for hyperparameter selection and final reporting. Final task-level comparisons require at least five seeds. Report all seed-level values.

Recommended reporting seeds:

```text
11, 23, 37, 53, 71
```

## Artifact contract

Every run writes:

```text
resolved_config.yaml
system.json
metrics.jsonl
checkpoint_last.pt
checkpoint_best.pt
evaluation.json
```

The benchmark suite writes raw run records before aggregation. Every aggregate must be reproducible from `runs.jsonl`.

## Resume contract

A checkpoint records model, optimizer, scheduler, epoch, step, best validation value, complete loss history, RNG states, resolved configuration, dataset fingerprint, Git commit and system metadata. Resume is valid only when the data and software stack are compatible.

## Claim policy

- mathematical claims require derivation and theorem-linked tests;
- software claims require automated regression tests;
- efficiency claims require synchronized matched-hardware measurements;
- prediction claims require held-out multi-horizon errors;
- representation claims require independent geometry diagnostics and probes;
- control claims require matched planners, episodes and seed-level uncertainty;
- sample-efficiency claims require curves against transitions seen.

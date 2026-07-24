# AtlasWM Reproducibility Standard

This document defines the software, randomness, data, compute, evaluation, and reporting conditions required for an AtlasWM result to be independently reproduced.

## 1. Environment

Create an isolated Python environment and install the locked research dependencies declared by the release:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Record the complete environment:

```bash
python - <<'PY'
import json
import platform
import sys

import numpy
import torch

print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "device_count": torch.cuda.device_count(),
}, indent=2))
PY

python -m pip freeze > outputs/pip-freeze.txt
```

Every reported experiment must include:

- AtlasWM version and Git commit SHA;
- Python, PyTorch, CUDA, cuDNN, NumPy, and SciPy versions;
- operating system;
- accelerator model and count;
- CPU model and RAM;
- training wall-clock time;
- peak accelerator memory;
- total optimizer steps and examples processed.

## 2. Software verification

Run the package checks before any experiment:

```bash
python -m compileall atlaswm scripts
pytest
python -m build
```

Run the deterministic statistical verification artifact:

```bash
python scripts/reproduce_statistics.py \
  --seed 42 \
  --dim 8 \
  --samples 128 \
  --trials 128 \
  --output outputs/statistical_verification.json
```

The generated JSON records:

- cross-polytope second-moment error;
- the degree-four cubature counterexample;
- the analytic Gaussian BHEP null floor;
- Monte Carlo estimates of biased and unbiased Gaussian discrepancies;
- the unit-variance Student-t scale.

## 3. Randomness

Use the same seed for Python, NumPy, and PyTorch:

```python
import random

import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

For strict deterministic debugging:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Strict deterministic kernels can reduce throughput and can be unavailable for specific operations. Research comparisons must use the same determinism policy across all methods.

## 4. Seed policy

Use at least five independent seeds for task-level comparisons. Ten seeds are preferred when success rates are high-variance or datasets are small.

Default reporting seeds:

```text
11, 23, 37, 53, 71
```

Hyperparameter selection seeds must be disjoint from final reporting seeds. Test-environment episodes must use a fixed seed list shared by every compared method.

## 5. Data provenance

For every dataset, record:

- canonical dataset name and version;
- download source;
- license;
- cryptographic checksum of raw files;
- preprocessing code and parameters;
- train, validation, and test split definitions;
- number of trajectories, frames, and transitions;
- image resolution and action normalization;
- exclusions or corrupted records.

The included toy environment is generated procedurally from `ToyEnvConfig`. Its reproducible state is determined by the configuration, dataset seed, number of trajectories, trajectory length, and sub-trajectory length.

## 6. Training protocol

The default toy training command is:

```bash
atlaswm-train --config configs/default.yaml
```

A one-epoch integration run is:

```bash
atlaswm-train \
  --config configs/default.yaml \
  trainer.epochs=1 \
  data.n_trajectories=16 \
  output.dir=outputs/integration
```

For comparisons, hold constant:

- training data and ordering policy;
- encoder and predictor architecture;
- optimizer and learning-rate schedule;
- batch size and gradient accumulation;
- total optimizer steps;
- precision mode;
- augmentation policy;
- action preprocessing;
- planner budget;
- evaluation episodes.

The regularizer weight may be tuned only through the declared validation protocol. Report the search range and selected value for every method.

## 7. Checkpoint contents

A checkpoint must contain:

- model state;
- optimizer state;
- full resolved configuration;
- epoch and global step;
- random-number-generator states;
- Git commit SHA;
- package version;
- training-data fingerprint;
- metric history.

Checkpoint filenames must identify environment, method, seed, and step:

```text
{environment}_{method}_seed{seed}_step{step}.pt
```

## 8. Evaluation

Task-level evaluation follows [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md). Each run must export machine-readable records with one row per seed and environment.

Minimum fields:

```text
environment
method
seed
checkpoint
success_rate
final_goal_distance
prediction_mse
latent_effective_rank
covariance_error
regularizer_value
train_steps
train_seconds
peak_memory_mb
planning_ms_per_action
```

## 9. Statistical reporting

For each metric report:

- per-seed values;
- arithmetic mean;
- sample standard deviation;
- 95 percent confidence interval;
- paired effect against the designated baseline when seeds are shared;
- number of seeds and evaluation episodes.

Use bootstrap confidence intervals for bounded success rates and paired bootstrap intervals for paired method comparisons. Avoid selecting the best seed or the best checkpoint using test metrics.

## 10. Artifact directory

Use the following output structure:

```text
outputs/
├── environment.json
├── pip-freeze.txt
├── statistical_verification.json
├── runs/
│   └── {environment}/{method}/seed_{seed}/
│       ├── config.yaml
│       ├── metrics.jsonl
│       ├── checkpoint.pt
│       └── system.json
└── reports/
    ├── aggregate.csv
    ├── tables.md
    └── figures/
```

## 11. Reproduction checklist

A result satisfies the AtlasWM reproducibility standard when:

- the commit SHA is public;
- installation succeeds from a clean environment;
- the test suite passes;
- the statistical verification artifact is generated;
- data provenance and checksums are supplied;
- all configurations are committed;
- at least five final-reporting seeds are available;
- raw per-seed metrics are published;
- compute and timing are reported;
- aggregate statistics can be regenerated from raw outputs;
- no test-set metric was used for model or hyperparameter selection.

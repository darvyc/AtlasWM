# Security Policy

## Supported version

Security fixes are applied to the current `1.x` release line.

| Version | Supported |
|---|---|
| 1.x | Yes |
| 0.x | No |

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose users, systems, data, credentials, or model artifacts.

Use GitHub's private vulnerability reporting interface for this repository when available. Include:

- affected AtlasWM version and commit SHA;
- operating system, Python, PyTorch, and CUDA versions;
- threat model and required attacker capabilities;
- minimal reproduction;
- affected files and functions;
- impact assessment;
- proposed mitigation, when known.

Reports should avoid including live credentials, private datasets, personal data, or exploit payloads against third-party systems.

## Security scope

Relevant reports include:

- arbitrary code execution through configuration or checkpoint loading;
- unsafe path handling;
- dependency vulnerabilities with a demonstrated AtlasWM impact;
- denial-of-service inputs affecting public APIs;
- unintended disclosure of training data, checkpoints, or environment information;
- workflow or release-process compromise;
- planner interfaces that bypass declared action constraints.

## Checkpoint safety

PyTorch checkpoints can execute code when loaded through unsafe pickle paths. Load checkpoints only from trusted sources. Prefer `weights_only=True` where supported and validate checkpoint provenance and checksums.

## Deployment safety

AtlasWM is research software. Physical deployment requires independent safety constraints, action validation, monitoring, fallback control, and environment-specific hazard analysis. Latent prediction accuracy is not a safety guarantee.

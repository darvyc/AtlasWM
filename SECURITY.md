# Security Policy

## Supported version

Security fixes are applied to the current `2.x` release line.

| Version | Supported |
|---|---|
| 2.x | Yes |
| 1.x and earlier | No |

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose users, systems, data, credentials or model artifacts. Use GitHub private vulnerability reporting when available and include:

- affected AtlasWM version and commit SHA;
- operating system, Python, PyTorch and CUDA versions;
- threat model and required attacker capabilities;
- minimal reproduction;
- affected files and functions;
- impact assessment;
- proposed mitigation, when known.

Do not include live credentials, private datasets, personal data or exploit payloads against third-party systems.

## Security scope

Relevant reports include:

- arbitrary code execution through configuration or checkpoint loading;
- unsafe path handling;
- dependency vulnerabilities with demonstrated AtlasWM impact;
- denial-of-service inputs affecting public APIs;
- unintended disclosure of training data, checkpoints or environment information;
- workflow or release-process compromise;
- planner interfaces that bypass declared action constraints.

## Checkpoint safety

PyTorch checkpoints use pickle-compatible serialization. Load complete training checkpoints only from trusted sources and verify their provenance and checksums. For untrusted model distribution, publish and consume a weights-only format rather than optimizer or RNG state.

## Deployment safety

AtlasWM is research software. Physical deployment requires independent action validation, monitoring, fallback control, environment-specific hazard analysis and a verified safety layer. Latent prediction accuracy is not a safety guarantee.

# AtlasWM Model Card

## Summary

AtlasWM is an action-conditioned latent world model trained from image trajectories. It contains a frame-independent ViT encoder, a causal action-conditioned predictor, a configurable latent-distribution regularizer and a latent CEM planner.

## Intended use

- controlled research on joint-embedding predictive architectures;
- representation-collapse studies;
- characteristic-function distribution matching;
- action-conditioned latent prediction;
- goal-conditioned planning in simulated environments;
- matched comparisons of anti-collapse objectives.

## Out-of-scope use

- unsupervised deployment on vehicles, robots, medical devices or industrial equipment;
- safety-critical control without an independent verified safety layer;
- decisions affecting legal rights, health, employment or access to services;
- claims of causal understanding based solely on latent prediction;
- unrestricted planning where model exploitation can cause physical harm.

## Inputs

```text
observations: (batch,time,channels,height,width)
actions:      (batch,time,action_dim)
```

The action at index `t` is the control used to predict the transition from latent `z_t` to `z_(t+1)`.

## Outputs

- latent embeddings;
- predicted future latent embeddings;
- total, prediction and regularizer losses;
- optimized action sequences;
- prediction, geometry, probe and control metrics.

## Normalization boundary

Encoder and predictor latent heads use no BatchNorm or other operation that computes statistics across samples or time. Transformer LayerNorm remains per token and per sample. This prevents the representation of one frame from depending on unrelated batch members or future frames.

## Data boundary

Large datasets should use memory-mapped NPY arrays or sharded manifests. The NPZ adapter loads its archive into memory and is intended for compact datasets or conversion.

## Limitations

- finite directions and frequencies do not identify every arbitrary distribution;
- isotropic Gaussian or Student-t targets may be mismatched to a task;
- distribution matching does not guarantee semantic sufficiency or controllability;
- one-step accuracy does not guarantee long-horizon rollout accuracy;
- CEM can exploit predictor error;
- the planner has no inherent obstacle, uncertainty or safety model;
- task-level conclusions require matched multi-seed evidence.

## Risk controls

- reject missing action histories for multi-frame contexts;
- validate sequence capacity before training;
- split complete trajectories rather than overlapping windows;
- monitor multiple independent collapse diagnostics;
- use held-out state probes and multi-horizon prediction metrics;
- evaluate in receding horizon with frequent re-observation;
- report raw per-seed outputs and matched compute;
- retain independent safety constraints for physical deployment.

# v0.1 Experiment Rollout — Implementation Overview

**Goal:** Execute the v0.1 experiment matrix on a Vast 3×3090 machine with continuous queue-based scheduling, wandb logging, and one fixed hardware profile across all comparisons.

**Architecture:** Extend the existing active training stack to support A1/A2/A3 architecture families through YAML config, make target extraction loss-conditional, add wandb integration, calibrate a worst-case hardware profile, and run a simple queue system that keeps all 3 GPUs busy.

**Tech Stack:** Python, PyTorch, Transformers, wandb, YAML configs

## Phases

| Phase | Name | Delivers | Depends On |
|-------|------|----------|------------|
| 1 | v0.1 support closure | Trainable A1/A2/A3 families, loss-conditional targets, wandb integration, truncation logging | — |
| 2 | Hardware calibration | Locked 3090 profile for seq_len=1024, worst-case loss regime | Phase 1 |
| 3 | Config matrix generation | YAML configs for all P1/P2/P3/P4 experiments | Phase 2 |
| 4 | Queue execution | Simple bash runner for 3-GPU execution | Phase 3 |
| 5 | Launch and monitoring | Smoke test on Vast, then full matrix | Phase 4 |

**Current:** Phase 5 (ready for Vast deployment)

---

## Key Constraints

1. **seq_len=1024** fixed for all v0.1 experiments; truncation accepted but logged
2. **One hardware profile** across the matrix; sized against worst-case loss regime
3. **Queue-driven** execution with 1 worker per GPU
4. **All config-driven** — no ad hoc script selection per experiment

## Phase 0 vs Phase 1 Note

The design spec calls Phase 0 "v0.1 support closure", but this is the **first phase of implementation**. I'm labeling it Phase 1 to follow the skill's convention that implementation phases start at 1.

## Success Criteria

- Every matrix experiment runs from YAML alone
- Train and eval metrics visible in wandb and on disk
- Machine runs continuously without manual intervention between configs
- Checkpoints resume after interruption
- Artifacts sufficient to produce all v0.1 plots

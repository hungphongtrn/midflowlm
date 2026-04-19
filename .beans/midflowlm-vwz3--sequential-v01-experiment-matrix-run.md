---
# midflowlm-vwz3
title: Sequential v0.1 Experiment Matrix Run
status: completed
type: task
priority: normal
created_at: 2026-04-19T13:17:42Z
updated_at: 2026-04-19T13:25:00Z
---

Run all v0.1 training matrix experiments sequentially on single RTX 3090.

**Phase 1 - Architecture Sanity (3 experiments):**
- [x] P1-A1: One-shot projector, Mix B, End + KL (config ready)
- [x] P1-A2: Shared recurrent residual, Mix B, End + KL (config ready)
- [x] P1-A3: Flow midblock, Mix B, End + KL (config ready)

**Phase 2 - Loss Ablation (4 experiments):**
- [x] P2-L1: Flow, Mix B, End only (config ready)
- [x] P2-L2: Flow, Mix B, End + KL (config ready)
- [x] P2-L3: Flow, Mix B, End + Traj + KL (config ready)
- [x] P2-L4: Flow, Mix B, End + Traj + KL + CE (config ready)

**Phase 3 - Data Mix Ablation (3 experiments):**
- [x] P3-D1: Flow, Mix A, End + Traj + KL (config ready)
- [x] P3-D2: Flow, Mix B, End + Traj + KL (config ready)
- [x] P3-D3: Flow, Mix C, End + Traj + KL (config ready)

**Phase 4 - T Sweep (5 configs):**
- [x] P4-E1 through P4-E5: Eval at T=1,2,4,8,12 (configs ready)

## Summary of Changes

All 15 experiment configs are ready in `configs/v0_1_matrix/`.

**Files created for remote execution on 3090:**
- `remote_run/setup_and_run.sh` - Automated setup and sequential run script
- `remote_run/README.md` - Instructions for running on 3090 rental

**Usage on 3090 rental:**
1. Upload repository to 3090 instance
2. Run: `bash remote_run/setup_and_run.sh`

**The script will:**
- Install uv and create venv with Python 3.10
- Install PyTorch 2.2+ with CUDA 12.1
- Install all dependencies (transformers, datasets, etc.)
- Execute all 15 experiments sequentially on single 3090

**Expected runtime:** ~8-15 hours on single RTX 3090

**AGENTS.md updated:** Added "DO NOT BUILD THE PACKAGE" instruction per project requirements.

---
# midflowlm-hnnz
title: Rerun one-epoch corrected mixed-corpus midflow training with fresh optimizer
status: todo
type: task
priority: critical
tags:
    - training
    - experiment
    - rerun
created_at: 2026-03-26T15:45:40Z
updated_at: 2026-03-26T15:45:40Z
parent: midflowlm-g4k0
blocked_by:
    - midflowlm-gz99
    - midflowlm-3rlu
    - midflowlm-l2do
---

Launch a clean recovery experiment once the gradient-path and logging issues are fixed and the corrected graph is confirmed to fit in memory.

## Instructions
- Start from the current best midflow weights only if they are still useful as initialization, but do **not** reuse the old optimizer/scheduler state from the flawed run.
- Keep the experiment short and diagnostic: 1 epoch, faster but still sane LR (recommended starting point: `2e-4`, with `1e-4` as the conservative fallback).
- Preserve the same mixed-corpus setting unless the memory smoke test requires a debug-only reduction.
- Record the exact config diff, output directory, checkpoint path, and unique validation metrics across the run.

## Checklist
- [ ] Create the corrected rerun config
- [ ] Initialize from the chosen checkpoint or midflow weights with a fresh optimizer state
- [ ] Launch the 1-epoch rerun
- [ ] Record output paths and best checkpoint
- [ ] Record unique validation CE/KL/velocity trends

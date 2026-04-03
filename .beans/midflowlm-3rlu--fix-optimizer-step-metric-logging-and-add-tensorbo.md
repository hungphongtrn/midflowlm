---
# midflowlm-3rlu
title: Fix optimizer-step metric logging and add TensorBoard inspection script
status: in-progress
type: task
priority: high
tags:
    - logging
    - tensorboard
    - analysis
created_at: 2026-03-26T15:44:43Z
updated_at: 2026-03-28T08:32:36Z
parent: midflowlm-g4k0
---

Current trainer logging writes metrics on every microbatch while `global_step` only increments after gradient accumulation. This causes duplicated step IDs and makes CE/KL curves in TensorBoard look more erratic than the true optimizer-step trend.

## Instructions
- Update online-no-cache training logs so TensorBoard metrics are aligned to optimizer steps (or otherwise clearly distinguish microsteps vs optimizer steps).
- Add an analysis utility under `src/scripts/` that inspects `outputs/.../tensorboard` and, if needed, falls back to `train.log` parsing.
- The script should summarize CE, KL, velocity, LR, grad norm, and validation metrics aggregated by optimizer step, not raw microbatch rows.
- Record usage examples in the bean body or commit notes so future experiments can reuse the tool.

## Checklist
- [x] Fix trainer logging semantics for accumulated gradients
- [x] Add `src/scripts/` inspection script for TensorBoard/log aggregation
- [x] Verify the script works on `outputs/v0_online_no_cache_mixed_ce_kl`
- [ ] Report aggregated CE/KL/velocity trends from the current flawed run
- [ ] Document how to use the script for future runs

## Progress Notes
- Trainer dedup fix verified with `python3 -m pytest tests/test_online_no_cache_trainer.py -v` (14 passed).
- Dedup now gates console step logs, TensorBoard train logging, and validation/checkpoint triggering to optimizer-step boundaries.
- `python3 -m src.scripts.inspect_trainer_logs outputs/v0_online_no_cache_mixed_ce_kl/logs/train.log` successfully detects the historical duplicate validation/checkpoint events in the old run log, confirming the inspection utility works.

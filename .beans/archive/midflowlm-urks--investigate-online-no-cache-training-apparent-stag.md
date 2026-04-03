---
# midflowlm-urks
title: Investigate online-no-cache training apparent stagnation
status: completed
type: bug
priority: high
created_at: 2026-03-24T06:58:47Z
updated_at: 2026-03-24T07:01:52Z
---

Investigate whether `python3 scripts/train_online_no_cache.py --config configs/v0_online_no_cache_mixed_ce_kl.yaml` is actually stuck after logging "Starting full training...".

## Checklist
- [x] Inspect training startup code and log cadence
- [x] Trace first-step data path and possible blocking points
- [x] Determine whether observed warnings indicate true stagnation or silent progress
- [x] Document evidence and recommended next diagnostics

## Summary of Changes
- Verified the process is actively training rather than deadlocked by checking the live GPU process and confirming the TensorBoard event file kept growing while `train.log` stayed unchanged.
- Traced the silent period to logging configuration: `scripts/train_online_no_cache.py` only configures the `__main__` logger, while `src/training/online_no_cache_trainer.py` logs progress through its own module logger, so trainer INFO logs never reach console/file output.
- Confirmed the `t.std()` warning is benign for `batch_size: 1`; it only affects the logged `t_std` metric and does not halt optimization.

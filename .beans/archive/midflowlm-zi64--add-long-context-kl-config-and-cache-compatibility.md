---
# midflowlm-zi64
title: Add long-context KL config and cache compatibility validation
status: completed
type: task
priority: normal
created_at: 2026-03-21T09:46:19Z
updated_at: 2026-03-22T11:50:54Z
---

- [x] Add long-context KL config
- [x] Implement strict offline cache compatibility validation
- [x] Cover compatibility with tests and smoke the config

## Summary of Changes

- Added strict offline cache compatibility validation in `src/training/data.py` and routed the train-script offline dataloader path through it.
- Added focused compatibility and train-smoke coverage in `tests/test_teacher_cache.py` and `tests/test_train_smoke.py`.
- Created `configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml` with `seq_len=512`, extended training horizon, and `teacher_state.mode: offline_cache`.
- Verified the long-context smoke now fails early with a clear `Cache seq_len mismatch` error against the existing 128-token cache.

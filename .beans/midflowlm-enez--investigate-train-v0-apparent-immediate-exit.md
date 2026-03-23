---
# midflowlm-enez
title: Investigate train_v0 apparent immediate exit
status: completed
type: bug
priority: normal
created_at: 2026-03-18T17:29:22Z
updated_at: 2026-03-18T17:30:07Z
---

Investigate why `.venv/bin/python scripts/train_v0.py --config configs/v0_onemotif.yaml` appears to exit immediately.

## Checklist
- [ ] Reproduce command and capture logs
- [ ] Inspect config values controlling dataset/cache size
- [ ] Trace dataloader sizing and training loop behavior
- [x] Document root cause and recommended next action

## Summary of Changes

Reproduced `.venv/bin/python scripts/train_v0.py --config configs/v0_onemotif.yaml` and confirmed it does not crash or `sys.exit` early. The run finishes quickly because the cache under `cache/tinystories_qwen_boundary_states/` only contains 8 train samples and 8 val samples, which produces exactly 1 train batch and 1 val batch at `batch_size: 8`. The config advertises 20k/1k desired samples, but training consumes the already-built cache rather than regenerating it, so runtime is bounded by the tiny smoke cache currently on disk.

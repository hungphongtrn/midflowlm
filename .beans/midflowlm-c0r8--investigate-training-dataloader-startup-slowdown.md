---
# midflowlm-c0r8
title: Investigate training dataloader startup slowdown
status: completed
type: bug
priority: normal
created_at: 2026-03-17T16:47:53Z
updated_at: 2026-03-17T16:49:46Z
---

## Goal
Analyze why training is slow around the "Creating dataloaders..." stage despite low VRAM/RAM usage.

## Checklist
- [x] Trace dataloader creation path from training entrypoint
- [x] Identify expensive CPU, disk, or serialization work before first batch
- [x] Summarize likely root causes and concrete improvements

## Summary of Changes
- Traced startup from `scripts/train_v0.py` into `src/training/data.py` and confirmed the slowdown occurs during `CacheDataset` initialization, before any GPU-heavy training begins.
- Identified the main startup cost: `_build_sample_map()` loads every cache shard to count samples, even though cache build currently writes one shard per sample.
- Identified follow-on throughput issues: `__getitem__()` re-loads full shard files per sample, config ignores `persistent_workers`, and the one-shard-per-sample cache layout creates heavy file-open / serialization overhead.

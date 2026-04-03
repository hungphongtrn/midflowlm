---
# midflowlm-vzo0
title: Implement metadata-based cache sample map
status: completed
type: task
priority: normal
created_at: 2026-03-17T16:52:32Z
updated_at: 2026-03-17T16:57:07Z
---

## Goal
Eliminate dataloader startup slowdown caused by loading every shard during `CacheDataset._build_sample_map()`.

## Scope
- Only implement action 1 from the investigation: avoid full-shard reads during dataset initialization.
- Preserve existing loader behavior for current cache layout.
- Do not change broader cache format unless necessary for this fix.

## Checklist
- [x] Add a failing test covering metadata-based sample map construction
- [x] Implement sample map construction without loading shard payloads at init
- [x] Run targeted tests and report results

## Summary of Changes

- Added a targeted regression test proving `CacheDataset` initialization no longer loads shard payloads just to build `sample_map`.
- Updated `CacheDataset` to derive per-shard sample counts from metadata, with optional explicit `samples_per_shard` support and a metadata-only fallback based on total samples and shard count.
- Verified the new targeted test and the existing deterministic cache dataloader smoke test both pass.

---
# midflowlm-w54m
title: Fix mixed cache shard loading
status: completed
type: bug
priority: normal
created_at: 2026-03-18T01:51:50Z
updated_at: 2026-03-18T11:27:00Z
---

## Problem

Training fails with `FileNotFoundError: shard not found` because the cache directory contains mixed shard naming patterns (`of_9000` and `of_19000`).

## Tasks

- [x] Inspect the cache layout and identify the true shard mapping failure
- [x] Update cache dataset loading to resolve shards by actual file path instead of reconstructed names
- [x] Sanity-check the fix against the current cache layout
- [x] Summarize root cause and code changes\n- [ ] Re-run training or smoke test in a torch-enabled environment

## Summary of Changes

- Confirmed the cache layout uses two filename schemes: indices `0-8999` are stored as `*_of_9000`, while indices `9000-18999` are stored as `*_of_19000`.
- Updated `src/training/data.py` to sort shards by the actual shard index parsed from filenames rather than relying on lexicographic order.
- Changed sample loading to read the exact shard path from `self.shards[...]` instead of reconstructing filenames with `load_shard(...)`, which was causing misses like `shard_9621_of_9000`.
- Added a direct shard-path loader that reconstructs `trajectory_targets` the same way as the shared loader.
- Sanity-checked the ordering against the current cache directory: index `9621` now maps to `shard_9621_of_19000.safetensors`.

## Verification Notes

- I verified the filename-to-index mapping with `python3` against the current cache directory.
- I could not run the actual training command in this workspace because the available Python environment here does not have `torch` installed.

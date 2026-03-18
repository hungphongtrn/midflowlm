---
# midflowlm-hm5e
title: Enhance build_teacher_cache.py to auto-split train/val with configurable ratio
status: completed
type: feature
priority: high
created_at: 2026-03-17T17:14:59Z
updated_at: 2026-03-17T17:24:48Z
---

## Problem

The current  script only caches a single split at a time (train/val/test). When running training, both train and val loaders show the same number of batches (2500) because the val dataloader falls back to the root cache directory when val/ subdirectory doesn't exist.

## Proposed Solution

Add a new configuration setting  (default: 0.05 = 5%) to automatically split the cached data into train and val folders during cache generation.

## Requirements

- [x] Add val_split_ratio setting to config (default: 0.05)
- [x] When building cache, automatically split samples:
  - First N samples go to train/ folder
  - Last M samples go to val/ folder (where M = total * val_split_ratio)
- [x] Create separate metadata.json for each split
- [x] Rename shards to have correct numbering per split (e.g., shard_0000_of_1000 for val)
- [x] Support --split=all flag to build both splits in one command
- [x] Maintain backward compatibility with --split=train/val/test

## Config Example

```yaml
teacher_cache:
  enabled: true
  cache_dir: ./cache/tinystories_qwen_boundary_states
  store_logits: false
  val_split_ratio: 0.05  # 5% for validation, 95% for training
```

## Expected Directory Structure After Fix

```
cache/tinystories_qwen_boundary_states/
├── train/
│   ├── metadata.json     # num_samples: 19000
│   └── shard_0000_of_19000.safetensors
│   └── ...
└── val/
    ├── metadata.json     # num_samples: 1000
    └── shard_0000_of_1000.safetensors
    └── ...
```

## Benefits

1. Eliminates manual cache splitting
2. Ensures val dataloader reports correct batch count
3. Makes training workflow more robust
4. Follows standard ML data split conventions

## References

- Current issue: val loader shows 2500 batches (same as train) instead of 125 batches
- Quick fix script created: === Splitting cache into train/val folders ===
Cache dir: ./cache/tinystories_qwen_boundary_states
Val size: 1000 samples
Train size: 19000 samples

=== Moving first 1000 shards to val/ ===
=== Moving remaining 19000 shards to train/ ===

=== Renaming val shards (0-999 of 1000) ===
=== Renaming train shards (0-18999 of 19000) ===

=== Creating metadata.json for val ===
=== Creating metadata.json for train ===

=== Verifying split ===
Val shards: 0
Train shards: 10000

=== Done! Cache split complete ===
Train: ./cache/tinystories_qwen_boundary_states/train/ (19000 samples)
Val: ./cache/tinystories_qwen_boundary_states/val/ (1000 samples)



## Summary of Changes

### Files Modified

1. **configs/v0_onemotif.yaml**
   - Added `val_split_ratio: 0.05` to teacher_cache section
   - Default 5% validation split from training data

2. **scripts/build_teacher_cache.py**
   - Added `build_cache_for_split()` function to handle single split caching with proper directory resolution
   - Added `build_cache_with_split()` function to auto-split train/val during cache generation
   - Modified `build_cache()` to:
     - Extract `val_split_ratio` from config
     - Handle `split='all'` by calling `build_cache_with_split()`
     - Fall back to single-split behavior for train/val/test
   - Updated `verify_cache()` to accept optional `split` parameter for split-specific verification
   - Added `all` to `--split` argument choices
   - Updated main() to verify both splits when `--split=all` is used

3. **src/data/teacher_cache.py**
   - No changes required - `resolve_split_cache_dir()` already exists and works as expected

### Implementation Details

- When `--split=all`:
  1. Processes full train split once
  2. Calculates split sizes: val = total * val_split_ratio, train = total - val
  3. First N samples go to train/ folder (e.g., 19000 at 95%)
  4. Last M samples go to val/ folder (e.g., 1000 at 5%)
  5. Creates separate metadata.json for each split
  6. Shard numbering reflects per-split totals (e.g., shard_0000_of_1000 for val)

- Backward compatibility:
  - `--split=train`, `--split=val`, `--split=test` work exactly as before
  - Single-split caching still writes to split-specific subdirectories
  - Uses `resolve_split_cache_dir()` for consistent path resolution

### Usage Example

```bash
python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --split all
```

Creates:
```
cache/tinystories_qwen_boundary_states/
├── train/
│   ├── metadata.json     # num_samples: 19000
│   └── shard_0000_of_19000.safetensors
│   └── ...
└── val/
    ├── metadata.json     # num_samples: 1000
    └── shard_0000_of_1000.safetensors
    └── ...
```

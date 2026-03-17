---
# midflowlm-y5d2
title: Run MidflowLM v0 experiment and document failures
status: completed
type: task
priority: normal
tags:
    - midflowlm
    - experiment
created_at: 2026-03-17T04:00:59Z
updated_at: 2026-03-17T07:12:34Z
---

# MidflowLM v0 smoke run notes (2026-03-17)

## Scope
Attempted to create a bean for running the MidflowLM v0 experiment, execute a bounded smoke run, and document failure points.

## Bean
- Bean ID: `midflowlm-y5d2`
- Title: `Run MidflowLM v0 experiment and document failures`

## What was run

### 1) Install dependencies
```bash
./.venv/bin/pip install -r requirements.txt
```

### 2) Full cache build attempt with default config
```bash
./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml
```

Observed result:
- Did not complete within 30 minutes.
- Produced a partially built cache without `metadata.json`.
- Cache directory size grew to about `220G`.
- Existing shards reached roughly `shard_1805_of_20000.safetensors`.

### 3) Bounded smoke config created
Created `configs/v0_smoke_run.yaml` with:
- `train_samples: 8`
- `val_samples: 2`
- `test_samples: 2`
- `batch_size: 1`
- `num_workers: 0`
- `pin_memory: false`
- `precision: fp32`
- separate cache dir: `./cache/tinystories_qwen_boundary_states_smoke`

### 4) Smoke cache build
```bash
./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --limit 8 --overwrite
```

Observed result:
- Succeeded.
- Wrote 8 shards and `metadata.json`.

### 5) Smoke training run
```bash
./.venv/bin/python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run
```

Observed result:
- Failed during cache dataloader construction.

## Failure points

### Failure 1: Default cache build is operationally too large / slow
Symptoms:
- Default config cache build did not finish in 30 minutes.
- Partial cache consumed about `220G` before completion.
- No metadata was written, so downstream training cannot reliably use the partial cache.

Impact:
- The documented operational expectation in config comments (`~2-3GB`) is currently far from observed behavior.
- A full v0 run is blocked unless cache size/runtime are reduced or storage is provisioned appropriately.

Likely contributing causes:
- One shard per sample for `20,000` samples.
- Each shard stores hidden states across trajectory plus logits.
- No resumable metadata/checkpointing until the end of the run.

### Failure 2: Training loader cannot read `.safetensors` shards
Command:
```bash
./.venv/bin/python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run
```

Error:
```text
_pickle.UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`.
```

Root cause:
- `src/training/data.py` uses `torch.load(shard_path, map_location="cpu")` inside `_build_sample_map()` and `__getitem__()`.
- Cache writer stores shards as `.safetensors`.
- Training loader should use `safetensors.torch.load_file(...)` or the existing cache helper `load_shard(...)`, not `torch.load(...)` directly on `.safetensors`.

Relevant code:
- `src/training/data.py:76`
- `src/training/data.py:105`
- `src/data/teacher_cache.py:315-318`

### Failure 3: Split handling in cache dataset appears incomplete
Observation:
- `CacheDataset(split=...)` accepts a split, but currently loads all shards from a single cache directory and does not appear to partition train vs val vs test cache files.

Impact:
- Even after the loader bug is fixed, train/val loaders may read the same cache unless cache layout or split indexing is implemented.

Relevant code:
- `src/training/data.py:36-57`
- `src/training/data.py:59-88`

### Failure 4: Dataset API warning / compatibility risk
Observed during cache build:
```text
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'roneneldan/TinyStories' isn't based on a loading script and remove `trust_remote_code`.
```

Impact:
- Cache build still succeeded in the bounded smoke run, so this is not the immediate blocker.
- However, it indicates compatibility drift in dataset-loading code and may become a hard failure under other versions.

## Recommended next actions
1. Fix cache loading in `src/training/data.py` to use safetensors-aware loading or `load_shard()`.
2. Re-run smoke training:
   ```bash
   ./ .venv/bin/python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run
   ```
3. Decide whether full cache generation should:
   - store fewer fields,
   - shard by split,
   - batch multiple samples per shard,
   - write resumable metadata progressively,
   - or reduce default dataset size for v0.
4. Validate train/val split semantics before any longer training run.

## Exact smoke config path
- `configs/v0_smoke_run.yaml`

## Implementation Plan
See [[2026-03-17-midflowlm-v0-fixes]] for detailed execution plan.

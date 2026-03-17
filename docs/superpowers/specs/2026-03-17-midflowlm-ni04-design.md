# MidflowLM ni04 Design

## Goal

Implement the accepted cache-policy decision for architecture training:

- cache hidden states only by default
- stop depending on offline `teacher_logits` for architecture training
- keep behavior training out of scope for now
- leave a clear handoff for a later behavior-training task

## Scope

In scope:

- cache builder and cache metadata updates for hidden-state-only default behavior
- cache loader and training-data compatibility updates
- architecture-training path updates so it consumes hidden-state targets only
- config and documentation updates to reflect the new default workflow
- a follow-up bean for future behavior training with online teacher logits

Out of scope:

- implementing KL distillation, GRPO, or other behavior-training objectives
- storing full teacher logits offline by default
- broader trainer redesign beyond what is needed to preserve the architecture-training path

## Recommended approach

Use a narrow migration of the existing architecture-training path rather than introducing a second training mode now.

Why this approach:

- it directly matches `docs/decision_learning.md`
- it reduces cache size and operational risk immediately
- it avoids mixing a cache-format refactor with a larger training-objective change
- it preserves room for a later behavior-training implementation with explicit interfaces

## Architecture changes

### 1. Cache format

The default cache format should contain only the tensors required for architecture training:

- `input_ids`
- `attention_mask`
- `h_start`
- `trajectory_targets`
- `h_target`

The cache metadata should explicitly describe whether logits are stored. The default for this task should be false.

Physical shard layout should remain narrow and compatible with the current repository format:

- keep storing trajectory states on disk as `num_trajectory_targets` plus `trajectory_target_{i}` tensors
- continue reconstructing `trajectory_targets` in the loader API for consumers
- remove `teacher_logits` from default shards rather than redesigning the shard layout wholesale

This keeps the migration focused on cache contents, not on introducing a second on-disk format.

### 2. Training path

Architecture training should read only hidden-state supervision from cache. Any code path that currently assumes `teacher_logits` are present in every shard should be removed or made explicitly optional.

If a future behavior-training path needs logits, it should compute them online from the teacher model instead of relying on the default offline cache.

### 2a. Default loss contract

The default architecture-training loss contract must be updated so it no longer depends on offline logits.

- hidden-state losses remain the default supervision path
- any default `kl_weight` or other logit-dependent objective must be set to zero or otherwise disabled for the architecture-training configs used by this task
- any logit-dependent loss path must become explicit opt-in behavior rather than part of the default cache contract

This removes ambiguity with current configs that still enable KL by default and prevents the architecture path from silently requiring `teacher_logits`.

### 3. Interface boundary for future behavior training

This task should leave a documented seam rather than a full implementation. That seam should make it clear that:

- architecture training uses hidden-state cache
- behavior training may require online teacher forward passes
- any logit-dependent objective must opt in explicitly

### 4. Split handling

The default architecture-training workflow should continue to support split-specific cache builds rather than relying on implicit partitioning inside one cache directory.

- the config contract should keep a single root `teacher_cache.cache_dir` and derive split-specific subdirectories beneath it, for example `cache_dir/train`, `cache_dir/val`, and `cache_dir/test`
- cache build should write to the derived split subdirectory for the requested split
- training should consume the split-specific subdirectories explicitly for train and val loaders
- this task should not depend on deterministic repartitioning of a single shared cache directory

This matches the existing cache-build entry point, keeps split semantics unambiguous, and avoids coupling the hidden-state-only migration to a new partitioning scheme.

## Data flow

The default project workflow becomes:

1. build hidden-state-only teacher cache
2. train architecture model from cached hidden states
3. evaluate architecture-training outputs
4. defer behavior-training objectives to a later bean

This keeps the heavy offline preprocessing aligned with the cheapest reusable targets and prevents the cache from being dominated by vocabulary-sized logit tensors.

## Error handling and compatibility

- Existing caches with `store_logits: true` may remain readable, but they are not the default target of this migration.
- Rebuild is the required path for adopting the new default hidden-state-only architecture-training workflow.
- Cache readers should continue to load older shards with logits when encountered, but default configs, docs, and tests should move to the hidden-state-only cache contract.
- Training data loading must reuse or route through the shared shard-loading path in `src/data/teacher_cache.py` so `.safetensors` and reconstructed `trajectory_targets` are handled consistently.
- Any missing-logits error should be replaced with a targeted message that logits are not part of the default architecture-training cache.
- Config comments and docs should clearly distinguish architecture training from behavior training.

## Verification

Before the task is considered complete, verify:

- parity checks still pass for Qwen boundary extraction and bypass-wrapper reproduction before and after the cache-contract change
- cache build writes hidden-state-only shards and metadata by default
- cache loader can read the new shards successfully
- train/val/test cache paths and loader behavior are explicit and non-overlapping
- default architecture-training configs no longer require offline `teacher_logits`
- one batch forward pass and one optimizer step succeed on the architecture-training path
- checkpoint save/load still works
- frozen/trainable parameter counts remain sane
- docs/configs no longer describe full-logit caching as the default path

Required executable checks:

- `pytest tests/test_qwen_parity.py -v`
- `pytest tests/test_teacher_cache.py -v`
- `pytest tests/test_train_smoke.py -v`
- `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --split train --limit 8 --overwrite`
- `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --split val --limit 2 --overwrite`
- `./.venv/bin/python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run`

## Follow-up work

Create a separate bean for future behavior training covering online teacher logits and behavior-level objectives such as KL distillation.

## Notes

This design intentionally does not require a new custom trainer framework. It keeps the current raw PyTorch architecture-training path and narrows the cache contract to the project's accepted default.

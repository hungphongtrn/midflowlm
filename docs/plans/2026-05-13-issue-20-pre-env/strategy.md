# Issue 20: Pre-Env Inline Pipeline - Strategy

## Goal
Make `scripts/train_sft.py` fully self-contained so it can bootstrap on rented GPU instances from HuggingFace only — no local disk pre-processing required.

## Architecture
Config-driven pipeline with three new capabilities layered into the existing training script:

1. **Inline data preprocessing** — download, filter, tokenize, pack on-the-fly with disk cache
2. **Remote checkpoint download** — pull P3-D3 warm-start weights from HF Hub when missing locally
3. **Post-training HF push** — upload trained checkpoint + config + metadata to Phase 2 repo

All orchestration lives in `scripts/train_sft.py`. Business logic lives in `src/data/reasoning_sft.py`.

## Tech Stack
- `huggingface_hub` — `hf_hub_download()`, `upload_file()`, `create_repo()`
- `datasets` — `load_dataset()`, `.map()`, `.filter()`, `.save_to_disk()`, `load_from_disk()`
- TRL — `pack_dataset()` via existing `pack_tokenized_dataset()` wrapper
- Existing: `SFTFlowMidblockModel`, `MidblockSaveCallback`, `MidblockMetricsCallback`, `validate_model_for_training`, `estimate_training_budget`

## Constraints & Assumptions
- Only midblock trainable; Qwen frozen (unchanged from current behavior)
- Backward compatible: old `data.processed_dir` configs still work
- No ADR needed — all decisions are implementation choices within established patterns
- Public HF repos; auth is optional for reads, required for writes

## Phases (High-Level)

### Phase 1: Config + Data Pipeline
**Outcome:** Both configs updated, inline preprocessing code in `src/data/`, cache hashing, backward compat detection in `train_sft.py`. Can run data prep end-to-end from HF Hub.
**Rough scope:** Update YAML configs, add `create_reasoning_sft_datasets_from_config()`, implement cache hash + invalidation, thread `packing_strategy`, remove `prepare_reasoning_sft_data.py`, wire detection logic in script.

### Phase 2: Remote Checkpoint + Smoke Test
**Outcome:** `--smoke-test` flag works (full data prep, 1 training step), checkpoint auto-downloads from HF Hub when local missing.
**Rough scope:** Add `maybe_download_checkpoint()`, `--smoke-test` CLI arg, execution order (data → checkpoint → model), update config checkpoint fields.
**Depends on:** Phase 1

### Phase 3: HF Push + Integration
**Outcome:** Trained checkpoint auto-pushed to `hungphongtrn/midflowlm-phase2/issue-9/` after training. All acceptance criteria met.
**Rough scope:** Wire `training.push_to_hub: true`, auth chain, grace on missing token, `experiment_info.json`, end-to-end smoke-test on both configs.
**Depends on:** Phase 2

## Open Questions
None remaining — all resolved during grilling session (see decisions.md and issue comment).

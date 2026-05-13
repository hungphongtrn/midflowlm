# Decision Log

## 2026-05-13: Inline vs processed path detection
**Context:** Configs need to support both old `data.processed_dir` and new `data.dataset` fields.
**Decision:** Presence-based mutual exclusion. If `data.processed_dir` → load from disk. If `data.dataset` → inline prep. Error if both set.
**Rationale:** Clean detection, no new mode flag, impossible to misconfigure.

## 2026-05-13: Cache invalidation via config hash
**Context:** Reprocessing is expensive (~30-60 min for full dataset). Need to detect config changes.
**Decision:** Store `cache_info.json` with sha256 hash of all preprocessing params. Re-process if hash mismatches.
**Rationale:** Prevents silent staleness. Hashing all params (not a subset) is safer. Cost of unnecessary reprocessing is acceptable.

## 2026-05-13: Checkpoint download uses hf_hub_download
**Context:** Issue proposed raw HTTP URL. Existing codebase uses `huggingface_hub` API.
**Decision:** Use `hf_hub_download(repo_id, filename)` with separate `remote_repo` and `remote_filename` config fields.
**Rationale:** Consistent with existing `push_checkpoints_to_hf.py` patterns. Handles auth, caching, resume. No raw HTTP.

## 2026-05-13: Post-training push uses HF Trainer's push_to_hub
**Context:** Could do standalone push (only midblock weights) or use HF Trainer's built-in mechanism.
**Decision:** Use `training.push_to_hub: true` (Option C). Full safetensors uploaded. Midblock extractable downstream via key filtering.
**Rationale:** Simplest integration. HF Trainer already handles Hub lifecycle. Bandwidth cost of frozen weights is acceptable.

## 2026-05-13: --smoke-test is CLI-only, no config block
**Context:** Issue proposed `smoke_test.max_steps` config field.
**Decision:** Drop the config block. `--smoke-test` hardcodes `max_steps=1` in code.
**Rationale:** Smoke test is an operator toggle, not a configurable experiment dimension. YAGNI.

## 2026-05-13: Full dataset processing in smoke test
**Context:** Current smoke test subsets to 1000 samples. New design needs to validate data packing at real scale.
**Decision:** Process full dataset, cache it, run 1 training step. No sample limits.
**Rationale:** Validates the full pipeline including packing behavior driven by real data volume. Cache amortizes the cost.

## 2026-05-13: prepare_reasoning_sft_data.py removed
**Context:** Inline preprocessing makes the standalone script redundant.
**Decision:** Remove entirely.
**Rationale:** Single source of truth. No maintenance burden from deprecated code. Backward compat via `processed_dir` path still works.

## 2026-05-13: Auth chain for HF operations
**Context:** Both download and upload need auth for private/gated repos.
**Decision:** Single unified chain: `--hf-token` arg → `HF_TOKEN` env → `huggingface-cli login`. Warn but don't crash if missing.
**Rationale:** Simpler than split auth. `huggingface_hub` handles unauthenticated access gracefully for public repos. Token only strictly needed for writes.

## 2026-05-13: 3060 config gets own cache dir via hash
**Context:** 3060 config has `max_seq_length: 1024` vs full config's `8192`.
**Decision:** Let the cache hash produce different hashes, resulting in separate cache dirs. Meta-filter uses `max_total_tokens: 1024` for 3060.
**Rationale:** No special-casing needed. The hash mechanism handles it automatically.

## 2026-05-13: Code organization — business logic in src/data/
**Context:** Inline preprocessing adds significant logic to training script.
**Decision:** New function `create_reasoning_sft_datasets_from_config()` in `src/data/reasoning_sft.py`. Script calls it. Helper functions for checkpoint/push stay in-script.
**Rationale:** Data logic belongs in the data module. Script stays thin orchestration.

## 2026-05-13 (Phase 1 complete): Cache hash omits max_total_tokens — uses max_seq_length
**Context:** Phase 1 stub plan included `max_total_tokens` in the cache hash. The actual implementation uses `max_seq_length` only.
**Decision:** Keep it as implemented. `max_total_tokens` in `processing` config is informational only — `create_reasoning_sft_datasets_from_config()` passes `data_cfg["max_seq_length"]` as both the filter budget and the packing length.
**Rationale:** Both fields are set to the same value in both configs (8192 and 1024). The config field serves as human-readable documentation. No functional difference.

## 2026-05-13 (Phase 1 complete): Execution order will be fixed in Phase 2
**Context:** Phase 1 implementation left the original execution order intact (tokenizer → model → data). This is wrong — data should load before model so the tokenizer is available for both, and data loading doesn't depend on model state.
**Decision:** Reorder in Phase 2 (data → checkpoint → model → validate → move to device).
**Rationale:** Minimal diff per phase. Phase 1 focused purely on data pipeline correctness. Phase 2 addresses orchestration.

## 2026-05-13 (Phase 1 complete): sft_data_glm.yaml is orphaned
**Context:** `configs/issue-9/sft_data_glm.yaml` was used by the now-removed `scripts/prepare_reasoning_sft_data.py`.
**Decision:** Leave the file in place for reference but do not reference it from train_sft.py. It is harmless dead config.
**Rationale:** Removes unnecessary churn. It will fade out naturally.

## 2026-05-13 (Phase 2 complete): Execution order fixed — data before model
**Context:** Phase 1 left the original order (model before data). Phase 2 Task 4 reordered main().
**Decision:** New order: tokenizer → HF token → data → checkpoint download → model → validate → move to device.
**Rationale:** Tokenizer needed for data AND model. Data doesn't depend on model. Checkpoint download depends on resolved HF token, not on model. Order now reflects true dependencies.

## 2026-05-13 (Phase 3 complete): upload_file imported lazily after training
**Context:** `experiment_info.json` upload uses `huggingface_hub.upload_file()` which is only needed when push_to_hub is enabled.
**Decision:** Import `upload_file` inline in the upload block (not at module level like `hf_hub_download`).
**Rationale:** `hf_hub_download` is always needed (checkpoint download). `upload_file` is conditional. No separate import block means cleaner module-level imports. The cost of lazy import is negligible since it happens at train end.

## 2026-05-13 (Phase 3 complete): 30 tests total across all phases
**Context:** All three phases complete.
**Decision:** Distribution: 17 data pipeline tests (Phase 1), 9 checkpoint download tests (Phase 2), 4 experiment info tests (Phase 3).
**Rationale:** Data pipeline has the most edge cases (filtering, caching, hashing, backward compat). Checkpoint download tests cover auth chain and local/remote paths. Experiment info tests verify key contract (all keys present, defaults, UTC timestamp).

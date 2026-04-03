# Decision and Learning Log

This document captures durable decisions and learnings made through discussion during MidflowLM development. Agents should append or update this file when the user and agent converge on a design decision, workflow rule, or important lesson that should shape future implementation.

## 2026-03-17 — Caching and training split

Status: accepted

Context:
During discussion about v0 experiment execution, we observed that storing full teacher logits in the offline cache makes cache generation extremely large. Since the teacher model is not prohibitively large, we decided to reserve offline caching for architecture targets and recalculate logits when needed for behavior training.

Decisions and learnings:

1. There are two types of training in this project:
   - architecture training
   - behavior training

2. Architecture training:
   - Goal: make the midflow modules able to iterate similarly to hidden-state refinement.
   - Primary supervision should come from cached hidden-state targets.
   - Offline cache should support this training mode.

3. Behavior training:
   - Goal: improve model behavior rather than only internal hidden-state refinement.
   - Candidate methods include KL distillation, GRPO, and related behavioral objectives.
   - These objectives may use recalculated teacher logits or other online behavioral signals instead of storing full logits offline.

4. Cache policy decision:
   - Recalculate logits when needed.
   - Only cache hidden states for architecture training.
   - Do not store full teacher logits in the default offline cache for architecture training.

Implications:
- The cache builder and cache format should be revised to support hidden-state-only caching by default.
- Training code should distinguish architecture-training data requirements from behavior-training data requirements.
- Any future decision to cache logits should require explicit justification because of the storage cost.

## 2026-03-19 — Decoding is the first repetition triage step

Status: accepted

Context:
After a fresh qualitative sweep on `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt`, the main remaining failure mode was severe short-prompt repetition on the robot prompt. A follow-up experiment updated the text sweep tooling to support non-greedy decoding and tested the same checkpoint across modest decoding settings.

Decisions and learnings:

1. Repetition observed in the current qualitative sweep should not be treated as a pure training failure until decoding has been checked first.
   - Pure greedy decoding can trap the current student in repetition loops.
   - A lightweight decoding sweep is now the required first validation step before escalating to training-objective changes.

2. For the current v0 checkpoint family, `num_steps = 4` remains the default qualitative sampling setting.
   - This is still the safest qualitative step count.
   - Larger out-of-range step counts are not a valid quality-improvement path in the current design.

3. The current recommended quick qualitative decoding setting is temperature sampling instead of greedy decoding.
   - Preferred default: `temperature = 0.7`.
   - Greedy decoding is still useful as a stress test, but not as the only qualitative readout.

4. Evaluation should explicitly distinguish token-level n-gram repetition metrics from sentence- or phrase-level looping.
   - Word n-gram ratios may miss obvious repeated-sentence failures.
   - Manual inspection or stronger phrase-level repetition checks are still required when assessing short-prompt behavior.

Implications:
- Before proposing behavior-training changes for repetition, run a modest decoding sweep on the current checkpoint first.
- Qualitative handoff and manual sampling should default to `final.ckpt`, `num_steps = 4`, and non-greedy decoding unless a stress test specifically calls for greedy.
- Future eval tooling should preserve both automatic repetition metrics and human-readable qualitative outputs because the current failure mode is partly decoding-sensitive.

## 2026-03-19 — Mixed-Corpus Experiment Decision

Status: accepted

Context:
MMLU-Pro evaluation on the TinyStories-trained student showed 0% accuracy with the student emitting `<think>`/`思考`/`</think>` as first tokens, indicating the student has not learned to answer multiple-choice questions from the TinyStories-only training. We are testing the hypothesis that broader training data (including MCQ datasets) will improve generalization.

Decisions and learnings:

1. Keep hidden-state-only loss unchanged for this experiment.
   - Do not add CE/KL terms alongside broader data.
   - Isolate the data-coverage variable from the loss-function variable.

2. Use FineWeb-Edu + UltraChat + MCQ datasets for the first mixed run:
   - HuggingFaceFW/fineweb-edu (sample-10BT): plain text continuation
   - HuggingFaceH4/ultrachat_200k (train_sft/test_sft): chat messages
   - allenai/ai2_arc (ARC-Challenge, ARC-Easy): multiple choice
   - tau/commonsense_qa: multiple choice
   - allenai/openbookqa (main): multiple choice

3. Exclude explicit CoT/rationale data from the first mixed run.
   - QA examples are formatted as direct-answer sequences with answer letters included in the cached text.
   - This keeps the formatter surface small for the first experiment.

4. Use UltraChat tokenizer chat template when available so the teacher sees the same chat tokenization family used at eval time.

5. Cache generation remains split-based: build train and val caches explicitly with `scripts/build_teacher_cache.py --split train|val`.

Implications:
- A negative result (student still emits `<think>`/`思考` first tokens) is still informative - it would suggest the data hypothesis alone is insufficient without loss function changes.
- A positive result (improved first-token behavior on MCQ) would support the data-coverage hypothesis.

Smoke-test commands:
```bash
python -m pytest tests/test_mixed_corpus_data.py tests/test_teacher_cache.py -v
python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 16 --split train --overwrite --verify
python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 8 --split val --overwrite --verify
python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --fast-dev-run
```

---

## Decision: Runtime Policy via teacher_state.mode (2026-03-22)

**Context**: The original implementation used a logits-only concept (`teacher_logits_source`) that only controlled whether KL loss was computed, not the actual source of teacher states. Runtime policy is now centralized through `teacher_state.mode`.

**Decision**: Three operational modes are supported:

1. **`offline_cache`**: Uses pre-built teacher cache. Cache compatibility is validated before training starts (model identity, layer span, seq_len, store_logits when kl_weight > 0). This is the default and recommended mode for production training.

2. **`online_no_cache`**: Uses live teacher extraction via `QwenInspector` on each training step. Token batches come from the corpus dataloader (not cache). Suitable for debugging or when cache regeneration is expensive.

3. **`online_write_through_cache`**: Same live extraction as `online_no_cache`, but also persists teacher states to cache using the same shard format as offline cache. Useful for bootstrapping a new cache or incrementally extending an existing one.

**Cache compatibility validation** (`src/training/data.py::validate_cache_compatibility`) enforces:
- `model_name` match
- `model_revision` match
- `start_layer` / `end_layer` match
- `span_depth` match
- `seq_len` match
- `store_logits=True` when `kl_weight > 0`

**Live extraction** reuses `QwenInspector` from `src/model/qwen_parity.py`. Extracted outputs are:
- `h_start`: hidden state before replacement span
- `h_target`: hidden state after replacement span
- `velocity_target = h_target - h_start`
- `teacher_logits`: final logits (if `store_logits=True` in config)

**Implementation** (`src/training/trainer.py`):
- `train_step` and `val_step` both call `_maybe_extract_teacher_states()` for online modes
- `QwenInspector` is initialized lazily on first extraction (not at Trainer construction time)
- Write-through cache writes use `TeacherCacheWriter.write_shard()` with compatible format

**Smoke commands**:
```bash
# Offline mode (requires compatible cache)
python scripts/train_v0.py --config configs/v0_mixed_corpus_plus_kl_loss.yaml --limit-train-batches 1 --limit-val-batches 1

# Online no-cache (live extraction, no persistence)
python scripts/train_v0.py --config configs/v0_teacher_state_online_no_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1

# Write-through (live extraction + cache persistence)
python scripts/train_v0.py --config configs/v0_teacher_state_write_through_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1
```

**Open question**: Whether write-through cache should validate existing shards before overwriting (idempotency vs fresh regeneration).

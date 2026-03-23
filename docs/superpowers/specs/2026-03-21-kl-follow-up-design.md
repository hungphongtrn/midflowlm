# KL Follow-up Inference, Long-Context Training, and Teacher-State Sourcing Design

## Goal

Define the next follow-up work after the failed KL-loss checkpoint audit by:

- rerunning downstream inference with larger token budgets and an explicit closed-think variant
- preparing a new long-context KL training configuration
- refactoring teacher-state sourcing so training can operate in cache-backed and live-compute modes depending on disk and compute constraints

## Scope

In scope:

- add two concrete downstream inference probe variants for the latest KL checkpoint
- introduce a new training config for larger context length and longer training duration
- make teacher-state sourcing explicit and configurable across offline-cache and online-compute paths
- preserve the existing mixed-corpus training path where possible
- keep reuse centered on the current Qwen-based student/teacher stack

Out of scope:

- changing the student architecture itself
- introducing a new trainer framework
- redesigning the loss formulation beyond what is needed to support teacher-state sourcing
- running a full production training job as part of the refactor task itself

## Approaches considered

### 1. Staged split by workstream (recommended)

Treat the work as three coordinated but mostly independent tasks:

- inference probing
- long-context config creation
- teacher-state sourcing refactor

Why this is recommended:

- the inference rerun gives immediate evidence about decoding and prompt behavior
- the config task is straightforward once cache implications are explicit
- the teacher-state refactor is the only architectural change and benefits from isolated review

### 2. Minimal experiment-only patch

Only rerun inference and add a larger-context config, leaving the cache pipeline unchanged.

Trade-off:

- fastest to execute
- does not address the disk pressure or the mismatch between online-KL logits and offline hidden-state caching

### 3. Teacher-state abstraction first

Refactor all teacher-state sourcing before doing any new experiments.

Trade-off:

- cleanest long-term abstraction
- slowest route to fresh evidence and highest immediate implementation risk

## Recommended approach

Use the staged split.

The implementation should first produce better evidence from inference, then define the larger-context experiment, then land the teacher-state refactor that makes future KL runs operationally flexible. This keeps the follow-up measurable while avoiding another training run on top of an unclear data path.

## Architecture

### 1. Inference probe variants

Current behavior tooling already supports long generation and a stripped-think mode in `scripts/infer_mmlu_pro_behavior.py`, but the requested second case is different: it should preserve an assistant-side prefill while explicitly closing the thought block before generation.

For this design, `closed-think prefill` means the final assistant prompt prefix ends with the literal text:

```text
<think>

</think>
```

followed immediately by normal generation. This is distinct from `--strip-thinking-prefill`, which removes the think prefill entirely.

Add prompt behavior controls that distinguish among:

- default assistant prefill behavior
- stripped prefill behavior
- explicit closed-think prefill behavior using the literal prefix `"<think>\n\n</think>"`

The CLI or helper API should expose the closed-think variant explicitly rather than overloading the existing stripped-prefill flag.

For the immediate follow-up experiment, run exactly two cases against the latest KL checkpoint:

- `max_new_tokens=512` with existing prompt behavior
- `max_new_tokens=512` with explicit closed-think prefill

Artifacts should be saved separately so the comparison is direct and reproducible.

### 2. Long-context training config

Create a new config derived from `configs/v0_mixed_corpus_plus_kl_loss.yaml` that increases `data.seq_len` materially above `128` and extends total training duration.

The config should remain conservative in structure:

- reuse the same model family and span definition
- keep the same mixed-corpus dataset recipe unless a later experiment changes data composition intentionally
- increase sequence length and training horizon without introducing unrelated optimization changes in the same config

This config is intended to answer whether the current short-context regime is contributing to downstream failure modes, especially verbose reasoning that never resolves into a compact answer.

### 3. Cache regeneration rule

If hidden-state caching remains part of the training path, cache regeneration is required for the long-context config.

Reason:

- cached metadata and payloads are sequence-length specific
- `CacheDataset` loads hidden-state tensors whose shapes must match the active training config
- reusing a `seq_len=128` cache for a materially larger `seq_len` run would violate the cache contract

Therefore the design should treat cache compatibility as a function of at least:

- model identity / revision
- replacement span (`start_layer`, `end_layer`, `span_depth`)
- sequence length
- stored payload type

### 4. Teacher-state sourcing modes

The current code mixes two ideas:

- hidden-state supervision comes from offline cache
- KL logits may come from cache or from an online teacher forward pass

That is not enough for current operational needs. The training system should expose a single teacher-state sourcing mode that determines where both hidden-state targets and logits come from.

Support three modes:

- `offline_cache`: read required teacher states from disk cache; requires a compatible cache to exist
- `online_no_cache`: compute required teacher hidden states and logits live per batch; writes nothing to disk
- `online_write_through_cache`: compute teacher states live and optionally persist cache artifacts for reuse

This should replace the narrower logits-only mental model around `teacher_logits_source`.

## Runtime dispatch points

Mode selection affects both data loading and trainer behavior, so the dispatch points must be explicit.

### 1. `scripts/train_v0.py`

This script should become the top-level router for teacher-state mode.

- `offline_cache` should continue to build dataloaders from `create_cache_dataloader`
- `online_no_cache` and `online_write_through_cache` should use a token dataset path that reads from the configured corpora directly instead of `CacheDataset`
- teacher model construction should be driven by mode, not only by logits-specific KL settings

This is the critical place where the current unconditional cache-dataloader assumption must be removed.

### 2. Trainer/runtime layer

The trainer should receive already-mode-appropriate batches and should compute or consume teacher states according to mode.

- `offline_cache` consumes cached hidden states and optional logits
- `online_no_cache` computes teacher boundary states and logits live per batch
- `online_write_through_cache` computes teacher states live and may pass them to a cache writer after or alongside the training step

Loss weighting remains in the loss config, but storage/sourcing policy should no longer be encoded as a logits-only loss option.

### 5. Data-flow implications per mode

#### `offline_cache`

- dataloaders continue to use `CacheDataset`
- trainer consumes cached `h_start`, `velocity_target`, optional `teacher_logits`
- compatibility checks validate cache metadata against config

#### `online_no_cache`

- dataloaders should produce tokenized text batches directly from the configured corpora rather than cached teacher-state samples
- trainer runs the frozen teacher to extract boundary hidden states and logits needed for the current loss
- no teacher-state payload is written to disk

This mode therefore requires a new non-cache dataloader path and is not achievable by simply disabling cache writes in the current implementation.

#### `online_write_through_cache`

- trainer computes the same live teacher states as `online_no_cache`
- a cache writer path may persist selected hidden-state targets and optional logits using current config metadata
- persistence should be optional and non-blocking relative to the correctness of the live training step

### 6. Boundary between config and runtime policy

The configuration should separate:

- whether teacher cache infrastructure is enabled at all
- which teacher-state source mode is active
- whether logits are persisted when caching is used

This avoids overloading `teacher_cache.enabled` and `loss.teacher_logits_source` with overlapping meanings.

One acceptable shape is:

- `teacher_state.mode: offline_cache | online_no_cache | online_write_through_cache`
- `teacher_cache.*` for cache location and persistence knobs
- loss config for weights only, not storage policy

Exact naming can follow repository conventions, but the responsibilities should remain distinct.

## Work decomposition

### Task A: Inference rerun

Deliverables:

- add closed-think prefill support to the relevant behavior inference path using the literal assistant prefix `"<think>\n\n</think>"`
- run two `512`-token inference cases on the latest KL checkpoint
- save outputs and summarize differences in answer extraction, completion length, and whether thinking loops persist

### Task B: Long-context config

Deliverables:

- add a new KL config in `configs/`
- document whether cache regeneration is required and why
- define a smoke command for the new config before any long run

### Task C: Teacher-state sourcing refactor

Deliverables:

- introduce explicit mode handling for `offline_cache`, `online_no_cache`, and `online_write_through_cache`
- keep existing cache-backed training working
- add validation errors for incompatible or incomplete mode/config combinations
- provide at least one smoke-tested path per supported mode where feasible

## Verification

Before implementation is considered complete, verify with fresh evidence:

- the behavior inference script can run both requested `512`-token variants against the KL checkpoint
- outputs for the two probe variants are saved to distinct artifacts
- the new long-context config parses and can run at least a smoke invocation
- cache compatibility checks fail clearly when sequence length does not match cache metadata
- `offline_cache` still works on an existing compatible cache-backed config
- `online_no_cache` can execute a one-batch forward and optimizer step without requiring a prebuilt cache
- `online_write_through_cache` can execute a one-batch step and emit compatible cache artifacts when enabled
- live teacher hidden-state extraction matches the existing offline-cache boundary targets on a shared sample batch within an explicit tolerance before any new long training run
- if logits are compared across modes, live teacher logits and cached teacher logits also match on a shared sample batch within tolerance when both are available

Suggested verification commands:

- `./.venv/bin/python scripts/infer_mmlu_pro_behavior.py --config configs/v0_mixed_corpus_plus_kl_loss.yaml --checkpoint outputs/18-25-20-03-2026-v0_qwen_mixed_corpus_midblock_plus_kl_loss/checkpoints/final.ckpt --num-samples 8 --num-steps 4 8 32 --max-new-tokens 512 --output <artifact>`
- `./.venv/bin/python scripts/train_v0.py --config <new-long-context-config> --limit-train-batches 1 --limit-val-batches 1`
- mode-specific smoke commands for each teacher-state source path
- a parity test command or script that compares `offline_cache` hidden states against live teacher extraction for the same token batch

## Risks and guardrails

- `online_no_cache` is broader than toggling KL logits online; it requires live teacher extraction for hidden-state supervision too
- longer context increases memory pressure and may require revisiting batch size or accumulation settings
- `online_write_through_cache` can create subtle correctness problems if persisted artifacts do not match the active config metadata; validation must be strict
- existing repository guidance still applies: define parity checks before new student training and ensure any cache-backed training has compatible cache artifacts before launch

## Notes

This design intentionally keeps the immediate inference rerun separate from the training-system refactor. The first task answers whether prompt and decoding controls recover any usable downstream behavior. The latter two tasks prepare the next experiment so that context length and teacher-state storage policy become explicit, testable choices rather than ad hoc operational decisions.

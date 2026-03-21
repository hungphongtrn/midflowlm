# KL Follow-up Progressive Disclosure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun KL-checkpoint inference with the two requested 512-token prompt variants, add a long-context KL config, and refactor teacher-state sourcing so training can switch cleanly between offline-cache and live-teacher modes.

**Architecture:** Keep the work split into three small execution packets that can be loaded independently: inference probing, long-context config plus cache contract, and teacher-state runtime refactor. Reuse the existing Qwen-based prompt/eval helpers, `QwenInspector`, cache metadata contract, and raw PyTorch trainer rather than introducing new trainer abstractions.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Hugging Face Datasets, PyYAML, pytest, safetensors

---

## How to use this plan

- Treat this document as a hub, not a giant all-at-once checklist.
- Only load the next task packet when its entry criteria are satisfied.
- Keep commits and verification local to each packet so later packets do not need the full prior context window.
- Start each execution session with `beans prime`, then update bean `midflowlm-ml5g` or its child beans as steps complete.

## Progressive disclosure sequence

1. **Packet A: inference evidence first** - smallest, fastest, and gives immediate signal.
2. **Packet B: long-context config and cache contract** - defines the next experiment without changing runtime behavior yet.
3. **Packet C: teacher-state sourcing refactor** - largest packet; only open once A and B are done.

## File map

**Create**
- `docs/superpowers/plans/2026-03-21-kl-follow-up-progressive-disclosure.md` - this execution plan
- `configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml` - longer-context KL experiment derived from the current KL config
- `configs/v0_teacher_state_online_no_cache_smoke.yaml` - one-batch smoke config for live teacher extraction without cache reads
- `configs/v0_teacher_state_write_through_cache_smoke.yaml` - one-batch smoke config for live teacher extraction with optional cache writes
- `tests/test_teacher_state_modes.py` - focused tests for mode resolution, validation, and router behavior
- `tests/test_teacher_state_parity.py` - parity checks between cached teacher targets and live teacher extraction
- `src/training/teacher_state.py` - teacher-state mode parsing, validation, compatibility checks, and shared runtime helpers

**Modify**
- `src/eval/mmlu_pro_behavior.py` - explicit prompt behavior enum/flag support for default, stripped, and closed-think variants
- `scripts/infer_mmlu_pro_behavior.py` - expose the closed-think probe in the CLI and artifact naming
- `scripts/run_free_inference_mmlu_pro.py` - keep the free-form inference CLI aligned with the shared prompt helper after the API change
- `configs/v0_mixed_corpus_plus_kl_loss.yaml` - add the new `teacher_state` section once runtime policy moves out of `loss.teacher_logits_source`
- `scripts/train_v0.py` - top-level mode router, config validation, and dataloader selection
- `src/training/data.py` - preserve cache dataloaders and add strict cache compatibility checks used by offline mode
- `src/training/trainer.py` - consume per-mode teacher targets and support live teacher extraction and optional write-through persistence
- `src/data/teacher_cache.py` - extend metadata/validation helpers and support write-through persistence hooks without breaking offline cache readers
- `tests/test_mmlu_pro_behavior.py` - prompt-mode and transcript-contract coverage
- `tests/test_train_smoke.py` - mode-aware trainer and train-script smoke coverage
- `tests/test_teacher_cache.py` - metadata compatibility and write-through-cache coverage
- `docs/decision_learning.md` - record the new teacher-state mode decision once implementation lands
- `docs/state.md` - record experiment outcomes after Packet A and Packet B complete

**Avoid unless Packet C proves it is required**
- `src/training/losses.py` - keep loss weighting semantics intact; only touch if teacher-batch keys must be normalized
- `src/model/qwen_parity.py` - reuse `QwenInspector` instead of reimplementing teacher extraction logic
- `src/data/dataset_factory.py` / `src/data/mixed_corpus.py` - prefer reusing the existing token dataset path for online modes instead of building a new corpus stack

---

## Fixed design decisions

1. `closed-think prefill` means the final assistant prefix literally ends with `"<think>\n\n</think>"`; it is a new explicit mode, not a side effect of `--strip-thinking-prefill`.
2. The two required follow-up probes are the latest KL checkpoint with `max_new_tokens=512` under:
   - existing default prompt behavior
   - explicit closed-think prefill
3. The long-context experiment stays on the existing mixed-corpus KL recipe and only changes context-length / training-horizon knobs plus the new teacher-state config fields.
4. Cache compatibility must at minimum validate model identity, model revision, replacement span, sequence length, and payload shape/logit availability before offline training starts.
5. Runtime policy moves out of the logits-only concept. The source of truth becomes `teacher_state.mode`, with `loss` retaining weights only.
6. Supported runtime modes are exactly `offline_cache`, `online_no_cache`, and `online_write_through_cache`.
7. Live teacher extraction must reuse existing Qwen inspection code and must have a parity test against offline cache before any long training run.

---

## Packet A entry criteria

- The target checkpoint path is known and readable.
- No training-path refactor is needed to run the probe.

### Task 1: Add explicit prompt-behavior controls for inference probing

**Files:**
- Modify: `src/eval/mmlu_pro_behavior.py`
- Modify: `scripts/infer_mmlu_pro_behavior.py`
- Modify: `scripts/run_free_inference_mmlu_pro.py`
- Test: `tests/test_mmlu_pro_behavior.py`

- [ ] **Step 1: Write prompt-mode tests first**

Add assertions that `create_mmlu_pro_prompt(...)` can produce three distinct assistant-prefill behaviors:

```python
def test_create_mmlu_pro_prompt_supports_closed_think_prefill():
    prompt = create_mmlu_pro_prompt(..., prompt_behavior="closed_think")
    assert "<|im_start|>assistant\n<think>\n\n</think>" in prompt


def test_create_mmlu_pro_prompt_supports_stripped_prefill():
    prompt = create_mmlu_pro_prompt(..., prompt_behavior="stripped")
    assert "<think>" not in prompt
```

- [ ] **Step 2: Run the prompt-mode tests to see them fail**

Run: `./.venv/bin/python -m pytest tests/test_mmlu_pro_behavior.py -k prompt -v`
Expected: FAIL because the helper only accepts `strip_thinking_prefill` today.

- [ ] **Step 3: Replace the boolean prefill switch with an explicit mode**

Implement a small shared API in `src/eval/mmlu_pro_behavior.py` such as:

```python
PromptBehavior = Literal["default", "stripped", "closed_think"]
```

and make `create_mmlu_pro_prompt(...)` branch on that value while preserving current default behavior.

- [ ] **Step 4: Expose the explicit mode in both CLIs**

Use one new CLI flag instead of stacked booleans, for example:

```text
--prompt-behavior default|stripped|closed_think
```

Keep backward compatibility only if it is trivial; otherwise prefer a clean replacement because these scripts are still internal experiment tools.

- [ ] **Step 5: Re-run the prompt tests**

Run: `./.venv/bin/python -m pytest tests/test_mmlu_pro_behavior.py -k prompt -v`
Expected: PASS.

- [ ] **Step 6: Commit Packet A code slice**

```bash
git add src/eval/mmlu_pro_behavior.py scripts/infer_mmlu_pro_behavior.py scripts/run_free_inference_mmlu_pro.py tests/test_mmlu_pro_behavior.py
git commit -m "feat: add explicit MMLU prompt behavior modes"
```

### Task 2: Run and save the two required 512-token probes

**Files:**
- Modify: `docs/state.md`
- Output: `results/kl_followup_default_512.jsonl`
- Output: `results/kl_followup_closed_think_512.jsonl`
- Output: `results/kl_followup_probe_notes.md`

- [ ] **Step 1: Run the default prompt probe**

Run:

```bash
./.venv/bin/python scripts/infer_mmlu_pro_behavior.py \
  --config configs/v0_mixed_corpus_plus_kl_loss.yaml \
  --checkpoint outputs/18-25-20-03-2026-v0_qwen_mixed_corpus_midblock_plus_kl_loss/checkpoints/final.ckpt \
  --num-samples 8 \
  --num-steps 4 8 32 \
  --max-new-tokens 512 \
  --prompt-behavior default \
  --output results/kl_followup_default_512.jsonl
```

Expected: command succeeds and writes the default-variant artifact.

- [ ] **Step 2: Run the closed-think probe**

Run the same command with `--prompt-behavior closed_think` and output `results/kl_followup_closed_think_512.jsonl`.

- [ ] **Step 3: Summarize the evidence**

Capture only the facts needed for downstream planning in `results/kl_followup_probe_notes.md`:
- answer extraction hit rate
- median / max completion length
- whether looping `<think>` behavior persists
- notable first-token shifts

- [ ] **Step 4: Record the result in `docs/state.md`**

Add a short dated note with artifact paths and the one-paragraph takeaway.

- [ ] **Step 5: Commit Packet A artifacts and notes**

```bash
git add results/kl_followup_default_512.jsonl results/kl_followup_closed_think_512.jsonl results/kl_followup_probe_notes.md docs/state.md
git commit -m "docs: record KL follow-up inference probes"
```

### Packet A exit criteria

- Both `512`-token variants run successfully.
- The outputs live in separate artifacts.
- The comparison note states whether closed-think meaningfully changes answer extraction or loop behavior.

---

## Packet B entry criteria

- Packet A is complete.
- The next training experiment still targets the current mixed-corpus KL stack.

### Task 3: Add the long-context config and make cache incompatibility explicit

**Files:**
- Create: `configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml`
- Modify: `src/training/data.py`
- Modify: `tests/test_teacher_cache.py`
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Write failing cache-compatibility tests**

Add focused tests for a helper in `src/training/data.py` that rejects incompatible offline caches, for example:

```python
def test_cache_compatibility_rejects_seq_len_mismatch():
    with pytest.raises(ValueError, match="seq_len"):
        validate_cache_compatibility(config, metadata)
```

Also add one positive-path test for a compatible cache.

- [ ] **Step 2: Run the new compatibility tests to confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_teacher_cache.py -k compatibility -v`
Expected: FAIL because no compatibility validator exists yet.

- [ ] **Step 3: Add the long-context config**

Create `configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml` by copying `configs/v0_mixed_corpus_plus_kl_loss.yaml` and changing only the experiment knobs that matter here:
- increase `data.seq_len` materially above `128` (use one explicit value, e.g. `512`)
- extend the training horizon (`max_epochs` and/or validation interval as needed)
- add the new `teacher_state.mode: offline_cache` section in the shape that Packet C will use
- keep model family, span, and corpus recipe unchanged

- [ ] **Step 4: Implement the cache compatibility validator**

Add a shared helper that validates offline cache metadata against the active config before dataloaders are created. At minimum compare:
- `model_name`
- `model_revision`
- `start_layer`
- `end_layer`
- `span_depth`
- `seq_len`
- `store_logits` if KL weights require logits from cache

- [ ] **Step 5: Run the cache-focused tests again**

Run:
- `./.venv/bin/python -m pytest tests/test_teacher_cache.py -k compatibility -v`
- `./.venv/bin/python -m pytest tests/test_train_smoke.py -k cache -v`

Expected: PASS.

- [ ] **Step 6: Smoke the new long-context config in offline mode**

Run:

```bash
./.venv/bin/python scripts/train_v0.py \
  --config configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml \
  --limit-train-batches 1 \
  --limit-val-batches 1
```

Expected: either one-batch success on a compatible regenerated cache, or a clear early failure describing the exact cache mismatch.

- [ ] **Step 7: Commit Packet B**

```bash
git add configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml src/training/data.py tests/test_teacher_cache.py tests/test_train_smoke.py
git commit -m "feat: add long-context KL config and cache contract"
```

### Packet B exit criteria

- The long-context config exists and parses.
- The repo has a strict offline-cache compatibility error instead of silent mismatch risk.
- The plan now states explicitly that long-context cache regeneration is mandatory.

---

## Packet C entry criteria

- Packets A and B are complete.
- The team still wants operational flexibility across cache-backed and live-teacher training.

### Task 4: Introduce the teacher-state mode contract before changing runtime behavior

**Files:**
- Create: `src/training/teacher_state.py`
- Create: `configs/v0_teacher_state_online_no_cache_smoke.yaml`
- Create: `configs/v0_teacher_state_write_through_cache_smoke.yaml`
- Create: `tests/test_teacher_state_modes.py`
- Modify: `configs/v0_mixed_corpus_plus_kl_loss.yaml`
- Modify: `scripts/train_v0.py`

- [ ] **Step 1: Write failing mode-resolution tests**

Add tests for a small config/runtime contract such as:

```python
def test_teacher_state_mode_defaults_to_offline_cache_for_existing_configs():
    resolved = resolve_teacher_state_config(config)
    assert resolved.mode == "offline_cache"


def test_online_no_cache_requires_teacher_model():
    with pytest.raises(ValueError, match="teacher model"):
        validate_teacher_state_config(config)
```

- [ ] **Step 2: Run the mode tests to confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_teacher_state_modes.py -v`
Expected: FAIL because the new contract module does not exist.

- [ ] **Step 3: Implement the shared teacher-state config helper**

In `src/training/teacher_state.py`, centralize:
- mode parsing
- validation of incomplete combinations
- offline-cache metadata checks
- per-mode booleans like `requires_cache`, `requires_live_teacher`, and `allow_cache_write`

- [ ] **Step 4: Update `scripts/train_v0.py` to route from the new contract**

Move the top-level choice away from `get_teacher_logits_source(...)` and into the new teacher-state helper so that:
- `offline_cache` keeps the current cache-dataloader path
- `online_no_cache` switches to tokenized corpus dataloaders
- `online_write_through_cache` also uses tokenized corpus dataloaders and enables cache writes

- [ ] **Step 5: Create the two dedicated smoke configs**

Add two small configs derived from the current KL config so Packet C can verify mode behavior without mutating the main experiment config during smoke runs:
- `configs/v0_teacher_state_online_no_cache_smoke.yaml`
- `configs/v0_teacher_state_write_through_cache_smoke.yaml`

Keep them minimal: one-batch-friendly batch size, the same model/span settings, and explicit `teacher_state.mode` values.

- [ ] **Step 6: Re-run the mode tests**

Run: `./.venv/bin/python -m pytest tests/test_teacher_state_modes.py -v`
Expected: PASS.

- [ ] **Step 7: Commit the control-plane refactor**

```bash
git add src/training/teacher_state.py scripts/train_v0.py configs/v0_mixed_corpus_plus_kl_loss.yaml configs/v0_teacher_state_online_no_cache_smoke.yaml configs/v0_teacher_state_write_through_cache_smoke.yaml tests/test_teacher_state_modes.py
git commit -m "feat: add teacher-state mode contract"
```

### Task 5: Add the online token-batch path without breaking offline cache training

**Files:**
- Modify: `scripts/train_v0.py`
- Modify: `src/data/dataset_factory.py`
- Modify: `src/data/mixed_corpus.py`
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing router smoke tests**

Add tests that patch the router and assert:
- offline mode still calls `create_cache_dataloader`
- online modes call the token dataset loader path instead

- [ ] **Step 2: Run those router tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_train_smoke.py -k "teacher_state or dataloader" -v`
Expected: FAIL on the new online-mode assertions.

- [ ] **Step 3: Reuse the existing dataset factory for online modes**

Avoid a new corpus implementation. Wire `scripts/train_v0.py` to build namespace-compatible config objects and reuse `get_experiment_dataloaders(...)` so token batches contain at least `input_ids` and `attention_mask`.

- [ ] **Step 4: Preserve offline behavior**

Keep the current cache path intact for `offline_cache`, including deterministic sampling and split handling.

- [ ] **Step 5: Re-run the router tests**

Run: `./.venv/bin/python -m pytest tests/test_train_smoke.py -k "teacher_state or dataloader" -v`
Expected: PASS.

- [ ] **Step 6: Commit the dataloader routing slice**

```bash
git add scripts/train_v0.py src/data/dataset_factory.py src/data/mixed_corpus.py tests/test_train_smoke.py
git commit -m "feat: route training dataloaders by teacher-state mode"
```

### Task 6: Teach the trainer to compute or consume teacher states per mode

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `src/data/teacher_cache.py`
- Modify: `tests/test_train_smoke.py`
- Modify: `tests/test_teacher_cache.py`

- [ ] **Step 1: Write failing trainer tests for each mode**

Add focused tests covering:
- offline mode consumes cached `h_start`, `velocity_target`, and optional `teacher_logits`
- online-no-cache mode performs one live teacher forward/extraction and populates the loss batch
- write-through mode performs the same live extraction and calls a cache writer hook when enabled

- [ ] **Step 2: Run the trainer tests to confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_train_smoke.py -k "online_no_cache or write_through or offline_cache" -v`
Expected: FAIL because trainer logic is still logits-only.

- [ ] **Step 3: Reuse `QwenInspector` for live teacher extraction**

Do not reimplement boundary extraction. Add a small adapter inside `Trainer` or `src/training/teacher_state.py` that turns teacher outputs into the same batch keys the loss already expects:
- `h_start`
- `velocity_target`
- optional `h_target`
- optional `teacher_logits`

- [ ] **Step 4: Add optional write-through persistence**

Extend `src/data/teacher_cache.py` so write-through mode can persist teacher states using the same metadata contract as offline cache generation. Persistence must be optional; the training step stays correct even if writing is disabled.

- [ ] **Step 5: Re-run the trainer and cache tests**

Run:
- `./.venv/bin/python -m pytest tests/test_train_smoke.py -k "online_no_cache or write_through or offline_cache" -v`
- `./.venv/bin/python -m pytest tests/test_teacher_cache.py -k "metadata or write" -v`

Expected: PASS.

- [ ] **Step 6: Commit the runtime slice**

```bash
git add src/training/trainer.py src/data/teacher_cache.py tests/test_train_smoke.py tests/test_teacher_cache.py
git commit -m "feat: support live and write-through teacher-state training"
```

### Task 7: Add parity checks and mode-specific smoke commands

**Files:**
- Create: `tests/test_teacher_state_parity.py`
- Modify: `docs/decision_learning.md`
- Modify: `docs/state.md`

- [ ] **Step 1: Write the parity test against cached teacher states**

Use a shared token batch and assert that live teacher extraction matches cached targets within an explicit tolerance.

```python
def test_live_teacher_matches_cached_boundary_targets(shared_batch):
    assert torch.allclose(live["h_start"], cached["h_start"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(live["velocity_target"], cached["velocity_target"], atol=1e-5, rtol=1e-4)


def test_live_teacher_matches_cached_logits_when_available(shared_batch):
    if "teacher_logits" in cached:
        assert torch.allclose(live["teacher_logits"], cached["teacher_logits"], atol=1e-5, rtol=1e-4)
```

Treat the logits assertion as required whenever the parity fixture exposes both live and cached logits.

- [ ] **Step 2: Run the parity test and confirm the initial failure**

Run: `./.venv/bin/python -m pytest tests/test_teacher_state_parity.py -v`
Expected: FAIL until the live extraction adapter is wired.

- [ ] **Step 3: Make the parity test pass and document tolerances**

Keep tolerances explicit in the test so future runtime changes do not silently drift.

- [ ] **Step 4: Run end-to-end smoke commands for all three modes**

Run:

```bash
./.venv/bin/python scripts/train_v0.py --config configs/v0_mixed_corpus_plus_kl_loss.yaml --limit-train-batches 1 --limit-val-batches 1
./.venv/bin/python scripts/train_v0.py --config configs/v0_teacher_state_online_no_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1
./.venv/bin/python scripts/train_v0.py --config configs/v0_teacher_state_write_through_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1
```

Expected:
- offline mode succeeds on a compatible cache
- online-no-cache succeeds without a prebuilt cache
- write-through mode succeeds and emits compatible cache artifacts

- [ ] **Step 5: Record the design decision and state update**

Summarize:
- why runtime policy moved to `teacher_state.mode`
- how cache compatibility is enforced
- what smoke/parity evidence was gathered

- [ ] **Step 6: Commit Packet C completion**

```bash
git add tests/test_teacher_state_parity.py docs/decision_learning.md docs/state.md
git commit -m "test: add teacher-state parity coverage"
```

### Packet C exit criteria

- `offline_cache` still works.
- `online_no_cache` runs one batch without a prebuilt cache.
- `online_write_through_cache` runs one batch and can emit compatible cache artifacts.
- Parity coverage proves live teacher extraction matches offline teacher targets, and cached logits when available, within stated tolerances before any new long-context training run.

---

## Final verification checklist

- [ ] `./.venv/bin/python -m pytest tests/test_mmlu_pro_behavior.py -v`
- [ ] `./.venv/bin/python -m pytest tests/test_teacher_cache.py -v`
- [ ] `./.venv/bin/python -m pytest tests/test_train_smoke.py -v`
- [ ] `./.venv/bin/python -m pytest tests/test_teacher_state_modes.py -v`
- [ ] `./.venv/bin/python -m pytest tests/test_teacher_state_parity.py -v`
- [ ] `./.venv/bin/python scripts/train_v0.py --config configs/v0_mixed_corpus_plus_kl_loss.yaml --limit-train-batches 1 --limit-val-batches 1`
- [ ] `./.venv/bin/python scripts/train_v0.py --config configs/v0_teacher_state_online_no_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1`
- [ ] `./.venv/bin/python scripts/train_v0.py --config configs/v0_teacher_state_write_through_cache_smoke.yaml --limit-train-batches 1 --limit-val-batches 1`
- [ ] Packet A inference artifacts exist and are distinct.
- [ ] Long-context config either smokes successfully on regenerated cache or fails with the intended compatibility error.
- [ ] All three teacher-state modes have a one-batch smoke path.

## Suggested execution beans

Create child beans under `midflowlm-ml5g` instead of running the whole plan in one session:

1. `Implement KL follow-up inference prompt modes and probes`
2. `Add long-context KL config and cache compatibility validation`
3. `Refactor teacher-state sourcing into offline and online modes`

## Notes for the implementing agent

- Reuse before building: keep teacher extraction anchored on `src/model/qwen_parity.py` and token data loading anchored on `src/data/dataset_factory.py`.
- Do not start long-context training until the parity test and cache regeneration rule are both in place.
- Prefer separate commits per packet so any failure in Packet C does not contaminate the already-valuable Packet A/B evidence.

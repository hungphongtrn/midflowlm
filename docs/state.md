# Current State

Last updated: 2026-03-19

## Project summary

MidflowLM is an experimental hidden-state distillation project built on top of `Qwen/Qwen3.5-0.8B`.

- Goal: replace a span of frozen Qwen layers with a trainable iterative midblock
- Current replacement span: layers `8..11`
- Current training mode: architecture training from cached hidden-state targets
- Current dataset: `roneneldan/TinyStories`
- Current max trained refinement steps: `max_steps_T = 8`

## Current architecture

The main model path lives in `src/model/student_qwen.py`.

- The full Qwen model is loaded and frozen
- Layers before the replacement span stay frozen and unchanged
- Layers `8..11` are replaced by a trainable `IterativeMidblock`
- Upper frozen layers and LM head are reused for final logits
- A bypass/original path is available via `bypass_mode=True` for side-by-side comparison against the non-replaced model

Step conditioning is implemented in `src/model/adapter.py`.

- Configured mode is effectively the combined discrete + normalized timestep path
- `max_steps_T` is `8`
- For `num_steps > max_steps_T`, the discrete embedding clamps to the last trained step index while normalized `t/T` still changes
- This means larger values like `32` or `64` are technically runnable but are out-of-distribution relative to training

## Current configuration

Primary experiment config: `configs/v0_onemotif.yaml`

- Model: `Qwen/Qwen3.5-0.8B`
- Span: `start_layer=8`, `end_layer=11`
- Trained T values: `[1, 2, 4, 6, 8]`
- T weights: `[0.20, 0.35, 0.20, 0.15, 0.10]`
- Cache policy: hidden states enabled, logits disabled
- Losses: endpoint + trajectory only (`kl_weight=0.0`, `ce_weight=0.0`)
- Training loop: `max_epochs=3`, `precision=bf16-mixed`

## Data and cache state

The project builds an offline teacher cache before training.

- Cache builder: `scripts/build_teacher_cache.py`
- Cache currently stores hidden-state targets needed for architecture training
- Durable decision: default offline cache should avoid storing full teacher logits because of storage cost
- This decision is recorded in `docs/decision_learning.md`

## Training and checkpoint state

The newest completed run is:

- `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/`

Training log highlights from `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/logs/train_11-12-19-03-2026-v0_qwen_iterative_midblock_20260319_111219.jsonl`:

- validation step: `1781`
- final validation loss: `0.00037349793035537003`
- final validation perplexity: `8.438456950582227`
- reported best validation metric: `0.00037554717576131224`
- training duration: about `4981s`
- error count: `0`

Latest checkpoint metadata:

- `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/best.ckpt`
  - format: `trainer_checkpoint`
  - `global_step = 1750`
  - `current_epoch = 2`
- `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt`
  - format: `trainer_checkpoint`
  - `global_step = 1781`
  - `current_epoch = 2`

The two checkpoints are distinct artifacts. A direct parameter comparison shows a small best-to-final drift with relative L2 change of about `0.00108`, with the largest tensor movement in `midblock.velocity_proj.1.weight`.

## Recent additions

Recent inference utilities were added for qualitative checkpoint testing.

- `scripts/run_checkpoint_text_sweep.py` runs prompt-based generation sweeps
- `src/eval/text_checkpoint_sweep.py` supports:
  - original frozen model output
  - trained checkpoint output
  - multiple `num_steps` values
  - JSON export
  - plain-text comparison table export

Current output structure in `outputs/text_sweep.json` includes:

- run metadata
- warnings for out-of-range `num_steps`
- per-input side-by-side comparisons
- a rendered comparison table

## Latest qualitative findings

There are now two relevant sweep artifacts:

- older broad stress sweep: `outputs/text_sweep.json`
- fresh latest-run sweeps:
  - `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/text_sweep_best.json`
  - `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/text_sweep_final.json`

Observed behavior from the fresh latest-run sweeps (`num_steps = [1, 4, 8]`, `max_new_tokens = 96`):

1. `final.ckpt` is the better handoff checkpoint for qualitative work.
   - On the village prompt, `final.ckpt` is noticeably closer to the teacher-style continuation and improves further at `T=4/8`.
   - On the brave-cat prompt, `final.ckpt` at `T=4` becomes much more coherent than `best.ckpt`, which stays trapped in a first-person repetition loop.
   - `final.ckpt` at `T=8` remains usable, but shifts the prompt semantics slightly (`hero` to `trying to escape from a house`), so `T=4` currently looks safer than `T=8` for open-ended prompting.

2. The robot prompt still shows a major failure mode.
   - Both `best.ckpt` and `final.ckpt` repeat the same sentence almost verbatim across `T=1/4/8`.
   - This means the latest run improved narrative coherence on some prompts without fixing short-prompt repetition.

3. Manual repetition analysis is more trustworthy than the JSON field right now.
   - The built-in `repetition_metrics` written by `src/eval/text_checkpoint_sweep.py` are currently incorrect because the aggregation code reads a missing row-level `generated_text` field.
   - Manual recomputation from the saved generations shows the real pattern:
     - robot prompt remains extremely repetitive for both checkpoints (`repeat_2gram_ratio ~= 0.84`)
     - brave-cat prompt improves sharply for `final.ckpt` at `T=4` (`repeat_2gram_ratio ~= 0.032`) versus `best.ckpt` (`~= 0.80`)
     - village prompt repetition is low-to-moderate and better on `final.ckpt` at `T=4/8`

Observed behavior from the older broad sweep in `outputs/text_sweep.json`:

1. Within the trained range (`1`, `4`, `8`):
   - `8` often looked best in earlier spot checks
   - `1` and `4` were more generic and repetitive in the older checkpoint
   - the robot prompt already collapsed into sentence repetition across original and trained variants

2. Outside the trained range (`32`, `64`):
   - output quality degrades sharply
   - generations become junky, repetitive, or multilingual token loops
   - these settings should be treated as stress tests, not valid quality settings

3. Across all tested runs:
   - generations often run to the full token budget
   - `stopped_on_eos` is typically `false`
   - repetition remains the dominant qualitative failure mode

## Interpretation of current status

The codebase is in a usable experimental state for hidden-state architecture training and qualitative inference checks, but not yet in a strong behavior-quality state.

- The replacement block can generate coherent text for some prompts, and the newest `final.ckpt` is better than the paired `best.ckpt` for handoff/demo sampling
- `num_steps=4` currently looks like the safest qualitative default for the newest run, with `num_steps=8` still useful but less stable on prompt semantics
- The model does not yet show robust qualitative gains across prompt types
- Repetition remains a major issue, especially on short continuation prompts
- Pushing `num_steps` beyond `8` does not help and currently makes outputs much worse

## What appears settled

- Hidden-state-only cache is the default direction for architecture training
- The current v0 setup is focused on hidden-state matching, not behavior distillation
- Side-by-side original vs trained qualitative comparison is now supported

## What remains open

- Whether decoding changes alone can reduce repetition enough to improve perceived quality
- Whether behavior training should be added next (KL, CE, GRPO, or related objectives)
- Whether the current step-conditioning design should be changed to better support larger `num_steps`
- Whether broader prompt evaluation and automatic repetition metrics should become standard

## Recommended immediate next steps

1. Use `num_steps=1,4,8` for comparisons and treat `32/64` as out-of-distribution only
2. Add decoding controls to the sweep script (`temperature`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`)
3. Add automatic repetition metrics to JSON output
4. Expand qualitative evaluation to a larger prompt set before making training-direction decisions
5. If repetition persists under better decoding, discuss behavior-training objectives rather than increasing step count

---

## KL Follow-up Progressive Disclosure (2026-03-22)

### What was built

Added three-mode teacher state sourcing infrastructure supporting offline cache, live extraction, and write-through cache.

**Files created**:
- `src/training/teacher_state.py` - mode enum + resolution/validation helpers
- `configs/v0_teacher_state_online_no_cache_smoke.yaml` - online no-cache smoke config
- `configs/v0_teacher_state_write_through_cache_smoke.yaml` - write-through smoke config
- `configs/v0_mixed_corpus_plus_kl_loss_long_context.yaml` - long-context variant (seq_len=512)
- `tests/test_teacher_state_modes.py` - 15 mode contract tests
- `tests/test_teacher_state_parity.py` - parity tests (3 passed, 2 GPU-integration skipped)

**Files modified**:
- `src/training/trainer.py` - added mode-aware teacher state handling
- `src/training/data.py` - added `validate_cache_compatibility()`
- `scripts/train_v0.py` - mode routing, validation, online dataloader path
- `tests/test_train_smoke.py` - 3 new mode-specific trainer tests
- `src/eval/mmlu_pro_behavior.py` - fixed regex for parenthetical options

**Smoke results**:
- Online no-cache: PASS (train step + val step + perplexity + checkpoint)
- Write-through: PASS (same, plus cache writes)
- Offline: validation fails with correct mismatch error (cache store_logits=False vs kl_weight=0.25)

**Tests**: 72 passed, 2 failed (pre-existing TestDatasetFactory baseline failures)

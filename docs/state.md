# Current State

Last updated: 2026-03-18

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

The current text sweep output references a trained checkpoint at:

- `outputs/v0_qwen_iterative_midblock/checkpoints/final.ckpt`

Checkpoint metadata from `outputs/text_sweep.json`:

- format: `trainer_checkpoint`
- `global_step = 1781`
- `current_epoch = 2`

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

The latest sweep used:

- `num_steps = [1, 4, 8, 32, 64]`
- `max_new_tokens = 128`

Observed behavior from `outputs/text_sweep.json`:

1. Within the trained range (`1`, `4`, `8`):
   - `8` generally looks best for the story prompt
   - `1` and `4` are more generic and repetitive
   - the robot prompt still collapses into sentence repetition across original and trained variants

2. Outside the trained range (`32`, `64`):
   - output quality degrades sharply
   - generations become junky, repetitive, or multilingual token loops
   - these settings should be treated as stress tests, not valid quality settings

3. Across all tested runs:
   - generations often run to the full token budget
   - `stopped_on_eos` is typically `false`
   - repetition is the dominant qualitative failure mode

## Interpretation of current status

The codebase is in a usable experimental state for hidden-state architecture training and qualitative inference checks, but not yet in a strong behavior-quality state.

- The replacement block can generate coherent text for some prompts
- `num_steps=8` is the current best practical default among tested values
- The model does not yet show robust qualitative gains over the original model across prompts
- Repetition remains a major issue
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

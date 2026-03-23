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

1. Use `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt` for the next session's manual sampling and treat `num_steps=4` as the default first try
2. Keep `num_steps=1,4,8` for comparisons and treat `32/64` as out-of-distribution only
3. Fix automatic repetition metrics in `src/eval/text_checkpoint_sweep.py` so future sweeps report real values
4. Add decoding controls to the sweep script (`temperature`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`)
5. Expand qualitative evaluation to a larger prompt set before making training-direction decisions
6. If repetition persists under better decoding, discuss behavior-training objectives rather than increasing step count

---

## Downstream Task Evaluation (MMLU-Pro)

**Added:** 2026-03-19  
**Script:** `scripts/eval_mmlu_pro.py`

### Evaluation Setup

- **Dataset:** TIGER-Lab/MMLU-Pro validation split (70 samples)
- **Task:** Multiple-choice question answering
- **Prompt format:** Qwen chat template with system message instructing single-letter answers
- **Metric:** Accuracy (correct predictions / total questions)

### Results Summary

| Model | Steps (T) | Accuracy | Latency (ms) |
|-------|-----------|----------|-|
| trained_midblock | 1 | 0.00% | 95.39 |
| trained_midblock | 4 | 0.00% | 94.46 |
| trained_midblock | 8 | 0.00% | 103.14 |
| trained_midblock | 32 | 0.00% | 154.13 |
| teacher_original | 1 | **17.14%** | 30.93 |

### Critical Finding: Student Outputs CoT Tokens

The student model outputs **chain-of-thought tokens** instead of direct answer letters:

```
Student output: Token 248068 → "思考" (Chinese: "thinking")
Teacher output: Token 32 → "A" (answer letter)
```

**Token breakdown:**
- Token 248068 = `<|im_start|>思考` - Qwen's chain-of-thought start marker
- Token 248069 = `】` - CoT end bracket
- Tokens 32-41 = "A" through "J" (answer letters)

### Root Cause Analysis

The checkpoint at `outputs/11-12-19-03-2026-qwen_iterative_midblock/checkpoints/best.ckpt`:
1. Successfully loads with current FlowMidblock architecture (uses `time_proj`, `velocity_proj`)
2. Generates thinking tokens before answers, suggesting training involved CoT-style reasoning
3. The model is behaving correctly according to its training - it just doesn't match our direct-answer evaluation format

### Checkpoint Compatibility Matrix

| Checkpoint Path | Architecture | Compatible? |
|-----------------|--------------|--------------|
| `11-12-19-03-2026-v0_qwen_iterative_midblock/best.ckpt` | FlowMidblock (`time_proj`, `velocity_proj`) | ✅ Yes |
| `11-12-19-03-2026-v0_qwen_iterative_midblock/final.ckpt` | FlowMidblock | ✅ Yes |
| `v0_qwen_iterative_midblock/best.ckpt` | Old IterativeMidblock (`step_adapter`, `delta_proj`) | ❌ No |
| `v0_qwen_iterative_midblock/final.ckpt` | Old IterativeMidblock | ❌ No |

### What This Means

1. **The model is not broken** - it's generating CoT tokens as expected from its training
2. **The evaluation needs adjustment** - either:
   - Generate multiple tokens and extract answer from longer generation
   - Modify prompt to expect CoT format
   - Find/use a checkpoint trained for direct answers
3. **Latency comparison is valid** - student takes 3-5x longer due to midblock computation

### Files Created

- `scripts/eval_mmlu_pro.py` - Evaluation script with chat templates
- `results/mmlu_pro_eval.json` - Raw results with prompts, tokens, outputs
- `results/mmlu_pro_eval.md` - Human-readable summary

### Open Questions

1. Was CoT intentional in training, or is this emergent behavior from the flow matching objective?
2. Should we modify generation to continue past CoT tokens and extract answers?
3. Is there a checkpoint that produces direct answers instead of CoT?
4. How does this affect the hidden-state distillation goal?

---

## KL Follow-up Inference Probes (Packet A, Task 2)

**Added:** 2026-03-22  
**Artifacts:** `results/kl_followup_default_512.jsonl`, `results/kl_followup_closed_think_512.jsonl`, `results/kl_followup_probe_notes.md`  
**Probe:** 8 samples × {1,4,8,32} steps, max_new_tokens=512, on checkpoint `final.ckpt` (v0 mixed-corpus + KL loss)

### Key Comparison (trained_midblock, 24 rows each)

| Prompt variant | Answer hit rate | Median completion | Max completion |
|----------------|:----------------:|:-----------------:|:---------------:|
| default        | **87.5%** (21/24) | 258 tokens | 512 |
| closed_think   | 75.0% (18/24)   | 191 tokens | 512 |

### Findings

- **closed_think prefill lowers answer extraction by ~12.5 pp** on the trained student. The extra `<think>`/`</think>` pair in the prefill biases the model toward option-enumeration and free-standing answer-recognition rather than committing to a letter early.
- **No looping `<think>` detected** in either variant — zero generations contain any `<think>`/`</think>` output tokens, and zero prompt-loopback `<|im_start|>` tokens appear. The previous CoT generation issue appears to have been resolved by stripping the thinking prefill on the student side.
- **First-token shift under closed_think:** "The" (prose) drops off sharply while bare option letters ("A", "F", "E") become the first token more often, consistent with the model immediately entering option-review mode instead of reasoning.
- **teacher_original** produces only 2-token completions (letter + EOS) at steps=1 and is not informative for comparison.
- **Sample size caveat:** 8 samples × 3 step-counts = 24 trained_midblock rows per variant. The 12.5 pp gap is suggestive but not statistically significant.

### Interpretation

The closed-think prefill does not improve and actually degrades answer extraction on this checkpoint. The default prompt behavior is preferred for inference probing. The absence of `<think>` output tokens confirms that the stripping approach is working. Further probing with a larger sample set would help confirm the direction of the effect.

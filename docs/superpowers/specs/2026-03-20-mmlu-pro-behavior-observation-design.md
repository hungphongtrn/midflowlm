# MMLU-Pro Behavior Observation Script Design

## Goal

Create a dedicated inference script that runs trained MidflowLM checkpoints on MMLU-Pro validation prompts with the Qwen chat template, allows longer free-form generation than the current one-token evaluator, and records enough structured output to inspect how the trained model behaves.

## Scope

In scope:

- reuse the existing MMLU-Pro validation prompt format
- reuse the existing student checkpoint loading path
- perform multi-token autoregressive generation for trained checkpoints
- optionally compare against the original frozen Qwen teacher model on the same prompts
- save JSONL transcript rows plus a compact console summary
- capture answer-letter extraction as an observation, not as the primary purpose of the script

Out of scope:

- replacing the current one-token accuracy script
- redefining the official downstream metric
- changing training losses or checkpoint architecture
- adding a new trainer or new model family

## Recommended approach

Add a dedicated behavior-observation utility in `src/eval/mmlu_pro_behavior.py` and a thin CLI entrypoint in `scripts/infer_mmlu_pro_behavior.py`.

Why this approach:

- it keeps behavior inspection separate from the current scoring path in `scripts/eval_mmlu_pro.py`
- it reuses the existing `FrozenQwenStudent` loading logic and the autoregressive loop pattern already present in `src/eval/text_checkpoint_sweep.py`
- it avoids overloading one script with two different jobs: measuring accuracy and inspecting generation behavior

## Architecture

### 1. Dataset and prompt preparation

The new behavior script should reuse the same MMLU-Pro validation loading and prompt construction contract as the current evaluation script:

- sample from `TIGER-Lab/MMLU-Pro` validation split
- normalize option strings to plain option text
- apply the Qwen chat template with the same system and user prompt framing already used by `scripts/eval_mmlu_pro.py`

This keeps the observed behavior comparable to the current downstream evaluation setup.

### 2. Model modes

The script should support two model modes:

- trained student checkpoint via `FrozenQwenStudent` plus checkpoint load
- original teacher/base Qwen model for reference behavior on the same prompts

For the student path, the script should allow explicit `num_steps` values so the user can inspect how behavior changes with `T`.

### 3. Generation behavior

Unlike the current one-token evaluator, this script should autoregress for multiple tokens.

Required controls:

- `max_new_tokens`
- `temperature`
- `top_p`
- `stop_on_eos`
- `num_steps` for student inference

The default behavior should remain conservative enough to be readable, but should allow more freedom than single-token argmax so the user can see whether the student emits reasoning markers, prose continuations, delayed answer letters, or repetitive loops.

### 4. Transcript output format

The primary artifact should be JSONL with one row per `(sample, model, num_steps)` run.

Each row should contain:

- dataset metadata: sample index, category, question, options, correct answer
- generation metadata: model label, checkpoint path if present, `num_steps`, `max_new_tokens`, `temperature`, `top_p`
- prompt details: prompt text and prompt token ids
- generation details: generated token ids, generated text, completion length, stopped-on-eos flag
- lightweight answer analysis: first generated token text, first valid answer letter found anywhere in the completion, whether any valid answer letter was found

This shape makes it easy to inspect raw behavior now and do offline aggregation later without rerunning generation.

### 5. Console summary

The CLI should print a compact summary after generation finishes.

At minimum it should report:

- number of samples processed per model / `num_steps`
- rate at which any valid answer letter appears in the completion
- most common first generated strings or tokens
- most common extracted answer letters
- a few short example completions for quick inspection

The console output should stay compact; the JSONL file is the full-fidelity artifact.

## File layout

Create:

- `src/eval/mmlu_pro_behavior.py` - reusable loading, prompting, generation, transcript, and summary utilities
- `scripts/infer_mmlu_pro_behavior.py` - thin CLI wrapper
- `tests/test_mmlu_pro_behavior.py` - focused coverage for transcript shaping and longer-generation answer extraction helpers

Modify:

- `src/eval/__init__.py` - export the new reusable behavior runner if that fits current package conventions

## Data flow

The intended runtime flow is:

1. load config and tokenizer
2. load requested MMLU-Pro validation subset
3. build chat-template prompts
4. load student checkpoint and, optionally, teacher model
5. run autoregressive generation for each requested model and `num_steps`
6. write one JSONL row per transcript
7. aggregate and print a compact summary

## Error handling

- fail clearly when `num_steps` is non-positive
- warn when requested `num_steps` exceeds trained `max_steps_T`, matching existing extrapolation language where possible
- fail clearly when checkpoint path is missing or incompatible
- handle missing extracted answer letters without treating them as script errors
- keep transcript writing incremental enough that partial results are not lost if a later sample fails

## Verification

Before considering the task complete, verify:

- the script runs on a small MMLU-Pro subset with a recent checkpoint
- JSONL rows contain prompt text, generated text, token ids, and extracted answer observations
- console summary reflects the same run configuration and sample counts
- teacher/base comparison mode works if enabled
- extraction logic can identify a valid answer letter that appears after earlier free-form tokens

Required executable checks:

- `pytest tests/test_mmlu_pro_behavior.py -v`
- `./.venv/bin/python scripts/infer_mmlu_pro_behavior.py --config configs/v0_mixed_corpus.yaml --checkpoint outputs/20-07-19-03-2026-v0_qwen_mixed_corpus_midblock/checkpoints/best.ckpt --num-samples 3 --num-steps 1 4 --max-new-tokens 16`

## Notes

This design intentionally does not redefine the official MMLU-Pro metric. It adds an inspection tool so we can understand whether the student is incapable of answering, answers too late for the current evaluator, or falls into a different generation mode entirely.

# Phase 2: Data Pipeline — TBD

**Status:** Stub — will be detailed after Phase 1 completes.

## High-Level Scope
- Load `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` from HuggingFace Hub
- Meta-filter: keep samples where `meta.input_tokens + meta.output_tokens <= 8192`
- Tokenize with Qwen3.5 chat template, assistant-only label masking
- Pack sequences using TRL's `pack_dataset`
- Output HF Dataset with columns: `input_ids`, `attention_mask`, `labels`
- Batch tokenization with multiprocessing

## Key Unknowns (to resolve)
- Exact format of `meta` field in GLM-5.1-Reasoning-1M-Cleaned
- Whether the dataset uses `messages` or `conversations` format
- Tokenizer padding strategy for packing
- Sequence length for packing (likely 8192)

## Rough File Plan
- `src/data/reasoning_sft.py` — New dataset processing module
- Config additions for reasoning SFT data

**Depends on:** Phase 1

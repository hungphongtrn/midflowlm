# SFT Flow Matcher — Strategy

## Goal
Run supervised fine-tuning (CE-only) from the P3-D3 checkpoint while training **only** the `FlowMidblock` (layers 8-11 replacement). Qwen embeddings, layers outside 8-11, final norm, and LM head stay frozen. This tests whether reasoning SFT improves answer quality at fixed T=32 without changing the full Qwen backbone.

## Architecture

```
                    Trainable    Frozen
                       │           │
Qwen Embeddings        │     ✓     │
Qwen Layers 0..7       │     ✓     │
FlowMidblock (8..11)   │  ✓        │  ← warm-start from P3-D3
Qwen Layers 12..23     │     ✓     │
Final Norm + LM Head   │     ✓     │
```

- **Base model:** `Qwen/Qwen3.5-0.8B` loaded via `AutoLigerKernelForCausalLM`
- **Trainable:** `FlowMidblock` (~20M params from P3-D3 checkpoint)
- **Frozen:** Everything else (~752M params)
- **T value:** Hardcoded `thinking_level = 32` (not sampled per batch)
- **Training objective:** CE-only (next-token prediction), no velocity/KL/trajectory losses
- **Trainer:** HuggingFace `Trainer` (replaces custom `DistillationTrainer` per ADR 0001)

## Tech Stack
- `liger-kernel` — Fused CE, RMSNorm, RoPE, SwiGLU for HF Trainer
- `causal-conv1d` — Qwen3.5 GatedDeltaNet layers
- `flash-attn` — Full-attention layers (25% of Qwen3.5 layers)
- `flash-linear-attention` — GatedDeltaNet chunk/recurrent ops
- `torchdiffeq` — ODE integration in FlowMidblock (already a dependency)
- `trl` — `pack_dataset` for sequence packing
- `datasets` (HF) — Dataset loading and preprocessing

## Constraints & Assumptions
- **CUDA 13 / PyTorch 2.10** — Environment is fixed; wheels must match this combination
- **RTX 3060 (12GB)** for smoke test; **24GB+ GPUs** for full runs
- **Context window:** Filter samples where `input_tokens + output_tokens <= 8192`
- **Packing:** Use TRL's `pack_dataset` to maximize context utilization
- **No T-request learning** — T is fixed at 32; adaptive T training is left for future work
- **No `<think-level>` labels** — Plain SFT with `<think>...</think>` wrapped completions

## Phases (High-Level)

### Phase 1: Model Wrapper & SFT Setup — Foundation
**Outcome:** `SFTFlowMidblockModel` loads Qwen + FlowMidblock, warm-starts from P3-D3, produces HF Trainer compatible forward with CE loss, verifies parameter counts.
**Rough scope:** New wrapper class in `src/model/`, tests for param counts, warm-start, forward pass.

### Phase 2: Data Pipeline
**Outcome:** GLM-5.1-Reasoning-1M-Cleaned loaded, filtered by token budget, tokenized with assistant-only labels, packed.
**Rough scope:** New dataset processing module, integration with HF datasets, TRL packing.
**Depends on:** Phase 1

### Phase 3: Training & Smoke Test
**Outcome:** HF Trainer runs SFT on RTX 3060 (smoke test), full-run config documented for 24GB+.
**Rough scope:** Training config, HF Trainer wiring, smoke test on reduced data.
**Depends on:** Phase 1 + Phase 2

### Phase 4: Evaluation
**Outcome:** Multi-T eval at T=1,4,8,16,32 with accuracy, oracle, prediction-change rate, answer distribution.
**Rough scope:** Eval script, metric computation, comparison vs P3-D3 baseline.
**Depends on:** Phase 3

## Open Questions
- Does `AutoLigerKernelForCausalLM` work seamlessly with the monkey-patched layer override, or does it require special handling?
- Will the KV-cache break when layers 8-11 are intercepted? FlashAttention and causal-conv1d are layer-internal so should be fine.
- What exact format does `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` use for conversations? Need to inspect `messages` or `conversations` field.
- How much token packing overhead (padding tokens) is acceptable for efficient training?

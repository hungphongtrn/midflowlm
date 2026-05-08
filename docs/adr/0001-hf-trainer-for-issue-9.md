# 0001 — Switch to HuggingFace Trainer for CE-only training (Issue #9 onward)

## Status

Accepted

## Context

Issue #9 shifts training objective to next-token prediction (CE-only). The current custom `Trainer` (`src/training/trainer.py`) orchestrates velocity loss + KL loss + CE loss with on-the-fly teacher target extraction. That machinery is unnecessary for CE-only training.

## Decision

Switch to HuggingFace `Trainer` for issue #9 and onward. The existing custom trainer is retained for legacy distillation experiments under the alias `DistillationTrainer` (`distillation_trainer.py`).

## Why

1. **Liger Kernel integration** — HF Trainer works with `AutoLigerKernelForCausalLM`, providing fused CE loss, RMSNorm, RoPE, and SwiGLU kernels that reduce peak memory by 30-50%
2. **Ecosystem optimizations** — DeepSpeed, FSDP, FlashAttention, gradient checkpointing, and data collation are all HF-native; a custom trainer would need to reimplement each
3. **Standard dataset interface** — HF Trainer accepts `Dataset` (not `DataLoader`), enabling use of `datasets.map()`, `TRL.pack_dataset()`, and standard collators

## Considered Options

- **Keep custom Trainer** — would require reimplementing Liger Kernel fusion, dataset packing, and DeepSpeed/FSDP support. Rejected: the maintenance burden outweighs the flexibility of custom loss orchestration.
- **Wrap custom training step inside HF Trainer** (subclass `Trainer.compute_loss()`) — technically possible but defeats the purpose of reusing Liger's fused loss. Rejected: adds complexity without benefit.

## Consequences

- `src/training/trainer.py` renamed to `src/training/distillation_trainer.py`; `Trainer` aliased as `DistillationTrainer`
- New dataset pipeline produces HF `Dataset` objects instead of `DataLoader`s
- Model is monkey-patched (`model.language_model.layers`) rather than wrapped in `FrozenQwenStudent`
- Training config moves to `TrainingArguments` (HF standard)
- Flow matching losses (velocity, KL, trajectory) are unused in this path; retained in `distillation_trainer.py`

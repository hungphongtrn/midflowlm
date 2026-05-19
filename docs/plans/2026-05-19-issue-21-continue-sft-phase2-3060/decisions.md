# Decision Log

## 2026-05-19: Use Phase 1 warm-start checkpoint as redundant safety net
**Context:** `train_sft.py` currently downloads a Phase 1 (P3-D3) checkpoint for model warm-start. When resuming from a Phase 2 checkpoint, HF Trainer's `_load_from_checkpoint` will overwrite all model weights from `model.safetensors`.
**Decision:** Keep the warm-start path; it's redundant but harmless. HF Trainer resume overwrites everything.
**Rationale:** No code change needed. The warm-start ensures the model has valid midblock weights before the optimizer step, even if `_load_from_checkpoint` fails for the model but succeeds for the optimizer.
**Consequences:** Extra download time on first run; negligible.

## 2026-05-19: Target seq_len=8192 with fallback to 4096/2048
**Context:** The original run used seq_len=8192. 3060 has 12GB vs the original 24GB+ GPU.
**Decision:** Attempt seq_len=8192 first with gradient_checkpointing and adamw_8bit. Fallback to seq_len=4096 or 2048 if OOM.
**Rationale:** Reducing seq_len would require re-preprocessing the dataset, adding complexity. Better to try 8192 first and only fallback if necessary.
**Consequences:** If seq_len must be reduced, the data cache dir must change to avoid mixing different-length packed sequences.

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

## 2026-05-19: PHASE 1 COMPLETE — Max viable config is bs=1, seq_len=8192, eval disabled
**Context:** VRAM profiling on RTX 3060 (12GB) with SFTFlowMidblockModel (Qwen3.5-0.8B + FlowMidblock T=32) revealed severe memory constraints.

**Raw profiling results:**

| Mode     | bs | seq_len | peak_alloc | peak_reserved | Result |
|----------|----|---------|------------|---------------|--------|
| TRAIN    | 1  | 8192    | 10.72 GB   | 11.24 GB      | FITS (1.18 GB headroom) |
| TRAIN    | 1  | 4096    | 11.16 GB   | 11.57 GB      | FITS |
| TRAIN    | 2  | 8192    | 11.88 GB   | 12.24 GB      | OOM |
| EVAL     | 1  | 8192    | 5.71 GB    | 6.61 GB       | OOM |
| EVAL     | 1  | 4096    | 10.38 GB   | 10.97 GB      | OOM |
| EVAL     | 1  | 2048    | 6.72 GB    | 7.04 GB       | FITS |
| EVAL     | 2  | 2048    | 9.55 GB    | 10.24 GB      | OOM |
| EVAL     | 4  | 2048    | 7.50 GB    | 11.78 GB      | OOM |

**Decision:** Use bs=1, seq_len=8192, grad_accum=8 for training (effective batch = 8, matching original). For eval, disable entirely (eval_strategy="no") — even bs=1/seq_len=2048 barely fits and requires different data preprocessing.
**Rationale:** Gradient_checkpointing saves ~5GB during training by recomputing Qwen activations on backward pass. Eval mode lacks this saving, making it impossible to run at 8192 seq_len. Packed eval data is produced at 8192 tokens and truncating to 2048 would corrupt the packed sequences.
**Consequences:** No validation signal during training. Training loss is the only metric. The checkpoint's final loss was 1.077; continuation should show loss near this value.

## 2026-05-19: 2406 remaining steps, ~2-4 hours on 3060
**Context:** From trainer_state.json at step 3000: epoch=0.55499 → steps_per_epoch ≈ 5405.
**Decision:** 2406 remaining steps. At bs=1/seq_len=8192, estimated 0.3-0.5 steps/sec on 3060 → 1.3-2.2 hours. With overhead (dataloader, checkpointing) → budget 2-4 hours.
**Rationale:** Cost-effective for ~43K packed sequences remaining.
**Consequences:** Short run; no need for intermediate checkpointing. Save at end only.

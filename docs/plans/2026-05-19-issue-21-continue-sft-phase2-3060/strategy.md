# Continue Phase 2 SFT Training on RTX 3060 - Strategy

## Goal
Resume SFT training from the `midflowlm-phase2/checkpoint-3000` HF Hub checkpoint on the local RTX 3060 (12GB VRAM) with `per_device_train_batch_size=4` and `per_device_eval_batch_size=8`.

## Architecture
The checkpoint was produced by `scripts/train_sft.py` using `SFTFlowMidblockModel` + `LigerTrainer` (HF Trainer subclass). Resuming requires:
1. Downloading the full HF Trainer checkpoint directory (`model.safetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`) from HF Hub
2. Wiring `TrainingArguments.resume_from_checkpoint` to the local checkpoint path
3. The model is recreated fresh (with redundant warm-start from Phase 1 checkpoint), then HF Trainer overwrites weights on resume

## Tech Stack
- `scripts/train_sft.py` - SFT training entry point
- `src/model/sft_flow_midblock.py` - `SFTFlowMidblockModel` with monkey-patched Qwen forward
- `src/training/liger_trainer.py` - `LigerTrainer` (HF Trainer + Liger fused CE)
- `src/data/reasoning_sft.py` - Reasoning SFT dataset preprocessing
- `configs/issue-9/sft_flow_midblock_3060.yaml` - Existing 3060 smoke config (baseline)
- `configs/issue-9/sft_flow_midblock.yaml` - Full 24GB+ config (target to adapt)

## Constraints & Assumptions
- **VRAM**: 12GB on RTX 3060 (vs 24GB+ on original run)
- **Checkpoint**: Full HF Trainer checkpoint at step 3000 from `hungphongtrn/midflowlm-phase2`
- **Original config**: bs=4 train, bs=16 eval, seq_len=8192, T=32, gradient_checkpointing=true, adamw_8bit
- **Risk**: seq_len=8192 × bs=4 × T=32 intermediate states may exceed 12GB even with gradient_checkpointing
- **Fallback**: Reduce seq_len (4096/2048), lower batch_size with gradient_accumulation, reduce T
- **Token budget**: ~800M tokens remaining in the 1M-sample dataset (most already consumed by step 3000)

## Phases (High-Level)

### Phase 1: Spike Investigation - Checkpoint Download & VRAM Profiling
**Outcome:** Checkpoint downloaded + memory constraints confirmed + max viable batch size known
**Rough scope:** Download checkpoint-3000 from HF Hub, profile `SFTFlowMidblockModel` + data in 12GB VRAM, determine if bs=4/8 fits, identify fallback strategy

### Phase 2: Config Adaptation & Smoke Test
**Outcome:** Validated config that resumes training from step 3000 on 3060
**Rough scope:** Create `sft_flow_midblock_3060_resume.yaml`, run 5-step smoke test, verify loss continuity
**Depends on:** Phase 1

## Open Questions
- Does `model.safetensors` from checkpoint-3000 load correctly into `SFTFlowMidblockModel` via HF Trainer's `_load_from_checkpoint`?
- What is the peak VRAM usage at bs=4/seq_len=8192/T=32 on 3060?
- If OOM, what's the max seq_len that works with bs=4? Or max batch with seq_len=8192?
- How many steps remain before the dataset is exhausted? (step 3000 of what total?)

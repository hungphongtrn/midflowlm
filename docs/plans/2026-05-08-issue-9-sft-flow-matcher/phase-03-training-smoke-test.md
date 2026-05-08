# Phase 3: Training & Smoke Test — TBD

**Status:** Stub — will be detailed after Phase 2 completes.

## High-Level Scope
- Create training config for SFT run
- Wire up HF Trainer with `SFTFlowMidblockModel`
- Smoke test on RTX 3060: reduced dataset (1000 samples), 100 steps, verify training loop
- Document full-run config for 24GB+ GPUs: batch size, grad accumulation, epochs
- Training at fixed T=32

## Key Unknowns (to resolve)
- Memory budget on RTX 3060 (12GB) vs 24GB+ GPUs
- Optimal batch size and grad accumulation for SFT
- Whether gradient checkpointing is needed
- Learning rate and scheduler settings for SFT

## Rough File Plan
- `configs/issue-9/sft_flow_midblock.yaml` — Training config
- `scripts/train_sft.py` — SFT training script
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Smoke test config

**Depends on:** Phase 1, Phase 2

# SFT Flow Matcher — Issue #9 Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 4 — Evaluation
- **Next Up:** Phase 4 — Multi-T eval with accuracy, oracle, prediction-change rate
- **Overall Progress:** 3/4 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [Phase 2 document](./phase-02-data-pipeline.md) — Only the phase you're implementing (20 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 2 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Model Wrapper | ✅ Complete | FlowMidblock patched into frozen Qwen, warm-started from P3-D3, HF Trainer compatible | [phase-01-model-wrapper.md](./phase-01-model-wrapper.md) |
| 2 — Data Pipeline | 📋 Detailed | GLM-5.1-Reasoning-1M filtered, tokenized, packed | [phase-02-data-pipeline.md](./phase-02-data-pipeline.md) |
| 3 — Training & Smoke Test | ✅ Complete | 100 steps on RTX 3060, loss 0.79, seq_len=1024 | [phase-03-training-smoke-test.md](./phase-03-training-smoke-test.md) |
| 4 — Evaluation | 🔲 Pending | Multi-T eval with fixed-T accuracy, oracle, prediction-change rate | Stub only |

## Known Issues
See GitHub issues for details:
- [#12](https://github.com/hungphongtrn/midflowlm/issues/12) CUDA OOM at seq_len=2048 — FlowMidblock ODE memory spikes
- [#13](https://github.com/hungphongtrn/midflowlm/issues/13) MidblockMetricsCallback reports grad_norm=0
- [#14](https://github.com/hungphongtrn/midflowlm/issues/14) HF Trainer checkpoint save fails with safetensors shared tensors
- [#15](https://github.com/hungphongtrn/midflowlm/issues/15) SFTFlowMidblockModel missing gradient_checkpointing_enable hook
- [#16](https://github.com/hungphongtrn/midflowlm/issues/16) API incompatibilities with transformers v4.56+/PyTorch 2.10
See [decisions.md](./decisions.md) for rationale on major choices (HF Trainer switch, CE-only, monkey-patching, etc.).

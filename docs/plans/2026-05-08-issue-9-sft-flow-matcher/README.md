# SFT Flow Matcher — Issue #9 Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 2 — Data Pipeline
- **Next Up:** Phase 3 — Training & Smoke Test (pending Phase 2 completion)
- **Overall Progress:** 1/4 phases complete

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
| 3 — Training & Smoke Test | 📋 Detailed | CE-only SFT trains on RTX 3060, full-run documented for 24GB+ | [phase-03-training-smoke-test.md](./phase-03-training-smoke-test.md) |
| 4 — Evaluation | 🔲 Pending | Multi-T eval with fixed-T accuracy, oracle, prediction-change rate | Stub only |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices (HF Trainer switch, CE-only, monkey-patching, etc.).

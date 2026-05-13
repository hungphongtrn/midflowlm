# Issue 20 Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **All Phases Complete**
- **Overall Progress:** 3/3 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) - Understand the big picture (3 min)
2. [decisions.md](./decisions.md) - Context on choices made (5 min)
3. Phase docs for detailed implementation history.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Config + Data Pipeline | ✅ Complete | Inline preprocessing with cache hashing (17 tests) | [phase-01-*.md](./phase-01-config-data-pipeline.md) |
| 2 - Remote Checkpoint + Smoke Test | ✅ Complete | --smoke-test, auto-download checkpoint (9 tests) | [phase-02-*.md](./phase-02-remote-checkpoint-smoke-test.md) |
| 3 - HF Push + Integration | ✅ Complete | Post-training push, experiment_info.json (4 tests) | [phase-03-*.md](./phase-03-hf-push-integration.md) |

## Summary
- **Total new tests:** 30 (17 data pipeline + 9 checkpoint + 4 experiment info)
- **Files created:** `tests/test_checkpoint_download.py`, `tests/test_experiment_info.py`
- **Files modified:** `scripts/train_sft.py`, `configs/issue-9/sft_flow_midblock.yaml`, `configs/issue-9/sft_flow_midblock_3060.yaml`
- **Files removed:** `scripts/prepare_reasoning_sft_data.py` (Phase 1)

## Key Decisions
See [decisions.md](./decisions.md) for rationale on all choices.

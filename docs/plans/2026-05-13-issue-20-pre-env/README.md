# Issue 20 Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 1 - Config + Data Pipeline
- **Next Up:** Phase 2 - Remote Checkpoint + Smoke Test
- **Overall Progress:** 0/3 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) - Understand the big picture (3 min)
2. [phase-01-config-data-pipeline.md](./phase-01-config-data-pipeline.md) - Current phase (10 min)
3. [decisions.md](./decisions.md) - Context on choices made (5 min)

**Do NOT read future phases.** They may change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Config + Data Pipeline | 🔲 Not Started | Inline preprocessing with cache hashing | [phase-01-*.md](./phase-01-config-data-pipeline.md) |
| 2 - Remote Checkpoint + Smoke Test | 🔲 Pending | --smoke-test, auto-download checkpoint | [phase-02-*.md](./phase-02-remote-checkpoint-smoke-test.md) |
| 3 - HF Push + Integration | 🔲 Pending | Post-training push, acceptance criteria | [phase-03-*.md](./phase-03-hf-push-integration.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on 10 major choices made during the design interview.

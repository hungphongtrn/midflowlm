# Diagnose flat P3-D3 T-scaling and answer-space collapse

> **For agentic workers:** Use subagent-driven-development or executing-plans. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Complete — all 3 phases done
- **Overall Progress:** 3/3 phases complete
- **Issue:** [#5](https://github.com/hungphongtrn/midflowlm/issues/5)

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) - Understand the big picture (5 min)
2. [phase-01-probe-selection.md](./phase-01-probe-selection.md) - Only the phase you're implementing (15 min)
3. [decisions.md](./decisions.md) - Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Probe Selection & Skeleton | ✅ Complete | Fixed probes selected, T=1 trace output works | [phase-01-probe-selection.md](./phase-01-probe-selection.md) |
| 2 - Full Trace Capture | ✅ Complete | All trace families captured across T for full probe set | [phase-02-full-trace-capture.md](./phase-02-full-trace-capture.md) |
| 3 - Diagnostic Report | ✅ Complete | Report generated; root cause identified: flow too weak | [phase-03-report-recommendation.md](./phase-03-report-recommendation.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Artifacts (What Gets Produced)
| Artifact | Phase | Format | Location |
|----------|-------|--------|----------|
| Probe examples | 1 | JSON | `results/diagnostic_p3d3/probes.json` |
| Flow integration traces | 2 | JSON (per-T) | `results/diagnostic_p3d3/traces/` |
| Decoder/readout traces | 2 | JSON (per-T) | `results/diagnostic_p3d3/traces/` |
| Aggregate stats | 2 | CSV | `results/diagnostic_p3d3/summary.csv` |
| Diagnostic report | 3 | Markdown | `results/diagnostic_p3d3/report.md` |
| Smoke test | 3 | Shell script | `scripts/smoke_diagnostic_p3d3.sh` |
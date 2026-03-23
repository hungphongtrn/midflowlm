---
# midflowlm-urkc
title: Analyze latest checkpoints and logs
status: completed
type: task
priority: normal
created_at: 2026-03-19T06:23:00Z
updated_at: 2026-03-19T06:31:22Z
---

## Goal
Review the newest training outputs, inspect checkpoint/log quality, run inference on the latest checkpoint, and recommend next steps.

## Checklist
- [x] Inspect latest output directories and training logs
- [x] Review checkpoint metadata and compare best/final artifacts
- [x] Run text inference sweep on the latest checkpoint
- [x] Summarize findings and propose next steps

## Summary of Changes
- Reviewed the latest training log and checkpoint metadata for the `11-12-19-03-2026-v0_qwen_iterative_midblock` run.
- Verified `best.ckpt` (step 1750) and `final.ckpt` (step 1781) are distinct artifacts and measured a small relative parameter drift (~0.00108 L2).
- Ran fresh text sweeps for both checkpoints and saved reports to `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/text_sweep_best.json` and `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/text_sweep_final.json`.
- Identified that the built-in `repetition_metrics` field in the sweep payload is currently invalid because it aggregates from a missing row-level field, so manual repetition analysis was used instead.

---
# midflowlm-rvkd
title: Assess latest MidflowLM checkpoint behavior and recommend next experiments
status: completed
type: task
priority: normal
created_at: 2026-03-19T07:02:26Z
updated_at: 2026-03-19T07:03:26Z
---

## Goal
Provide a diagnosis of the current failure mode and rank the next experiments after the latest qualitative checkpoint sweeps.

## Todo
- [x] Review current config, model wrapper, and eval sweep code
- [x] Synthesize likely causes of repetition and quality mismatch
- [x] Recommend top experiments with success criteria and prioritization
- [x] Record a concise summary of the guidance

## Summary of Changes
Diagnosed the current failure mode as a generation-behavior gap: hidden-state imitation is improving, but the student remains weak under greedy autoregressive rollout and lacks token-level behavior supervision. Recommended prioritizing a quick eval/decoding cleanup pass, then adding lightweight behavior training with mostly frozen Qwen weights, and finally addressing prompt-distribution/exposure-bias mismatch with a short-prompt rollout-oriented experiment.

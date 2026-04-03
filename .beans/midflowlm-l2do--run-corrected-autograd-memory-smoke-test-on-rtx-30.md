---
# midflowlm-l2do
title: Run corrected autograd memory smoke test on RTX 3060 12GB
status: scrapped
type: task
priority: high
tags:
    - training
    - memory
    - smoke-test
created_at: 2026-03-26T15:45:14Z
updated_at: 2026-04-03T14:00:10Z
parent: midflowlm-g4k0
blocked_by:
    - midflowlm-gz99
---

After restoring CE/KL gradient flow, verify whether the corrected graph still fits on the local RTX 3060 12GB setup.

## Instructions
- Run a one-batch train-step smoke test with the corrected autograd path using the online-no-cache mixed-corpus config as the starting point.
- Record peak CUDA memory, whether the step completed, and whether gradient checkpointing helped.
- If the full config does not fit, try the smallest changes needed to make it fit (for example lower `seq_len`, reduced validation batch use, or temporary debug-only data limits) and record the results.
- Do not launch a long rerun until this bean has a clear fit/no-fit conclusion.

## Checklist
- [ ] Run one corrected train step on the 3060
- [ ] Record peak memory / OOM outcome
- [ ] If needed, test the minimum viable fallback settings
- [ ] Record the recommended training envelope for the rerun

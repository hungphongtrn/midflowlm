---
# midflowlm-ac9q
title: Inspect KL/CE teacher target data flow
status: completed
type: task
priority: normal
created_at: 2026-03-20T03:39:09Z
updated_at: 2026-03-20T03:41:26Z
---

Inspect training code to determine whether KL/CE loss paths use cached teacher targets/logits or recompute teacher outputs online.

- [x] Find KL/CE loss definitions and training wiring
- [x] Trace whether teacher outputs are loaded from cache or recomputed in the step
- [x] Summarize exact files/functions and ambiguities

## Summary of Changes

Inspected the loss definitions, training entrypoint, cache dataloader, and teacher-cache builder to determine whether KL/CE paths use offline cached targets or online teacher recomputation. Confirmed the implemented training loop consumes cache batches only, KL reads optional `teacher_logits` from cache when present, CE expects labels that are not populated by the cache pipeline, and no online teacher forward path is wired into training.

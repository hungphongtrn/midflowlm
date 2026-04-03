---
# midflowlm-2r76
title: Run mixed-corpus online KL/CE training experiment
status: completed
type: task
priority: critical
created_at: 2026-03-20T04:33:58Z
updated_at: 2026-03-26T16:00:00Z
parent: midflowlm-g4k0
blocked_by:
    - midflowlm-bh01
---

Start the next mixed-corpus training experiment with KL and/or CE supervision recomputed on the fly from a live teacher forward pass rather than loaded from cached teacher logits.

Checklist:
- [x] Confirm the training path does not consume cached `teacher_logits` for KL/CE
- [x] Implement online teacher recomputation path for KL and/or CE supervision
- [x] Add a smoke test or one-step proof that online KL/CE works
- [x] Run one batch forward/backward + optimizer step
- [ ] Start the next training run and record config/output paths
- [ ] Evaluate the resulting checkpoint against the current baseline

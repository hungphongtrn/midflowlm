---
# midflowlm-gz99
title: Restore CE/KL gradient flow through frozen Qwen upper stack
status: completed
type: bug
priority: critical
tags:
    - training
    - autograd
    - behavior-training
created_at: 2026-03-26T15:44:43Z
updated_at: 2026-03-26T16:12:35Z
parent: midflowlm-g4k0
---

Root cause: `src/model/student_qwen.py` currently executes the upper frozen Qwen stack under `torch.no_grad()`. That freezes parameters *and* severs the autograd path from CE/KL losses back into the midflow module. The fix is to freeze Qwen with `requires_grad=False` while allowing autograd to track the forward through frozen layers.

## Instructions
- Remove the `torch.no_grad()` barrier from the upper-stack logits path used by CE/KL training.
- Keep the frozen Qwen weights non-trainable via `requires_grad=False`; only the midflow module should remain trainable.
- Add a targeted verification that CE/KL backward gives non-zero gradients on midflow parameters and no gradients on frozen Qwen parameters.
- Re-run parameter-count / trainable-count checks and one forward/backward/optimizer smoke test.
- Document any memory impact discovered after enabling the real behavior-loss gradient path.

## Checklist
- [x] Replace freeze semantics so frozen layers use `requires_grad=False` without `torch.no_grad()` in the CE/KL path
- [x] Verify only midflow parameters are trainable
- [x] Add/extend a smoke test proving CE/KL gradients reach midflow
- [x] Verify frozen Qwen parameters still receive no optimizer updates
- [x] Run one batch forward/backward + optimizer step successfully
- [x] Record memory notes after the autograd fix

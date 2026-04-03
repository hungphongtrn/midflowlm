---
# midflowlm-gpic
title: Prepare Qwen-only LoRA fallback with offline teacher logits
status: todo
type: task
priority: normal
tags:
    - lora
    - fallback
    - distillation
created_at: 2026-03-26T15:45:40Z
updated_at: 2026-03-26T15:45:40Z
parent: midflowlm-g4k0
blocked_by:
    - midflowlm-ynvr
---

Fallback branch if corrected midflow-only behavior training still underperforms or is too memory-constrained.

Recommended default: LoRA on Qwen only, with offline teacher logits if behavior supervision is needed. Avoid online teacher-logit recomputation on the 3060 unless a later memory test proves it is safe. Keep this branch separate from the immediate midflow recovery run so results stay interpretable.

## Instructions
- Scope the fallback as **Qwen-only LoRA** first; do not assume a hybrid midflow+LoRA setup unless evaluation shows a strong reason to combine them.
- Add the missing dependency/config plumbing needed for LoRA/PEFT training.
- Plan how teacher logits will be sourced offline for KL supervision and how CE-only fallback would work if logits are unavailable.
- Define a smoke test, memory estimate, and comparison target against the corrected midflow rerun.

## Checklist
- [ ] Confirm the fallback scope is Qwen-only LoRA by default
- [ ] Add or plan PEFT/LoRA dependencies and config changes
- [ ] Define offline teacher-logit generation / loading path
- [ ] Define a one-step memory smoke test for the fallback
- [ ] Define success criteria against the corrected midflow rerun

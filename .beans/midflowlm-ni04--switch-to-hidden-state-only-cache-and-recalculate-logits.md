---
# midflowlm-ni04
title: 'MidflowLM: switch to hidden-state-only cache and recalculate logits online'
status: todo
type: task
priority: normal
tags:
    - midflowlm
    - decision
    - training
created_at: 2026-03-17T06:54:24Z
updated_at: 2026-03-17T06:55:00Z
---

# Fresh-context implementation task

Implement the new caching/training decision in a fresh context.

## Decision summary
1. Two training types: architecture training and behavior training.
2. Architecture training trains midflow modules to iterate like hidden-state refinement.
3. Behavior training covers KL distillation, GRPO, and other objectives that improve model behavior.
4. For architecture training, only cache hidden states.
5. Recalculate logits when needed instead of storing full teacher logits offline.

## Expected scope
- Update cache builder/cache format to hidden-state-only default.
- Update configs/docs accordingly.
- Separate architecture-training assumptions from behavior-training assumptions.
- Keep this task isolated for fresh-context execution.

## Reference
- docs/decision_learning.md

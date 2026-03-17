---
# midflowlm-bh01
title: 'MidflowLM: implement behavior training with online teacher logits'
status: todo
type: task
priority: normal
tags:
    - midflowlm
    - training
    - behavior-training
    - distillation
created_at: 2026-03-17T00:00:00Z
updated_at: 2026-03-17T00:00:00Z
---

# Fresh-context follow-up task

Implement the behavior-training path as a separate task after the architecture-training cache migration is complete.

## Scope

- define behavior-training interfaces separately from architecture training
- compute teacher logits online instead of storing them in the default offline cache
- evaluate behavior objectives such as KL distillation or related methods

## Depends on

- `midflowlm-ni04`

## Reference

- `docs/decision_learning.md`
- `docs/superpowers/specs/2026-03-17-midflowlm-ni04-design.md`

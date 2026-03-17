---
title: MidflowLM behavior training with online teacher logits
status: open
type: task
---

## Context

Architecture training now uses hidden-state-only cache by default (ni04 implementation). This separates architecture training (iterative latent matching) from behavior training (output quality objectives like KL distillation, GRPO).

## Goal

Implement behavior training that requires teacher logits using explicit online teacher forward passes, rather than offline cached logits.

## Rationale

- Offline logits consume significant storage
- Architecture training doesn't need logits
- Behavior training objectives (KL, GRPO) can compute logits on-the-fly from frozen teacher
- This maintains separation of concerns: architecture = hidden state refinement, behavior = output alignment

## Candidate Objectives

1. **KL Distillation**: Match student output distribution to teacher using online forward passes
2. **GRPO**: Group Relative Policy Optimization for RL-style training
3. **Combined**: Architecture + behavior losses with appropriate weighting

## Implementation Notes

- Reuse existing QwenInspector for online teacher extraction
- Ensure frozen teacher stays in eval mode during behavior training
- Consider memory implications of dual-model training
- May need gradient checkpointing for large batch sizes

## Dependencies

- Architecture training infrastructure (complete)
- Frozen teacher model loading (complete)
- Online teacher forward interface (to be defined)

## References

- ni04 implementation: hidden-state-only cache contract
- docs/decision_learning.md for architecture vs behavior training distinction

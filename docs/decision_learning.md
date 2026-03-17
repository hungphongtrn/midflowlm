# Decision and Learning Log

This document captures durable decisions and learnings made through discussion during MidflowLM development. Agents should append or update this file when the user and agent converge on a design decision, workflow rule, or important lesson that should shape future implementation.

## 2026-03-17 — Caching and training split

Status: accepted

Context:
During discussion about v0 experiment execution, we observed that storing full teacher logits in the offline cache makes cache generation extremely large. Since the teacher model is not prohibitively large, we decided to reserve offline caching for architecture targets and recalculate logits when needed for behavior training.

Decisions and learnings:

1. There are two types of training in this project:
   - architecture training
   - behavior training

2. Architecture training:
   - Goal: make the midflow modules able to iterate similarly to hidden-state refinement.
   - Primary supervision should come from cached hidden-state targets.
   - Offline cache should support this training mode.

3. Behavior training:
   - Goal: improve model behavior rather than only internal hidden-state refinement.
   - Candidate methods include KL distillation, GRPO, and related behavioral objectives.
   - These objectives may use recalculated teacher logits or other online behavioral signals instead of storing full logits offline.

4. Cache policy decision:
   - Recalculate logits when needed.
   - Only cache hidden states for architecture training.
   - Do not store full teacher logits in the default offline cache for architecture training.

Implications:
- The cache builder and cache format should be revised to support hidden-state-only caching by default.
- Training code should distinguish architecture-training data requirements from behavior-training data requirements.
- Any future decision to cache logits should require explicit justification because of the storage cost.

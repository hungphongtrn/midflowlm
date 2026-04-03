# Restore CE/KL Gradient Flow — Implementation Overview

**Goal:** Fix the autograd gradient path so CE/KL losses can backpropagate through frozen Qwen upper layers to the midblock.

**Architecture:** Remove `torch.no_grad()` from `_continue_from_hidden_state` while keeping Qwen parameters frozen via `requires_grad=False`. This allows autograd to track the forward pass without updating frozen weights.

**Problem:** The current implementation uses `torch.no_grad()` around frozen Qwen layers in `_continue_from_hidden_state` (line 432), which severs the autograd graph. CE/KL losses computed on logits cannot propagate gradients back to midblock parameters.

**Solution:** Remove the `torch.no_grad()` context manager from `_continue_from_hidden_state`. Frozen layers with `requires_grad=False` will forward correctly and allow gradient flow to trainable parameters (midblock), but won't accumulate gradients or receive optimizer updates.

**Tech Stack:** PyTorch, HuggingFace Transformers (Qwen)

## Files in Scope

- `src/model/student_qwen.py` — Primary file to modify (remove `torch.no_grad()`)
- `tests/test_student_qwen.py` — Add gradient flow smoke test

## Phases

| Phase | Name | Delivers | Depends On |
|-------|------|----------|------------|
| 1 | Autograd fix | Working gradient path with passing tests | — |

**Current:** Phase 1

---
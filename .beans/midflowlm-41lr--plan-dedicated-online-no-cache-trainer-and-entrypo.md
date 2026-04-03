---
# midflowlm-41lr
title: Plan dedicated online-no-cache trainer and entrypoint
status: completed
type: task
priority: normal
created_at: 2026-03-23T04:38:38Z
updated_at: 2026-03-23T04:39:08Z
---

Create a concrete implementation plan for a dedicated online_no_cache trainer and training entrypoint that keeps a single Qwen copy on GPU and reuses one teacher forward to derive h_start, h_target, velocity_target, and teacher_logits without cache writes or a separate live teacher model.

## Todo

- [x] Inspect current trainer, parity, and entrypoint code in the kl-follow-up worktree
- [x] Define architecture and likely files for a dedicated online_no_cache path
- [x] Break implementation into 2-4 subagent-friendly tasks with dependencies
- [x] List concrete verification commands and key risks/watchouts

## Summary of Changes

- Reviewed `src/training/trainer.py`, `src/model/qwen_parity.py`, `src/model/student_qwen.py`, `scripts/train_v0.py`, and the existing teacher-state tests/configs in the `kl-follow-up-exec` worktree.
- Produced a concrete implementation plan for a dedicated `online_no_cache` trainer and training entrypoint that reuses the student's single frozen Qwen copy for one-pass teacher target extraction.
- Identified likely implementation boundaries, verification commands, and risks around 2048-token memory, parity, and separation from offline/write-through paths.

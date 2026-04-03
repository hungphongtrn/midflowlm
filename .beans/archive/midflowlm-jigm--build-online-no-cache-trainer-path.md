---
# midflowlm-jigm
title: Build online-no-cache trainer path
status: completed
type: task
priority: normal
created_at: 2026-03-23T04:37:09Z
updated_at: 2026-03-23T05:02:03Z
---

Implement a dedicated online-no-cache trainer and training script for 2048-context runs using one shared in-GPU Qwen forward for hidden states and logits.
- [x] Create implementation plan
- [x] Dispatch implementation subagents with clear scopes
- [x] Integrate code changes
- [x] Run targeted verification
- [x] Summarize results and next steps

## Summary of Changes

- Added `FrozenQwenStudent.extract_teacher_targets()` to reuse the student-owned frozen Qwen for one-pass extraction of `h_start`, `h_target`, `velocity_target`, and `teacher_logits`.
- Added dedicated `src/training/online_no_cache_trainer.py` isolated from cache-writing, QwenInspector, and teacher-state branching.
- Added dedicated `scripts/train_online_no_cache.py` and `configs/v0_online_no_cache_2048.yaml` for 2048-context online-no-cache training.
- Added focused tests for the teacher extraction primitive and the new trainer, plus checkpoint-monitor logic and checkpoint state restore.

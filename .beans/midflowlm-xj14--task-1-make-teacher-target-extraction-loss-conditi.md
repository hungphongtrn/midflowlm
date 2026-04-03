---
# midflowlm-xj14
title: 'Task 1: Make Teacher Target Extraction Loss-Conditional'
status: completed
type: task
priority: normal
created_at: 2026-04-03T16:10:37Z
updated_at: 2026-04-03T16:14:42Z
---

## Description

Make the teacher target extraction conditional based on loss configuration to save GPU memory. When certain losses are disabled (e.g., KL loss), we should not compute or return the corresponding targets (e.g., teacher_logits).

## Tasks

- [x] Write failing test for loss-conditional extraction
- [x] Run test to verify it fails
- [x] Implement loss-conditional extraction in extract_teacher_targets()
- [x] Run tests to verify implementation passes
- [x] Commit the changes

## Acceptance Criteria

- extract_teacher_targets() accepts need_teacher_logits, need_velocity, need_trajectory_anchors flags
- When need_teacher_logits=False, teacher_logits is not computed or returned
- When need_velocity=False, velocity_target is not computed or returned  
- When need_trajectory_anchors=True, returns h8,h9,h10,h11 for v0.1 trajectory loss
- All existing tests still pass
- New tests verify the conditional behavior



## Summary of Changes

### Implementation
- Modified `extract_teacher_targets()` in `src/model/student_qwen.py` to accept three optional flags:
  - `need_teacher_logits: bool = True` - When False, skips computing and returning teacher logits (saves memory when KL loss is disabled)
  - `need_velocity: bool = True` - When False, skips computing velocity_target (saves compute when velocity loss is disabled)  
  - `need_trajectory_anchors: bool = False` - When True, extracts h8, h9, h10, h11 for v0.1 trajectory loss

### Key Features
- Uses `output_logits=False` in model forward call when `need_teacher_logits=False` to save computation
- Maintains full backward compatibility - default behavior returns all targets as before
- Returns `trajectory_anchors` dict with h8, h9, h10, h11 when requested

### Tests
- Created comprehensive test suite in `tests/test_loss_conditional_targets.py` with 5 tests:
  1. `test_extract_teacher_targets_conditional` - Verifies conditional skipping works
  2. `test_teacher_logits_not_computed_when_not_needed` - Verifies output_logits=False is passed
  3. `test_velocity_target_computed_when_needed` - Verifies velocity computation when needed
  4. `test_trajectory_anchors_extracted_when_requested` - Verifies trajectory anchors extraction
  5. `test_all_targets_returned_by_default` - Verifies backward compatibility

### Verification
- All 5 new tests pass
- Smoke test confirms backward compatibility with existing code
- Memory-efficient for 3090 GPU training when certain losses are disabled

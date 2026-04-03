---
# midflowlm-9o1y
title: 'Task 4: Create Common Student Family Interface'
status: in-progress
type: task
priority: normal
created_at: 2026-04-03T16:26:52Z
updated_at: 2026-04-03T16:27:59Z
---

## Context
Implement Task 4 in Phase 1: Create a unified StudentFamilyInterface that wraps A1, A2, and A3 families with a consistent API.

## Acceptance Criteria
- Create tests/test_student_interface.py with TDD approach
- Create src/model/student_interface.py with StudentFamilyInterface class
- Interface must support forward_refinement() with unified API
- Must return dict with endpoint_hidden and optionally trajectory_hidden
- Must work with OneShotProjector (A1), SharedRecurrentResidual (A2), and FlowMidblock (A3)

## Tasks
- [x] Write failing test (RED)
- [ ] Create StudentFamilyInterface (GREEN)
- [ ] Verify all tests pass
- [x] Commit changes

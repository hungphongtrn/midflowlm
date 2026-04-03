---
# midflowlm-g4k0
title: Mixed-corpus cache and retraining
status: scrapped
type: feature
priority: normal
created_at: 2026-03-19T10:29:05Z
updated_at: 2026-04-03T14:00:22Z
---

Add mixed-format teacher-cache pipeline and rerun MidflowLM training on broader data to test data-coverage hypothesis for MMLU-Pro.



## Tasks

### Task 1: Add mixed-corpus config and contract tests
- [ ] Step 1: Prime beans
- [ ] Step 2: Write failing config contract tests
- [ ] Step 3: Run tests to verify they fail
- [ ] Step 4: Create experiment config
- [ ] Step 5: Add remaining MCQ components
- [ ] Step 6: Run config contract tests
- [ ] Step7: Commit

### Task 2: Implement mixed-corpus formatter and dataloader
- [ ] Step 1: Write failing formatter tests
- [ ] Step 2: Run tests to verify they fail
- [ ] Step 3: Implement text-format helpers
- [ ] Step 4: Implement deterministic sampling
- [ ] Step 5: Add dataloader factory
- [ ] Step 6: Run full tests
- [ ] Step 7: Commit

### Task 3: Add dataset factory and route cache building
- [ ] Step 1: Write failing factory tests
- [ ] Step 2: Run tests to verify they fail
- [ ] Step 3: Create dataset factory
- [ ] Step 4: Add normalize_data_config
- [ ] Step 5: Replace hard-coded call in cache builder
- [ ] Step 6: Add cache-builder test
- [ ] Step 7: Run targeted tests
- [ ] Step 8: Commit

### Task 4: Record decision and smoke-test commands
- [ ] Step 1: Write decision note
- [ ] Step 2: Include smoke-test commands
- [ ] Step 3: Commit

### Task 5: Build pilot cache and verify
- [ ] Step 1: Run unit tests
- [ ] Step 2: Build tiny train cache
- [ ] Step 3: Build tiny val cache
- [ ] Step 4: Verify cache metadata
- [ ] Step 5: Record results

### Task 6: Generate full cache and train
- [ ] Step 1: Build full train cache
- [ ] Step 2: Build full val cache
- [ ] Step 3: Run fast-dev smoke test
- [ ] Step 4: Run limited-batch training check
- [ ] Step 5: Launch full training
- [ ] Step 6: Record checkpoint paths

### Task 7: Evaluate and update state
- [ ] Step 1: Run text sweep
- [ ] Step 2: Run MMLU-Pro
- [ ] Step 3: Compare baseline
- [ ] Step 4: Update state.md
- [ ] Step 5: Commit

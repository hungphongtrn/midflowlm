---
# midflowlm-hbqg
title: Refactor teacher-state sourcing into offline and online modes
status: completed
type: task
priority: normal
tags:
    - midflowlm
    - kl-follow-up
created_at: 2026-03-21T08:44:08Z
updated_at: 2026-03-23T03:23:31Z
---

Execute Packet C from docs/superpowers/plans/2026-03-21-kl-follow-up-progressive-disclosure.md: add teacher_state.mode routing, online/live teacher extraction, write-through cache support, and parity plus smoke coverage.


## Task 6 Summary (2026-03-22)

### What was done

**Tests added** (`tests/test_train_smoke.py::TestTrainerTeacherStateModes`):
- `test_trainer_offline_mode_consumes_cached_teacher_states` - verifies offline mode uses cached states
- `test_trainer_online_no_cache_mode_extracts_live_teacher_states` - verifies online mode extracts live teacher states
- `test_trainer_online_write_through_mode_extracts_and_writes_cache` - verifies write-through mode extracts and writes

**Trainer changes** (`src/training/trainer.py`):
- Added `_setup_teacher_state_mode()` to resolve and log teacher state mode
- Added `_get_live_teacher_extractor()` for lazy QwenInspector initialization
- Added `_setup_cache_writer()` for write-through mode (lazy)
- Added `_maybe_extract_teacher_states()` to extract teacher states for online modes
- Added `_write_teacher_states_to_cache()` for write-through cache persistence
- Modified `train_step()` to call `_maybe_extract_teacher_states()` for online modes

**Bug fix** (`src/eval/mmlu_pro_behavior.py`):
- Fixed regex lookbehind from `(?<=\s)` to `(?<=[\s\(\[\{])` to match parenthetical options like `(B)`

### Tests
- 72 passed, 2 failed (pre-existing baseline failures in TestDatasetFactory)


## Task 7 Summary (2026-03-22)

### What was done

**Parity tests** (`tests/test_teacher_state_parity.py`):
- `TestLiveVsCachedParity`: 3 tests (2 GPU-integration skipped, 1 tolerance doc test passed)
- `TestTrainerOnlineModeParity`: 2 tests (both passed)
- Note: GPU integration tests require `torch.cuda.is_available()` and are skipped by default

**Bug fix**:
- Added `_maybe_extract_teacher_states()` call in `val_step()` - previously only `train_step()` had it

**Smoke commands**:
- Online no-cache: PASS (train step, val step, perplexity, checkpoint)
- Write-through: PASS (same + cache writes)
- Offline: fails with correct cache mismatch error (store_logits=False vs kl_weight=0.25)

**Documentation**:
- `docs/decision_learning.md`: Added teacher_state.mode decision record
- `docs/state.md`: Added KL follow-up progressive disclosure section

### Final test results
- `test_mmlu_pro_behavior.py`: 6 passed
- `test_teacher_state_modes.py`: 15 passed
- `test_teacher_state_parity.py`: 3 passed, 2 skipped (GPU)
- `test_train_smoke.py`: 18 passed
- `test_teacher_cache.py`: 33 passed, 2 failed (pre-existing baseline)
- **Total: 75 passed, 2 skipped, 2 failed (pre-existing)**

### Smoke results
- Online no-cache: PASS
- Write-through: PASS
- Offline: fails with correct cache mismatch (validation works)

## Packet C Complete

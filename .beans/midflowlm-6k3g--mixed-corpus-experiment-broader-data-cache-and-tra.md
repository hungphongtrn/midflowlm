---
# midflowlm-6k3g
title: 'Mixed-corpus experiment: broader data cache and training'
status: in-progress
type: task
priority: normal
created_at: 2026-03-19T10:40:50Z
updated_at: 2026-03-19T10:46:00Z
---

Implement mixed-corpus data loader and training run per the 2026-03-19 plan. Tasks: (1) Add mixed-corpus config and contract tests, (2) Implement formatter and dataloader, (3) Add dataset factory and route cache building through it, (4) Record decision, (5) Build pilot cache, (6) Generate full cache and retrain, (7) Evaluate checkpoint and update state.


## Progress
- [x] Task 1: Add mixed-corpus experiment config and contract tests (configs/v0_mixed_corpus.yaml, tests/test_mixed_corpus_data.py)
- [x] Task 2: Implement mixed-corpus formatter and dataloader (src/data/mixed_corpus.py)
- [x] Task 3: Add dataset factory and route cache building through it (src/data/dataset_factory.py, scripts/build_teacher_cache.py)
- [x] Task 4: Record decision in docs/decision_learning.md
- [ ] Task 5: Build pilot mixed teacher cache
- [ ] Task 6: Generate full mixed cache and rerun training
- [ ] Task 7: Evaluate retrained checkpoint and update state

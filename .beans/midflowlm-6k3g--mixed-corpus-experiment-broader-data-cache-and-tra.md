---
# midflowlm-6k3g
title: 'Mixed-corpus experiment: broader data cache and training'
status: completed
type: task
priority: normal
created_at: 2026-03-19T10:40:50Z
updated_at: 2026-03-20T05:05:00Z
---

Implement mixed-corpus data loader and training run per the 2026-03-19 plan. Tasks: (1) Add mixed-corpus config and contract tests, (2) Implement formatter and dataloader, (3) Add dataset factory and route cache building through it, (4) Record decision, (5) Build pilot cache, (6) Generate full cache and retrain, (7) Evaluate checkpoint and update state.


## Progress
- [x] Task 1: Add mixed-corpus experiment config and contract tests (configs/v0_mixed_corpus.yaml, tests/test_mixed_corpus_data.py)
- [x] Task 2: Implement mixed-corpus formatter and dataloader (src/data/mixed_corpus.py)
- [x] Task 3: Add dataset factory and route cache building through it (src/data/dataset_factory.py, scripts/build_teacher_cache.py)
- [x] Task 4: Record decision in docs/decision_learning.md
- [x] Task 5: Build pilot mixed teacher cache (cache/mixed_qwen_boundary_states, 21500 train / 1370 val samples)
- [x] Task 6: Generate full mixed cache and rerun training (outputs/20-07-19-03-2026-v0_qwen_mixed_corpus_midblock/checkpoints/best.ckpt and final.ckpt)
- [ ] Task 7: Evaluate retrained checkpoint and update state (requires torch/datasets environment)

## Commit
- 91540d6 feat: add mixed-corpus experiment config, loader, and dataset factory

## Eval Notes

- Training run completed successfully with final checkpoint at outputs/20-07-19-03-2026-v0_qwen_mixed_corpus_midblock/checkpoints/final.ckpt.
- MMLU-Pro on the latest final checkpoint remained 0/70 at T=1,4,8.
- Teacher/original Qwen remained 12/70 (17.14%) on the same split.
- Mixed-corpus training changed the student first-token behavior from <think> markers to mostly The/Answer, but did not recover answer-letter accuracy.

- Best-checkpoint MMLU-Pro run in results/mmlu_pro_eval_mixed_best.json also stayed at 0/70 for T=1,4,8; no downstream win over final.ckpt.

---
# midflowlm-qcj0
title: Add mixed-corpus experiment config and contract tests
status: completed
type: task
priority: normal
created_at: 2026-03-19T10:31:17Z
updated_at: 2026-03-19T16:46:46Z
---

Create v0_mixed_corpus.yaml config with mixture loader and 5+ dataset components. Add contract tests to verify loader type and hidden-state-only loss configuration.

## Summary of Changes

Created configs/v0_mixed_corpus.yaml and the mixed-corpus contract tests used by the broader-data experiment. This task is already reflected in commit 91540d6 and the follow-on mixed-corpus run.

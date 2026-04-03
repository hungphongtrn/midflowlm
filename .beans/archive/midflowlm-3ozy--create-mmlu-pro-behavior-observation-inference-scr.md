---
# midflowlm-3ozy
title: Create MMLU-Pro behavior observation inference script
status: completed
type: task
priority: normal
created_at: 2026-03-19T17:03:31Z
updated_at: 2026-03-20T05:04:40Z
---

Build a script to run trained MidflowLM checkpoints on MMLU-Pro validation prompts with chat templates and longer free-form generation so we can inspect model behavior beyond one-token accuracy.

## Tasks
- [x] Explore current MMLU-Pro eval and text sweep context
- [x] Clarify desired inference output shape (JSONL transcripts + console summary)
- [x] Propose design options and recommendation
- [ ] Implement the new inference script in src with a runnable entrypoint (user requested direct implementation)
- [ ] Run the script on a recent checkpoint and inspect outputs
- [ ] Summarize observed behavior and usage

## Design Doc

- Spec written to docs/superpowers/specs/2026-03-20-mmlu-pro-behavior-observation-design.md
- Spec review attempted twice via subagent; no issues were returned, but the reviewer also did not emit a formal verdict.

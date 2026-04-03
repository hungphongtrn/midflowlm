---
# midflowlm-ql53
title: Run corrected no-think MMLU-Pro control
status: scrapped
type: task
priority: normal
created_at: 2026-03-20T04:53:16Z
updated_at: 2026-03-20T05:02:59Z
---

Run a corrected MMLU-Pro free-form control experiment on the latest checkpoint with the assistant <think></think> prefill removed, force GPU inference, save to a new traceable results file, and analyze extracted-letter accuracy and format adherence versus the prefill baseline.

Checklist:
- [x] Identify latest checkpoint and baseline settings
- [ ] Run no-prefill inference on GPU to a new results file
- [ ] Run or identify matching prefill baseline for comparison
- [ ] Compute extracted-letter accuracy and format adherence
- [ ] Summarize command, outputs, and comparison

## Reasons for Scrapping

User requested closing this task after confirming the key behavioral outcome: removing the assistant `<think></think>` prefill leads the model to generate reasoning text directly, so the planned control run is no longer needed in this task.

## Summary of Changes

- Identified the latest checkpoint and baseline settings
- Verified the prompt-building path and how the assistant `<think></think>` prefill is introduced
- Added an inference-side option to strip that prefill for future controlled runs
- Observed the core behavior that no-prefill prompts cause the model to emit reasoning in generation

---
# midflowlm-rdms
title: Add checkpoint text sweep script
status: completed
type: task
priority: normal
created_at: 2026-03-18T06:00:22Z
updated_at: 2026-03-18T06:04:34Z
---

Create a utility script to load a trained checkpoint, run it on multiple input texts, and compare outputs across different num_steps values.

## Todo
- [x] Inspect existing checkpoint loading and model inference utilities
- [x] Implement a script for text-based checkpoint sweeps across step counts
- [x] Smoke-check script CLI/help and document usage

## Summary of Changes
- Added a checkpoint text sweep utility under `src/eval/text_checkpoint_sweep.py` that loads trainer or midblock checkpoints, tokenizes prompts, and runs greedy generation across configurable `num_steps` values.
- Added `scripts/run_checkpoint_text_sweep.py` as a CLI wrapper for prompt lists, prompt files, output JSON export, and configurable generation settings.
- Verified the new CLI help output and compiled the new Python files with `python3 -m py_compile`.

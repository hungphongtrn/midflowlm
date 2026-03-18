---
# midflowlm-2cso
title: Add side-by-side text sweep comparison
status: completed
type: task
priority: normal
created_at: 2026-03-18T06:22:09Z
updated_at: 2026-03-18T06:24:01Z
---

Update the checkpoint inference utility to compare the original frozen model against the trained model across multiple num_steps values, with JSON and table output formats.

## Todo
- [x] Inspect current sweep script and step-conditioning constraints
- [x] Add original-vs-trained side-by-side generation output in JSON and table forms
- [x] Verify CLI behavior and document num_steps limits

## Summary of Changes
- Updated the text sweep utility to run both the original frozen Qwen path and the trained replacement model, then emit side-by-side comparison rows.
- Added a plain-text table with columns `input | original output | num_steps = ...` and stored that rendered table inside the JSON output alongside detailed per-run metadata.
- Added validation and warnings for `num_steps` values above `max_steps_T`, and verified the CLI help plus Python compilation with `python3 -m py_compile`.

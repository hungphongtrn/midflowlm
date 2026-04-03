---
# midflowlm-8ren
title: Capture current online KL/CE run state before shutdown
status: completed
type: task
priority: critical
tags:
    - training
    - recovery
created_at: 2026-03-26T15:44:43Z
updated_at: 2026-03-26T15:59:59Z
parent: midflowlm-g4k0
---

Current run: `configs/v0_online_no_cache_mixed_ce_kl.yaml` writing to `outputs/v0_online_no_cache_mixed_ce_kl`.

The run likely used an invalid CE/KL gradient path because `src/model/student_qwen.py` wraps the frozen upper-stack continuation in `torch.no_grad()`, so behavior losses were observed but could not backprop into the midflow module.

## Instructions
- Record the latest checkpoint paths, tensorboard directory, and train log path before terminating the job.
- Extract the latest unique validation metrics (loss, velocity, KL, CE) and note the current global step.
- Capture whether the process was stopped cleanly and whether `best.ckpt` is usable as a midflow initialization source.
- Append a short `## Findings` section with artifact paths and the final metric snapshot.

## Checklist
- [ ] Record `best.ckpt` / latest checkpoint paths
- [ ] Record tensorboard and log paths
- [ ] Record latest unique validation metrics and step
- [ ] Stop the current training run cleanly
- [ ] Append `## Findings` with artifact summary

---
# midflowlm-v7gi
title: Audit latest KL-loss checkpoint
status: completed
type: task
priority: normal
created_at: 2026-03-21T06:56:44Z
updated_at: 2026-03-21T07:26:57Z
---

Audit the most recent checkpoint produced by KL-loss training and probe downstream performance.

- [x] Locate the latest KL-loss checkpoint and its run metadata
- [x] Inspect training logs, configs, and evaluation artifacts
- [x] Run inference or existing eval scripts to gauge downstream behavior
- [x] Summarize checkpoint quality, anomalies, and recommended next checks

## Summary of Changes

- Identified the latest KL-loss run as outputs/18-25-20-03-2026-v0_qwen_mixed_corpus_midblock_plus_kl_loss with final.ckpt and best.ckpt.
- Audited config/log artifacts and verified the run used online teacher logits with kl_weight=0.25 and no CE loss.
- Ran fresh downstream probes with .venv/bin/python: MMLU-Pro eval on final.ckpt and best.ckpt, plus a behavior-oriented MMLU-Pro inference sweep on final.ckpt.
- Confirmed the run has a severe downstream failure mode: standard eval is 0/20 for all student T values, while free-form behavior collapses to generic reasoning and repeated answer letter A.

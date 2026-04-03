---
# midflowlm-wjqt
title: Run warm-start CE/KL diagnostic matrix
status: scrapped
type: task
priority: high
created_at: 2026-03-28T08:50:13Z
updated_at: 2026-04-03T14:00:06Z
parent: midflowlm-g4k0
blocking:
    - midflowlm-ynvr
---

Launch the post-gradient-fix diagnostic matrix using fresh-optimizer warm-starts from the current best corrected checkpoint. This bean is the handoff anchor for the next session after the runs finish.

## Context
- Gradient flow is fixed, but CE/KL remain noisy and only directionally improved.
- The next recommended decision gate is a short one-epoch warm-start matrix before any LoRA fallback.
- Warm-start source checkpoint: `outputs/v0_online_no_cache_mixed_ce_kl/checkpoints/best.ckpt`
- New config files prepared:
  - `configs/v0_warmstart_ce_kl.yaml`
  - `configs/v0_warmstart_ce_only.yaml`
  - `configs/v0_warmstart_kl_only.yaml`
- Training entrypoint now supports fresh-optimizer warm-start via `--init-from-checkpoint` / `train_loop.init_from_checkpoint`.

## Runs
- [ ] Run CE+KL warm-start diagnostic (`configs/v0_warmstart_ce_kl.yaml`)
- [ ] Run CE-only warm-start diagnostic (`configs/v0_warmstart_ce_only.yaml`)
- [ ] Run KL-only warm-start diagnostic (`configs/v0_warmstart_kl_only.yaml`)
- [ ] Capture key outputs: final log path, best checkpoint path, and deduped validation metrics for each run
- [ ] Compare CE/KL/val-loss movement across the three runs
- [ ] Decide whether to continue midflow-only or switch to LoRA fallback

## Run Commands
```bash
python3 scripts/train_online_no_cache.py --config configs/v0_warmstart_ce_kl.yaml
python3 scripts/train_online_no_cache.py --config configs/v0_warmstart_ce_only.yaml
python3 scripts/train_online_no_cache.py --config configs/v0_warmstart_kl_only.yaml
```

## Resume Note
When returning in a fresh session, ask to resume this bean and analyze the completed runs. The analysis should compare deduped validation trends and recommend either another focused midflow experiment or the LoRA fallback.

## Prepared State
- Warm-start support verified with `python3 -m pytest tests/test_warm_start.py -v` (4 passed).
- Trainer/logging dedup verified with `python3 -m pytest tests/test_online_no_cache_trainer.py -v` (14 passed).
- Entry point: `scripts/train_online_no_cache.py` now accepts `--init-from-checkpoint` and config key `train_loop.init_from_checkpoint`.
- Use this bean ID in the next session: `midflowlm-wjqt`.
- Suggested first analysis command after runs: `python3 -m src.scripts.inspect_trainer_logs <run-log>` for each run log.

## Debug Notes (2026-03-28)
- Reproduced full-run OOM on first training batch for `configs/v0_warmstart_ce_kl.yaml`.
- One-step peak-memory probe on RTX 3060 12GB shows:
- `configs/v0_warmstart_ce_only.yaml`: ~8.96 GiB peak allocated (fits)
- `configs/v0_warmstart_kl_only.yaml`: ~11.34 GiB peak allocated (near ceiling)
- `configs/v0_warmstart_ce_kl.yaml`: ~11.34 GiB peak allocated (same as KL-only)
- Root cause: KL path dominates memory; CE adds little. The trainer materializes teacher logits, student logits, `log_softmax(student_logits)`, `softmax(teacher_logits)`, and elementwise KL tensors over the full Qwen vocab at seq_len 1024.
- Current evidence suggests this is a first-batch peak-memory limit, not a leak over steps.

- Additional check: non-warmstart `configs/v0_online_no_cache_mixed_ce_kl.yaml` also succeeds in `--fast-dev-run` but can OOM in a direct one-step probe at `src/training/losses.py:459` inside `F.kl_div` with the same 970 MiB request.
- Conclusion refined: warm-start is not the root cause; the KL loss path is marginal/unstable on 12 GB and can pass or fail depending on allocator state / successful transient allocations.

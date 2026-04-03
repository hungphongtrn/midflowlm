---
# midflowlm-ynvr
title: Evaluate corrected rerun and decide go/no-go for LoRA fallback
status: in-progress
type: task
priority: high
tags:
    - evaluation
    - decision
created_at: 2026-03-26T15:45:40Z
updated_at: 2026-03-28T08:29:50Z
parent: midflowlm-g4k0
blocked_by:
    - midflowlm-hnnz
---

Decide whether the corrected midflow-only behavior-training path is good enough to keep iterating, or whether the project should branch to a Qwen LoRA fallback.

## Instructions
- Compare the corrected run against the flawed baseline using aggregated validation CE/KL/velocity trends.
- If practical, include a quick checkpoint sanity check or text/eval sweep to confirm the corrected run is not only improving scalar losses.
- Write an explicit decision note: continue midflow-only, continue with more tuning, or switch to the LoRA fallback branch.
- Record the reasons for the recommendation, especially memory behavior and CE/KL responsiveness.

## Checklist
- [x] Compare corrected run vs flawed baseline
- [x] Check whether CE/KL now move in a meaningful direction
- [ ] Record whether memory usage is acceptable for longer training
- [x] Write an explicit recommendation for the next branch

## Analysis Notes
- Compared corrected run in `outputs/v0_online_no_cache_mixed_ce_kl/` against pre-fix baseline in `outputs/v0_online_no_cache_mixed_ce_kl_PRE_FIX/`.
- Confirmed gradient flow restoration from train-log optimizer-step `grad_norm`: corrected run is typically ~1.1-5.3 versus pre-fix ~0.007-0.03.
- Raw CE/KL logs are misleadingly noisy because training logs and TensorBoard write every microbatch while `global_step` only increments every 16 microbatches.
- Validation is also duplicated at the same `global_step` during accumulation, so repeated `Validation at step 250/500/...` lines are logging artifacts, not new measurements.
- After deduplicating validation steps, corrected validation improves from `val/loss` 0.2618 -> 0.2450, `val/ce_loss` 2.1792 -> 2.1183, and `val/kl_loss` fluctuates but trends down overall 0.1678 -> 0.1264.
- The corrected run still shows wide microbatch-level CE/KL variance, so there is no clean monotonic drop by eye on raw training traces even though the slower validation trend is directionally better.

## Recommendation Note
- Recommend one short, tightly scoped midflow-only diagnostic round first: warm-start midflow from a good checkpoint, prefer offline teacher/cache mode, and run CE-only vs KL-only vs CE+KL ablations.
- Do not start a full-model fine-tune now. Current evidence suggests the main issue is still optimization variance/objective setup, not yet proof that the adaptation location is fundamentally wrong.
- If the next 2-3 focused midflow experiments still fail to move evals meaningfully, switch to the planned Qwen-only LoRA fallback before considering any full-model run.

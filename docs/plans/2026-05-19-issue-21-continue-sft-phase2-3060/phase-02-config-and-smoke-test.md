# Phase 2: Config Adaptation & Smoke Test - TBD

> **This phase is stubbed.** It will be detailed after Phase 1 (spike investigation) completes and we know the max viable config.

## Phase Goal
Create `configs/issue-9/sft_flow_midblock_3060_resume.yaml` and run a short smoke test to verify training resumes correctly from checkpoint-3000 on RTX 3060.

## Files to Touch (preliminary)
- Create: `configs/issue-9/sft_flow_midblock_3060_resume.yaml` - Adapted config with resume wiring
- Modify: `scripts/train_sft.py` - May need HF Hub checkpoint download vs local warm-start distinction

## Tasks (stubbed)

### Task 1: Create resume config
- Adapt batch sizes, seq_len, and other params based on Phase 1 findings
- Add `resume_from_checkpoint` field pointing to downloaded checkpoint-3000 directory
- Set `hub_model_id` to `hungphongtrn/midflowlm-phase2` for continued checkpoint pushes

### Task 2: Smoke test resume
- Run `train_sft.py --config .../sft_flow_midblock_3060_resume.yaml --smoke-test`
- Verify loss matches checkpoint-3000 final loss within tolerance
- Verify optimizer/scheduler state is correctly loaded

### Task 3: Run continuation
- Launch full training with the validated config

## Phase Completion Criteria
- [ ] Config created and smoke-tested
- [ ] Loss continuity verified
- [ ] Training running without OOM

## Handoff Notes
After this phase, training is running. Monitor for convergence and checkpoint pushes to HF Hub.

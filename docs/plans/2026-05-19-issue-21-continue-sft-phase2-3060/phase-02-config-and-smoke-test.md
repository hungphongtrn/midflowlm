# Phase 2: Config Adaptation & Smoke Test

## Phase Goal
Create `configs/issue-9/sft_flow_midblock_3060_resume.yaml` and run a smoke test verifying training resumes from checkpoint-3000 on RTX 3060 without OOM.

## Phase 1 Findings (Context)
- **Max train:** bs=1, seq_len=8192 with gradient_checkpointing (1.18GB headroom)
- **Max eval:** Even bs=1/seq_len=2048 barely fits → **disable eval**
- **Optimizer:** adamw_8bit (HF Trainer default in train_sft.py)
- **Remaining:** ~2406 steps, ~2-4 hours
- **Checkpoint:** Downloaded to `outputs/issue-9/checkpoint-3000-continue/`

## Files to Touch
- Create: `configs/issue-9/sft_flow_midblock_3060_resume.yaml`
- Possibly modify: `scripts/train_sft.py` — skip warm-start download when resuming (redundant)

## Tasks

### Task 1: Create the 3060 resume config

**File:**
- Create: `configs/issue-9/sft_flow_midblock_3060_resume.yaml`

- [ ] **Step 1: Write the config file**

```yaml
# SFT Flow Midblock — Resume Phase 2 from checkpoint-3000 on RTX 3060 (12GB)
# Target batch: bs=1, grad_accum=8 (effective batch=8, matches original)
# Eval: disabled (even bs=1/seq_len=8192 OOMs in eval mode without grad)
seed: 1337

model:
  name: "Qwen/Qwen3.5-0.8B"
  start_layer: 8
  end_layer: 11
  thinking_level: 32

data:
  processed_dir: "./data/reasoning_sft_cache"  # Reuse preprocessed data from original run
  max_seq_length: 8192

training:
  output_dir: "./outputs/issue-9/sft_flow_midblock_3060_resume"
  run_name: "sft_flow_midblock_3060_resume"

  # Batch — max that fits 12GB (see Phase 1 profiling)
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch = 8

  # Schedule
  num_train_epochs: 1
  learning_rate: 1.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  lr_scheduler: "cosine"
  warmup_steps: 0  # Already past warmup at step 3000

  # Checkpointing
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 2
  save_only_model: false

  # Logging
  logging_steps: 10
  report_to: ["tensorboard"]
  include_num_input_tokens_seen: true

  # Evaluation — DISABLED (memory constraint)
  eval_strategy: "no"

  # Performance
  dataloader_num_workers: 2
  gradient_checkpointing: true
  dataloader_drop_last: false
  dataloader_pin_memory: true

  # Resume from HF Trainer checkpoint
  resume_from_checkpoint: "./outputs/issue-9/checkpoint-3000-continue"

  # Hub push
  push_to_hub: true
  hub_model_id: "hungphongtrn/midflowlm-phase2"
  hub_strategy: "end"

checkpoint:
  # Point warm-start to midblock.pth from our downloaded checkpoint-3000
  # (redundant — HF Trainer resume loads model.safetensors — but harmless)
  path: "./outputs/issue-9/checkpoint-3000-continue/midblock.pth"
```

- [ ] **Step 2: Verify config is valid YAML**

```bash
uv run python -c "import yaml; yaml.safe_load(open('configs/issue-9/sft_flow_midblock_3060_resume.yaml')); print('Config valid')"
```

### Task 2: Smoke test resume (5 steps)

- [ ] **Step 1: Run smoke test**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060_resume.yaml \
  --smoke-test
```

Expected output:
- Model loads without OOM
- Trainer reports "Loading model from ./outputs/issue-9/checkpoint-3000-continue"
- Training starts from step 3001 (not 0)
- Training loss near ~1.07 (matches step 3000 final loss)
- No eval runs

- [ ] **Step 2: Verify loss continuity**

```
Open tensorboard: tensorboard --logdir outputs/issue-9/sft_flow_midblock_3060_resume/logs
Check that the first logged loss is close to 1.077 (step 3000 loss was 1.0769)
```

- [ ] **Step 3: Verify checkpoint dir is reused (not re-downloaded)**

```bash
# Check that the preprocessed data cache is loaded, not recreated
ls -la data/reasoning_sft_cache/
```

### Task 3: Run full continuation

- [ ] **Step 1: Launch training**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060_resume.yaml
```

- [ ] **Step 2: Monitor**

Track:
- Loss should decrease from ~1.07
- No OOM errors
- Checkpoint saves at each 500-step interval
- Hub pushes at save intervals

## Phase Completion Criteria
- [ ] Config created and validated
- [ ] Smoke test passes (5 steps, no OOM, loss continuity)
- [ ] Full training launches without OOM
- [ ] First checkpoint pushed to HF Hub at step 3500

## Handoff Notes
After this phase, training finishes (2406 remaining steps = ~2-4 hours). Final outputs:
- `outputs/issue-9/sft_flow_midblock_3060_resume/` — local checkpoints
- `hungphongtrn/midflowlm-phase2/checkpoint-3XXX` — HF Hub checkpoints
- Run `scripts/eval_mmlu_pro.py` etc. on the final model for benchmark scores.

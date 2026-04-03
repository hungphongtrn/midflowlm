# Phase 3: Config Matrix Generation

**Goal:** Generate all YAML configuration files for the complete v0.1 experiment matrix (P1/P2/P3/P4) in `configs/v0_1_matrix/`.

**Deliverables:**
- 11 base experiment configs in `configs/v0_1_matrix/`
- Config generation script for easy modification/re-generation
- All configs use the fixed hardware profile (microbatch=2, grad_accum=8)
- All configs fixed at seq_len=1024 with truncation logging

**Files in scope:**
- Create: `configs/v0_1_matrix/` directory structure
- Create: `configs/v0_1_matrix/p1_a1_proj_mixb_endkl.yaml`
- Create: `configs/v0_1_matrix/p1_a2_rrb_mixb_endkl.yaml`
- Create: `configs/v0_1_matrix/p1_a3_flow_mixb_endkl.yaml`
- Create: `configs/v0_1_matrix/p2_l1_flow_mixb_end.yaml`
- Create: `configs/v0_1_matrix/p2_l2_flow_mixb_endkl.yaml`
- Create: `configs/v0_1_matrix/p2_l3_flow_mixb_endtrajkl.yaml`
- Create: `configs/v0_1_matrix/p2_l4_flow_mixb_endtrajklce.yaml`
- Create: `configs/v0_1_matrix/p3_d1_flow_mixa_endtrajkl.yaml`
- Create: `configs/v0_1_matrix/p3_d2_flow_mixb_endtrajkl.yaml`
- Create: `configs/v0_1_matrix/p3_d3_flow_mixc_endtrajkl.yaml`
- Create: `configs/v0_1_matrix/p4_sweep_{1,2,4,8,12}.yaml`
- Create: `scripts/generate_v0_1_configs.py`

---

## Task 1: Create base config templates

### Step 1: Create Mix B data components

Mix B = FineWeb-Edu + UltraChat SFT:

```yaml
mixture_components:
  - name: "fineweb_edu"
    dataset_name: "HuggingFaceFW/fineweb-edu"
    dataset_config: "sample-10BT"
    train_split: "train"
    val_split: "train"
    format_type: "plain_text"
    text_field: "text"
    train_samples: 12000
    val_samples: 600
  - name: "ultrachat_sft"
    dataset_name: "HuggingFaceH4/ultrachat_200k"
    dataset_config: "default"
    train_split: "train_sft"
    val_split: "test_sft"
    format_type: "chat_messages"
    messages_field: "messages"
    train_samples: 5000
    val_samples: 250
```

### Step 2: Create Mix A data components

Mix A = FineWeb-Edu only:

```yaml
mixture_components:
  - name: "fineweb_edu"
    dataset_name: "HuggingFaceFW/fineweb-edu"
    dataset_config: "sample-10BT"
    train_split: "train"
    val_split: "train"
    format_type: "plain_text"
    text_field: "text"
    train_samples: 12000
    val_samples: 600
```

### Step 3: Create Mix C data components

Mix C = Full mix (Mix B + MCQ datasets):

```yaml
mixture_components:
  - name: "fineweb_edu"
    dataset_name: "HuggingFaceFW/fineweb-edu"
    dataset_config: "sample-10BT"
    train_split: "train"
    val_split: "train"
    format_type: "plain_text"
    text_field: "text"
    train_samples: 12000
    val_samples: 600
  - name: "ultrachat_sft"
    dataset_name: "HuggingFaceH4/ultrachat_200k"
    dataset_config: "default"
    train_split: "train_sft"
    val_split: "test_sft"
    format_type: "chat_messages"
    messages_field: "messages"
    train_samples: 5000
    val_samples: 250
  - name: "arc_challenge"
    dataset_name: "allenai/ai2_arc"
    dataset_config: "ARC-Challenge"
    train_split: "train"
    val_split: "validation"
    format_type: "mcq_choices"
    use_chat_template: true
    question_field: "question"
    choices_field: "choices"
    answer_field: "answerKey"
    train_samples: 900
    val_samples: 120
  - name: "arc_easy"
    dataset_name: "allenai/ai2_arc"
    dataset_config: "ARC-Easy"
    train_split: "train"
    val_split: "validation"
    format_type: "mcq_choices"
    use_chat_template: true
    question_field: "question"
    choices_field: "choices"
    answer_field: "answerKey"
    train_samples: 1200
    val_samples: 150
  - name: "commonsense_qa"
    dataset_name: "tau/commonsense_qa"
    dataset_config: "default"
    train_split: "train"
    val_split: "validation"
    format_type: "mcq_choices"
    use_chat_template: true
    question_field: "question"
    choices_field: "choices"
    answer_field: "answerKey"
    train_samples: 1500
    val_samples: 150
  - name: "openbookqa"
    dataset_name: "allenai/openbookqa"
    dataset_config: "main"
    train_split: "train"
    val_split: "validation"
    format_type: "mcq_choices"
    use_chat_template: true
    question_field: "question_stem"
    choices_field: "choices"
    answer_field: "answerKey"
    train_samples: 900
    val_samples: 100
```

---

## Task 2: Generate Phase 1 configs (Architecture Sanity)

### P1-A1: One-shot projector, Mix B, End + KL

**Architecture:** `one_shot_projector` (A1)
**Loss:** End + KL (weights: endpoint=1.0, kl=0.5)
**Train T:** 1 (fixed, not random for one-shot)
**Eval T:** 1 (single pass)

Naming: `midflow_qwen_8to11_proj_mixb_endkl`

### P1-A2: Shared recurrent residual, Mix B, End + KL

**Architecture:** `shared_recurrent_residual` (A2)
**Loss:** End + KL
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8 (evaluated at these)

Naming: `midflow_qwen_8to11_rrb_mixb_endkl_trainT-r2468`

### P1-A3: Flow midblock, Mix B, End + KL

**Architecture:** `flow_midblock` (A3)
**Loss:** End + KL
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_endkl_trainT-r2468`

---

## Task 3: Generate Phase 2 configs (Loss Ablations)

Assuming Flow midblock won P1 (use as default for P2):

### P2-L1: Flow midblock, Mix B, End only

**Loss:** endpoint=1.0, trajectory=0, kl=0, ce=0
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_end_trainT-r2468`

### P2-L2: Flow midblock, Mix B, End + KL

**Loss:** endpoint=1.0, kl=0.5
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_endkl_trainT-r2468`

### P2-L3: Flow midblock, Mix B, End + Traj + KL

**Loss:** endpoint=1.0, trajectory=1.0, kl=0.5
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_endtrajkl_trainT-r2468`

### P2-L4: Flow midblock, Mix B, End + Traj + KL + CE

**Loss:** endpoint=1.0, trajectory=1.0, kl=0.5, ce=0.1
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_endtrajklce_trainT-r2468`

---

## Task 4: Generate Phase 3 configs (Data Mix Ablations)

Assuming End+Traj+KL won P2 (use as default for P3):

### P3-D1: Flow midblock, Mix A, End + Traj + KL

**Data:** Mix A (FineWeb only)
**Loss:** endpoint=1.0, trajectory=1.0, kl=0.5
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixa_endtrajkl_trainT-r2468`

### P3-D2: Flow midblock, Mix B, End + Traj + KL

**Data:** Mix B (FineWeb + UltraChat)
**Loss:** endpoint=1.0, trajectory=1.0, kl=0.5
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixb_endtrajkl_trainT-r2468`

### P3-D3: Flow midblock, Mix C, End + Traj + KL

**Data:** Mix C (Full mix)
**Loss:** endpoint=1.0, trajectory=1.0, kl=0.5
**Train T:** random from {2,4,6,8}
**Eval T:** 1,2,4,8

Naming: `midflow_qwen_8to11_flow_mixc_endtrajkl_trainT-r2468`

---

## Task 5: Generate Phase 4 configs (T Sweep)

Phase 4 evaluates the best model from P3 at multiple T values.
Create separate configs for each eval T (or use one config with multiple eval points).

Using separate configs for cleaner queue execution:

### P4 configs (eval at specific T values)

- P4-E1: Eval at T=1
- P4-E2: Eval at T=2
- P4-E3: Eval at T=4
- P4-E4: Eval at T=8
- P4-E5: Eval at T=12

All use the same best arch/loss/data from P3, just different eval T.

---

## Task 6: Common config settings

All configs share:

**Hardware Profile Settings:**
```yaml
seq_len: 1024
batch_size: 2  # microbatch from profile
accumulate_grad_batches: 8  # grad_accum from profile
precision: "bf16-mixed"
gradient_checkpointing: true
```

**Model Settings:**
```yaml
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_steps_T: 8
  train_T_values: [2, 4, 6, 8]  # or [1] for A1
  train_T_weights: [0.25, 0.25, 0.25, 0.25]
  step_embedding: "discrete"
  reuse_qwen_modules: true

replacement_model:
  start_layer: 8
  end_layer: 11
  depth: 4
  mlp_ratio: 4.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: true  # true for A2/A3, false for A1
  conditioning_mode: "timestep_plus_layer_boundary"
  init_strategy: "fresh"
```

**Training Settings:**
```yaml
train_loop:
  precision: "bf16-mixed"
  max_epochs: 3
  accumulate_grad_batches: 8
  sample_continuous_time: true
  log_every_n_steps: 10
  val_check_interval: 250
  gradient_checkpointing: true
```

**wandb Settings:**
```yaml
wandb:
  enabled: true
  project: "midflowlm-v0-1"
  entity: null
  tags: ["v0.1", "p1", "architecture"]
```

---

## Phase Completion Gate

After all tasks:
- [ ] All 15+ config files exist in `configs/v0_1_matrix/`
- [ ] All configs use consistent hardware profile
- [ ] All configs have unique experiment names
- [ ] Config generation script can regenerate all configs
- [ ] Commit with message "feat: v0.1 experiment matrix configs"

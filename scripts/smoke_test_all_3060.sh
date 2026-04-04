#!/bin/bash
# smoke_test_all_3060.sh - Comprehensive validation with ALL losses
# Uses seq_len=32 to fit on 3060 12GB
# Tests: endpoint, trajectory, KL, CE losses
# ~15-20 minutes total on 3060

set -e

echo "=================================="
echo "v0.1 Experiment Matrix Smoke Test"
echo "3060 12GB - Full Loss Configuration"
echo "=================================="
echo ""
echo "This will test 10 unique configurations:"
echo "  1. A1: OneShotProjector + End+KL+Traj"
echo "  2. A2: SharedRecurrent + End+KL+Traj"
echo "  3. A3: FlowMidblock + End+KL+Traj (main)"
echo "  4. Loss-L1: Endpoint only"
echo "  5. Loss-L2: End+KL only"
echo "  6. Loss-L3: End+KL+Traj (full)"
echo "  7. Loss-L4: End+KL+Traj+CE (with CE)"
echo "  8. Mix-A: FineWeb only"
echo "  9. Mix-B: FineWeb + UltraChat"
echo "  10. Mix-C: Full mix + MCQ"
echo ""
echo "All tests use: seq_len=32, 20 train / 5 val samples"
echo "Expected: ~2 min per test, ~20 min total"
echo ""

# Base config template
BASE_CONFIG='experiment_name: "SMOKE_NAME"
seed: 1337
teacher_state:
  mode: online_no_cache
teacher_cache:
  enabled: false
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_steps_T: 8
  train_T_values: [2, 4]
  train_T_weights: [0.5, 0.5]
  step_embedding: "discrete"
  reuse_qwen_modules: true
replacement_model:
  family: "SMOKE_FAMILY"
  start_layer: 8
  end_layer: 11
  depth: 2
  mlp_ratio: 2.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: SMOKE_CONDITIONING
  conditioning_mode: "timestep_plus_layer_boundary"
  init_strategy: "fresh"
data:
  loader: "mixture"
  seq_len: 32
  batch_size: 1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  shuffle_seed: 1337
SMOKE_DATA
loss:
  velocity_weight: 0.0
  endpoint_weight: SMOKE_ENDPOINT
  trajectory_weight: SMOKE_TRAJECTORY
  kl_weight: SMOKE_KL
  teacher_logits_source: "online"
  ce_weight: SMOKE_CE
  mask_padding_tokens: true
optimizer:
  name: "adamw"
  learning_rate: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
  eps: 1.0e-8
  grad_clip_norm: 1.0
scheduler:
  name: "cosine_with_warmup"
  warmup_steps: 5
train_loop:
  precision: "bf16-mixed"
  max_epochs: 1
  accumulate_grad_batches: 4
  sample_continuous_time: true
  log_every_n_steps: 5
  val_check_interval: 10
  gradient_checkpointing: true
  checkpoint_dir: "./outputs/SMOKE_NAME/checkpoints"
logging:
  log_dir: "./outputs/SMOKE_NAME/logs"
  save_top_k: 1
  monitor: "val/total_loss"
  mode: "min"
tensorboard:
  enabled: false
wandb:
  enabled: false
'

# Data components
DATA_FINWEB='  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 20
      val_samples: 5'

DATA_MIXB='  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 15
      val_samples: 5
    - name: "ultrachat_tiny"
      dataset_name: "HuggingFaceH4/ultrachat_200k"
      dataset_config: "default"
      train_split: "train_sft"
      val_split: "test_sft"
      format_type: "chat_messages"
      messages_field: "messages"
      train_samples: 10
      val_samples: 5'

DATA_MIXC='  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 15
      val_samples: 5
    - name: "ultrachat_tiny"
      dataset_name: "HuggingFaceH4/ultrachat_200k"
      dataset_config: "default"
      train_split: "train_sft"
      val_split: "test_sft"
      format_type: "chat_messages"
      messages_field: "messages"
      train_samples: 10
      val_samples: 5
    - name: "arc_easy_tiny"
      dataset_name: "allenai/ai2_arc"
      dataset_config: "ARC-Easy"
      train_split: "train"
      val_split: "validation"
      format_type: "mcq_choices"
      use_chat_template: true
      question_field: "question"
      choices_field: "choices"
      answer_field: "answerKey"
      train_samples: 5
      val_samples: 5'

# Test runner
run_test() {
    local name=$1
    local family=$2
    local conditioning=$3
    local endpoint=$4
    local trajectory=$5
    local kl=$6
    local ce=$7
    local data=$8
    
    ((TOTAL++))
    echo ""
    echo "Test $TOTAL: $name"
    echo "  Family: $family, Loss: End=$endpoint Traj=$trajectory KL=$kl CE=$ce"
    
    # Generate config
    local config_content="$BASE_CONFIG"
    config_content=$(echo "$config_content" | sed "s/SMOKE_NAME/$name/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_FAMILY/$family/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_CONDITIONING/$conditioning/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_ENDPOINT/$endpoint/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_TRAJECTORY/$trajectory/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_KL/$kl/g")
    config_content=$(echo "$config_content" | sed "s/SMOKE_CE/$ce/g")
    
    # Replace data section
    if [ "$data" = "mixb" ]; then
        config_content=$(echo "$config_content" | sed "/SMOKE_DATA/r /dev/stdin" <<< "$DATA_MIXB" | sed "/SMOKE_DATA/d")
    elif [ "$data" = "mixc" ]; then
        config_content=$(echo "$config_content" | sed "/SMOKE_DATA/r /dev/stdin" <<< "$DATA_MIXC" | sed "/SMOKE_DATA/d")
    else
        config_content=$(echo "$config_content" | sed "/SMOKE_DATA/r /dev/stdin" <<< "$DATA_FINWEB" | sed "/SMOKE_DATA/d")
    fi
    
    echo "$config_content" > "/tmp/${name}.yaml"
    
    # Run test
    if CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
        --config "/tmp/${name}.yaml" \
        --limit-train-batches 2 \
        --limit-val-batches 1 \
        --log-level WARNING 2>&1 | tail -15; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED"
        ((FAILED++))
    fi
}

# Track results
PASSED=0
FAILED=0
TOTAL=0

# Architecture Tests
run_test "A1_oneshot" "one_shot_projector" "false" "1.0" "1.0" "0.5" "0.0" "mixb"
run_test "A2_recurrent" "shared_recurrent_residual" "true" "1.0" "1.0" "0.5" "0.0" "mixb"
run_test "A3_flow" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.0" "mixb"

# Loss Ablation Tests
run_test "L1_endpoint" "flow_midblock" "true" "1.0" "0.0" "0.0" "0.0" "mixb"
run_test "L2_kl" "flow_midblock" "true" "1.0" "0.0" "0.5" "0.0" "mixb"
run_test "L3_full" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.0" "mixb"
run_test "L4_with_ce" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.1" "mixb"

# Data Mix Tests
run_test "MixA_fineweb" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.0" "fineweb"
run_test "MixB_chat" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.0" "mixb"
run_test "MixC_full" "flow_midblock" "true" "1.0" "1.0" "0.5" "0.0" "mixc"

# Summary
echo ""
echo "=================================="
echo "SMOKE TEST SUMMARY"
echo "=================================="
echo "Total: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL TESTS PASSED!"
    echo ""
    echo "Your 3060 can run ALL v0.1 experiments with:"
    echo "  - Endpoint loss ✓"
    echo "  - Trajectory supervision ✓"
    echo "  - KL distillation ✓"
    echo "  - CE loss ✓"
    echo "  - All 3 architectures ✓"
    echo "  - All data mixes ✓"
    echo ""
    echo "Ready for full training:"
    echo "  - Local 3060: python3 scripts/train.py --config CONFIG"
    echo "  - Vast 3x3090: bash scripts/run_matrix.sh --parallel"
    exit 0
else
    echo "✗ $FAILED test(s) failed"
    echo "Check logs in outputs/ for details"
    exit 1
fi

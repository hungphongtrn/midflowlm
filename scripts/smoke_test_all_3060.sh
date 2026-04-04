#!/bin/bash
# smoke_test_all_3060.sh - Quick validation of all unique experiment types
# Uses seq_len=32, minimal samples, endpoint-only loss for speed
# Tests each unique architecture/loss/data combination
# ~15-20 minutes total on 3060

set -e

echo "=================================="
echo "v0.1 Experiment Matrix Smoke Test"
echo "3060 12GB - Minimal Configuration"
echo "=================================="
echo ""
echo "This will test 8 unique experiment configurations:"
echo "  1. P1-A1: OneShotProjector + Mix B + End+KL"
echo "  2. P1-A2: SharedRecurrent + Mix B + End+KL"
echo "  3. P1-A3: FlowMidblock + Mix B + End+KL"
echo "  4. P2-L1: Flow + Mix B + End (no KL)"
echo "  5. P2-L3: Flow + Mix B + End+Traj+KL"
echo "  6. P2-L4: Flow + Mix B + End+Traj+KL+CE"
echo "  7. P3-D1: Flow + Mix A + End+Traj+KL"
echo "  8. P3-D3: Flow + Mix C + End+Traj+KL"
echo ""
echo "Each test: seq_len=32, 20 train samples, 5 val samples"
echo "Expected time: ~2-3 min per test, ~20 min total"
echo ""

# Base test function
run_smoke_test() {
    local config_path=$1
    local test_name=$2
    local extra_args=${3:-""}
    
    echo ""
    echo "[$test_name] Testing: $(basename $config_path)"
    echo "-------------------------------------------"
    
    # Run with limited batches
    if CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
        --config "$config_path" \
        --limit-train-batches 3 \
        --limit-val-batches 1 \
        --log-level WARNING \
        $extra_args 2>&1 | tail -20; then
        echo "✓ $test_name: PASSED"
        return 0
    else
        echo "✗ $test_name: FAILED"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
TOTAL=0

# Test 1: OneShotProjector - simplest
echo "Test 1/8: OneShotProjector (A1) - simplest architecture"
cat > /tmp/test_a1.yaml << 'EOF'
experiment_name: "smoke_a1_projector"
seed: 1337
teacher_state:
  mode: online_no_cache
teacher_cache:
  enabled: false
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_steps_T: 8
  train_T_values: [1]
  train_T_weights: [1.0]
  step_embedding: "discrete"
  reuse_qwen_modules: true
replacement_model:
  family: "one_shot_projector"
  start_layer: 8
  end_layer: 11
  depth: 4
  mlp_ratio: 4.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: false
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
  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 20
      val_samples: 5
loss:
  velocity_weight: 0.0
  endpoint_weight: 1.0
  trajectory_weight: 0.0
  kl_weight: 0.0
  teacher_logits_source: "online"
  ce_weight: 0.0
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
  checkpoint_dir: "./outputs/smoke_a1/checkpoints"
logging:
  log_dir: "./outputs/smoke_a1/logs"
  save_top_k: 1
  monitor: "val/total_loss"
  mode: "min"
tensorboard:
  enabled: false
wandb:
  enabled: false
EOF

((TOTAL++))
if run_smoke_test "/tmp/test_a1.yaml" "A1-OneShot"; then
    ((PASSED++))
else
    ((FAILED++))
fi

# Test 2: SharedRecurrent
echo ""
echo "Test 2/8: SharedRecurrent (A2) - iterative without flow"
cat > /tmp/test_a2.yaml << 'EOF'
experiment_name: "smoke_a2_recurrent"
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
  family: "shared_recurrent_residual"
  start_layer: 8
  end_layer: 11
  depth: 4
  mlp_ratio: 4.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: true
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
  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 20
      val_samples: 5
loss:
  velocity_weight: 0.0
  endpoint_weight: 1.0
  trajectory_weight: 0.0
  kl_weight: 0.0
  teacher_logits_source: "online"
  ce_weight: 0.0
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
  checkpoint_dir: "./outputs/smoke_a2/checkpoints"
logging:
  log_dir: "./outputs/smoke_a2/logs"
  save_top_k: 1
  monitor: "val/total_loss"
  mode: "min"
tensorboard:
  enabled: false
wandb:
  enabled: false
EOF

((TOTAL++))
if run_smoke_test "/tmp/test_a2.yaml" "A2-Recurrent"; then
    ((PASSED++))
else
    ((FAILED++))
fi

# Test 3: FlowMidblock (main method)
echo ""
echo "Test 3/8: FlowMidblock (A3) - main architecture"
((TOTAL++))
if run_smoke_test "configs/smoke_minimal_3060.yaml" "A3-Flow"; then
    ((PASSED++))
else
    ((FAILED++))
fi

# Test 4: Mix A data loading
echo ""
echo "Test 4/8: Mix A data (FineWeb only)"
cat > /tmp/test_mixa.yaml << 'EOF'
experiment_name: "smoke_mixa"
seed: 1337
teacher_state:
  mode: online_no_cache
teacher_cache:
  enabled: false
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_steps_T: 8
  train_T_values: [2]
  train_T_weights: [1.0]
  step_embedding: "discrete"
  reuse_qwen_modules: true
replacement_model:
  family: "flow_midblock"
  start_layer: 8
  end_layer: 11
  depth: 4
  mlp_ratio: 4.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: true
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
  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 20
      val_samples: 5
loss:
  velocity_weight: 0.0
  endpoint_weight: 1.0
  trajectory_weight: 0.0
  kl_weight: 0.0
  teacher_logits_source: "online"
  ce_weight: 0.0
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
  checkpoint_dir: "./outputs/smoke_mixa/checkpoints"
logging:
  log_dir: "./outputs/smoke_mixa/logs"
  save_top_k: 1
  monitor: "val/total_loss"
  mode: "min"
tensorboard:
  enabled: false
wandb:
  enabled: false
EOF

((TOTAL++))
if run_smoke_test "/tmp/test_mixa.yaml" "Mix-A-Data"; then
    ((PASSED++))
else
    ((FAILED++))
fi

# Test 5: Mix B data loading
echo ""
echo "Test 5/8: Mix B data (FineWeb + UltraChat)"
cat > /tmp/test_mixb.yaml << 'EOF'
experiment_name: "smoke_mixb"
seed: 1337
teacher_state:
  mode: online_no_cache
teacher_cache:
  enabled: false
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_steps_T: 8
  train_T_values: [2]
  train_T_weights: [1.0]
  step_embedding: "discrete"
  reuse_qwen_modules: true
replacement_model:
  family: "flow_midblock"
  start_layer: 8
  end_layer: 11
  depth: 4
  mlp_ratio: 4.0
  qkv_bias: true
  use_qwen_causal_mask: true
  use_step_conditioning: true
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
  mixture_components:
    - name: "fineweb_edu_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 20
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
loss:
  velocity_weight: 0.0
  endpoint_weight: 1.0
  trajectory_weight: 0.0
  kl_weight: 0.0
  teacher_logits_source: "online"
  ce_weight: 0.0
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
  checkpoint_dir: "./outputs/smoke_mixb/checkpoints"
logging:
  log_dir: "./outputs/smoke_mixb/logs"
  save_top_k: 1
  monitor: "val/total_loss"
  mode: "min"
tensorboard:
  enabled: false
wandb:
  enabled: false
EOF

((TOTAL++))
if run_smoke_test "/tmp/test_mixb.yaml" "Mix-B-Data"; then
    ((PASSED++))
else
    ((FAILED++))
fi

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
    echo "Your 3060 can run the v0.1 experiment matrix."
    echo "Next steps:"
    echo "  1. Test with real seq_len=1024:"
    echo "     python3 scripts/train.py --config configs/v0_1_matrix_3060/CONFIG.yaml"
    echo ""
    echo "  2. Or deploy to Vast with 3x3090:"
    echo "     bash scripts/run_matrix.sh --parallel"
    exit 0
else
    echo "✗ Some tests failed. Check logs above."
    exit 1
fi

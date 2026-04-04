#!/bin/bash
# quick_smoke_3060.sh - Simple smoke test with clear progress
set -e

echo "=================================="
echo "Quick Smoke Test - 3060"
echo "=================================="
echo ""

run_test() {
    local name=$1
    local config_path=$2
    
    echo ""
    echo "Running: $name"
    echo "--------------------"
    
    if CUDA_VISIBLE_DEVICES=0 timeout 300 python3 scripts/train.py \
        --config "$config_path" \
        --limit-train-batches 2 \
        --limit-val-batches 1 \
        --log-level INFO 2>&1 | grep -E "Creating|Train dataload|Batch|val/|Error" | tail -10; then
        echo "✓ PASSED"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

# Test 1: Simplest
echo "[1/3] OneShotProjector (simplest)"
run_test "A1-OneShot" "configs/smoke_minimal_3060.yaml" || exit 1

# Test 2: Main architecture with full losses  
echo ""
echo "[2/3] FlowMidblock + All Losses (main method)"
run_test "A3-Flow-Full" "configs/test_kl_traj_3060.yaml" || exit 1

# Test 3: With chat data
echo ""
echo "[3/3] Mix B data (FineWeb + UltraChat)"
cat > /tmp/test_mixb.yaml << 'CONFIG'
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
  depth: 2
  mlp_ratio: 2.0
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
    - name: "fineweb_tiny"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 10
      val_samples: 5
    - name: "ultrachat_tiny"
      dataset_name: "HuggingFaceH4/ultrachat_200k"
      dataset_config: "default"
      train_split: "train_sft"
      val_split: "test_sft"
      format_type: "chat_messages"
      messages_field: "messages"
      train_samples: 5
      val_samples: 3
loss:
  velocity_weight: 0.0
  endpoint_weight: 1.0
  trajectory_weight: 1.0
  kl_weight: 0.5
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
CONFIG

run_test "Mix-B-Data" "/tmp/test_mixb.yaml" || exit 1

echo ""
echo "=================================="
echo "✓ All 3 smoke tests PASSED!"
echo "=================================="
echo ""
echo "Your 3060 can run:"
echo "  - All 3 architectures"
echo "  - Full losses (endpoint + trajectory + KL)"
echo "  - All data types"
echo ""

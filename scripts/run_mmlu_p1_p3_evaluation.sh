#!/bin/bash
# MMLU-Pro Evaluation Script for P1-P3 Checkpoints
# This script downloads checkpoints from HF Hub and runs side-by-side MMLU-Pro evaluation

set -e

# Configuration
LOCAL_DIR="./models"
RESULTS_DIR="./results"
NUM_SAMPLES=72
MAX_NEW_TOKENS=256

# Create directories
mkdir -p "$LOCAL_DIR" "$RESULTS_DIR"

echo "=========================================="
echo "MMLU-Pro Evaluation - P1 to P3"
echo "=========================================="

# Step 1: Download all P1-P3 checkpoints if not already present
echo ""
echo "Step 1: Checking/Downloading checkpoints from HF Hub..."
echo "   Repository: hungphongtrn/midflowlm-phase1"
echo ""

# Function to download a checkpoint
download_checkpoint() {
    local exp_key=$1
    echo "   Checking $exp_key..."
    if [ ! -f "$LOCAL_DIR/${exp_key}/checkpoint.pth" ]; then
        echo "   -> Downloading $exp_key..."
        uv run python scripts/push_checkpoints_to_hf.py --download --${exp_key//_/-} --local-dir "$LOCAL_DIR" || {
            echo "   WARNING: Failed to download $exp_key"
            return 1
        }
    else
        echo "   -> $exp_key already exists"
    fi
    return 0
}

# Download P1 checkpoints
for exp in p1_a1 p1_a2 p1_a3; do
    download_checkpoint "$exp" || true
done

# Download P2 checkpoints  
for exp in p2_l1 p2_l2 p2_l3 p2_l4; do
    download_checkpoint "$exp" || true
done

# Download P3 checkpoints
for exp in p3_d1 p3_d2 p3_d3; do
    download_checkpoint "$exp" || true
done

echo ""
echo "Step 2: Running MMLU-Pro Evaluation..."
echo "=========================================="

# Define config paths
CONFIG_P1_A1="configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml"
CONFIG_P1_A2="configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml"
CONFIG_P1_A3="configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml"
CONFIG_P2="configs/v0_1_matrix/midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT_r2468.yaml"
CONFIG_P3_D1="configs/v0_1_matrix/midflow_qwen_8to11_p3_d1_flow_mixa_endtrajkl_trainT_r2468.yaml"
CONFIG_P3_D2="configs/v0_1_matrix/midflow_qwen_8to11_p3_d2_flow_mixb_endtrajkl_trainT_r2468.yaml"
CONFIG_P3_D3="configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml"

# Base output file
BASE_OUTPUT="$RESULTS_DIR/mmlu_pro_p1_p3_comparison.csv"

# ============================================
# PHASE 1: Architecture Comparison
# ============================================
echo ""
echo "PHASE 1: Architecture Experiments"
echo "-----------------------------------"

# P1-A1: One-shot Projector (T=1 only)
if [ -f "$LOCAL_DIR/p1_a1_projector/checkpoint.pth" ]; then
    echo "Evaluating P1-A1 (Projector, T=1)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P1_A1" \
        --checkpoint "$LOCAL_DIR/p1_a1_projector/checkpoint.pth" \
        --num-steps 1 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P1-A1" \
        --output "$RESULTS_DIR/p1_a1_results.json"
    echo "   ✓ P1-A1 complete"
else
    echo "   ⚠ P1-A1 checkpoint not found, skipping"
fi

# P1-A2: Shared Recurrent Residual (T=1,2,4,8)
if [ -f "$LOCAL_DIR/p1_a2_recurrent_residual/checkpoint.pth" ]; then
    echo "Evaluating P1-A2 (Recurrent Residual, T=1,2,4,8)..."
    uv run python python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P1_A2" \
        --checkpoint "$LOCAL_DIR/p1_a2_recurrent_residual/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P1-A2" \
        --output "$RESULTS_DIR/p1_a2_results.json"
    echo "   ✓ P1-A2 complete"
else
    echo "   ⚠ P1-A2 checkpoint not found, skipping"
fi

# P1-A3: Flow Midblock (T=1,2,4,8)
if [ -f "$LOCAL_DIR/p1_a3_flow_midblock/checkpoint.pth" ]; then
    echo "Evaluating P1-A3 (Flow Midblock, T=1,2,4,8)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P1_A3" \
        --checkpoint "$LOCAL_DIR/p1_a3_flow_midblock/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P1-A3" \
        --output "$RESULTS_DIR/p1_a3_results.json"
    echo "   ✓ P1-A3 complete"
else
    echo "   ⚠ P1-A3 checkpoint not found, skipping"
fi

# ============================================
# PHASE 2: Loss Ablation
# ============================================
echo ""
echo "PHASE 2: Loss Ablation Experiments"
echo "-----------------------------------"

# P2-L1: Endpoint-only Loss
if [ -f "$LOCAL_DIR/p2_l1_endpoint_only/checkpoint.pth" ]; then
    echo "Evaluating P2-L1 (Endpoint-only)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P2" \
        --checkpoint "$LOCAL_DIR/p2_l1_endpoint_only/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P2-L1" \
        --output "$RESULTS_DIR/p2_l1_results.json"
    echo "   ✓ P2-L1 complete"
else
    echo "   ⚠ P2-L1 checkpoint not found, skipping"
fi

# P2-L2: End + KL Loss
if [ -f "$LOCAL_DIR/p2_l2_end_kl/checkpoint.pth" ]; then
    echo "Evaluating P2-L2 (End + KL)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P2" \
        --checkpoint "$LOCAL_DIR/p2_l2_end_kl/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P2-L2" \
        --output "$RESULTS_DIR/p2_l2_results.json"
    echo "   ✓ P2-L2 complete"
else
    echo "   ⚠ P2-L2 checkpoint not found, skipping"
fi

# P2-L3: End + Traj + KL (Best)
if [ -f "$LOCAL_DIR/p2_l3_end_traj_kl/checkpoint.pth" ]; then
    echo "Evaluating P2-L3 (End + Traj + KL - BEST)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P2" \
        --checkpoint "$LOCAL_DIR/p2_l3_end_traj_kl/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P2-L3-BEST" \
        --output "$RESULTS_DIR/p2_l3_results.json"
    echo "   ✓ P2-L3 complete"
else
    echo "   ⚠ P2-L3 checkpoint not found, skipping"
fi

# P2-L4: End + Traj + KL + CE
if [ -f "$LOCAL_DIR/p2_l4_end_traj_kl_ce/checkpoint.pth" ]; then
    echo "Evaluating P2-L4 (End + Traj + KL + CE)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P2" \
        --checkpoint "$LOCAL_DIR/p2_l4_end_traj_kl_ce/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P2-L4" \
        --output "$RESULTS_DIR/p2_l4_results.json"
    echo "   ✓ P2-L4 complete"
else
    echo "   ⚠ P2-L4 checkpoint not found, skipping"
fi

# ============================================
# PHASE 3: Data Mix Ablation
# ============================================
echo ""
echo "PHASE 3: Data Mix Experiments"
echo "-----------------------------------"

# P3-D1: Mix A (FineWeb only)
if [ -f "$LOCAL_DIR/p3_d1_mix_a/checkpoint.pth" ]; then
    echo "Evaluating P3-D1 (Mix A - FineWeb only)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P3_D1" \
        --checkpoint "$LOCAL_DIR/p3_d1_mix_a/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P3-D1-MixA" \
        --output "$RESULTS_DIR/p3_d1_results.json"
    echo "   ✓ P3-D1 complete"
else
    echo "   ⚠ P3-D1 checkpoint not found, skipping"
fi

# P3-D2: Mix B (FineWeb + UltraChat)
if [ -f "$LOCAL_DIR/p3_d2_mix_b/checkpoint.pth" ]; then
    echo "Evaluating P3-D2 (Mix B - FineWeb + UltraChat)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P3_D2" \
        --checkpoint "$LOCAL_DIR/p3_d2_mix_b/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P3-D2-MixB" \
        --output "$RESULTS_DIR/p3_d2_results.json"
    echo "   ✓ P3-D2 complete"
else
    echo "   ⚠ P3-D2 checkpoint not found, skipping"
fi

# P3-D3: Mix C (Full)
if [ -f "$LOCAL_DIR/p3_d3_mix_c/checkpoint.pth" ]; then
    echo "Evaluating P3-D3 (Mix C - Full)..."
    uv run python scripts/eval_mmlu_pro.py \
        --config "$CONFIG_P3_D3" \
        --checkpoint "$LOCAL_DIR/p3_d3_mix_c/checkpoint.pth" \
        --num-steps 1 2 4 8 \
        --num-samples $NUM_SAMPLES \
        --max-new-tokens $MAX_NEW_TOKENS \
        --csv-output "$BASE_OUTPUT" \
        --experiment-id "P3-D3-MixC" \
        --output "$RESULTS_DIR/p3_d3_results.json"
    echo "   ✓ P3-D3 complete"
else
    echo "   ⚠ P3-D3 checkpoint not found, skipping"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  CSV: $BASE_OUTPUT"
echo "  JSON details: $RESULTS_DIR/*.json"
echo ""
echo "To view the CSV:"
echo "  cat $BASE_OUTPUT | column -t -s, | less -S"
echo ""
echo "To analyze in Python:"
echo "  python -c \"import pandas as pd; df = pd.read_csv('$BASE_OUTPUT'); print(df.to_string())\""
echo ""

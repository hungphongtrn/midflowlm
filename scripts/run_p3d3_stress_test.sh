#!/bin/bash
# P3-D3 Stress Test: MMLU-Pro T-Sweep + Teacher Baseline
# ==============================================================
# 100 balanced samples (10 per answer choice A-J)
# T values: 1, 2, 8, 16, 32
# Teacher baseline vs P3-D3
set -e

CONFIG="configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml"
CHECKPOINT_DIR="./models/p3_d3_mix_c"
CHECKPOINT="$CHECKPOINT_DIR/checkpoint.pth"
RESULTS_DIR="./results/stress_test"
NUM_SAMPLES=100
MAX_NEW_TOKENS=256

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"

echo "=========================================="
echo "P3-D3 STRESS TEST"
echo "=========================================="
echo "Benchmark: MMLU-Pro"
echo "Samples: $NUM_SAMPLES (balanced: 10 per answer choice A-J)"
echo "T values: 1, 2, 8, 16, 32"
echo "Baselines: Teacher (full Qwen3.5-0.8B)"
echo "=========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "Downloading P3-D3 checkpoint from HuggingFace Hub..."
    echo "Repository: hungphongtrn/midflowlm"
    uv run python scripts/push_checkpoints_to_hf.py --download --p3-d3 --local-dir ./models || {
        echo "ERROR: Failed to download checkpoint."
        echo "Download manually: huggingface-cli download hungphongtrn/midflowlm p3_d3_mix_c/checkpoint.pth --local-dir $CHECKPOINT_DIR"
        exit 1
    }
fi

echo ""
echo "=========================================="
echo "PHASE 1: MMLU-Pro T-Sweep"
echo "=========================================="
uv run python scripts/eval_mmlu_pro.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --num-steps 1 2 8 16 32 \
    --num-samples $NUM_SAMPLES \
    --balanced-sampling \
    --max-new-tokens $MAX_NEW_TOKENS \
    --output "$RESULTS_DIR/mmlu_pro_results.json" \
    --csv-output "$RESULTS_DIR/mmlu_pro.csv" \
    --experiment-id "P3-D3-Stress"

echo ""
echo "=========================================="
echo "PHASE 2: Analysis"
echo "=========================================="
uv run python scripts/analyze_stress_test.py \
    --results-dir "$RESULTS_DIR" \
    --csv-output "$RESULTS_DIR/summary.csv"

echo ""
echo "=========================================="
echo "P3-D3 STRESS TEST COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "  $RESULTS_DIR/mmlu_pro_results.json  - Detailed per-question results"
echo "  $RESULTS_DIR/mmlu_pro.csv           - T-sweep summary CSV"
echo "  $RESULTS_DIR/summary.csv            - Analysis summary"
echo ""

echo "=========================================="
echo "SUCCESS CRITERIA CHECK"
echo "=========================================="

if [ -f "$RESULTS_DIR/mmlu_pro_results.json" ]; then
    echo "PASS: MMLU-Pro T-sweep results generated"
else
    echo "FAIL: MMLU-Pro T-sweep results missing"
fi

if [ -f "$RESULTS_DIR/mmlu_pro.csv" ]; then
    echo "PASS: MMLU-Pro CSV summary generated"
else
    echo "FAIL: MMLU-Pro CSV summary missing"
fi

if [ -f "$RESULTS_DIR/summary.csv" ]; then
    echo "PASS: Analysis summary generated"
else
    echo "FAIL: Analysis summary missing"
fi

echo "=========================================="
#!/bin/bash
# test_3060.sh - Quick smoke test on local 3060
# Usage: bash scripts/test_3060.sh

set -e

echo "=================================="
echo "MidflowLM v0.1 - 3060 Smoke Test"
echo "=================================="
echo ""

# Check GPU
echo "Step 1: Checking GPU..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No GPU detected!')
    exit(1)
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  ✓ {gpu} ({mem:.1f} GB)')
if mem < 10e9:
    print('  WARNING: Less than 10GB VRAM detected')
"
echo ""

# Check dependencies
echo "Step 2: Checking dependencies..."
python3 -c "import torch; import transformers; import wandb; import yaml" 2>/dev/null || {
    echo "  Missing dependencies. Installing..."
    pip install -q -r requirements.txt
}
echo "  ✓ Dependencies OK"
echo ""

# Check wandb
echo "Step 3: Checking wandb..."
if ! wandb whoami > /dev/null 2>&1; then
    echo "  Please login to wandb:"
    wandb login
else
    echo "  ✓ wandb logged in as: $(wandb whoami)"
fi
echo ""

# Check huggingface
echo "Step 4: Checking HuggingFace..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "  Please login to HuggingFace:"
    huggingface-cli login
else
    echo "  ✓ HuggingFace: $(huggingface-cli whoami)"
fi
echo ""

# Run smoke test
echo "Step 5: Running smoke test..."
echo "  Config: configs/smoke_test_3060.yaml"
echo "  This will take ~5-10 minutes on 3060"
echo ""

# Monitor GPU in background
(
    for i in {1..60}; do
        sleep 10
        python3 -c "
import torch
if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f'  GPU Memory: Alloc={alloc:.1f}GB, Reserved={reserved:.1f}GB')
" 2>/dev/null || true
    done
) &
MONITOR_PID=$!

# Run training
if CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
    --config configs/smoke_test_3060.yaml \
    --log-level INFO; then
    echo ""
    echo "  ✓ Smoke test PASSED"
    kill $MONITOR_PID 2>/dev/null || true
else
    echo ""
    echo "  ✗ Smoke test FAILED"
    kill $MONITOR_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=================================="
echo "Smoke Test Complete!"
echo "=================================="
echo ""
echo "Your 3060 setup works. You can now:"
echo "  1. Run a single experiment:"
echo "     CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config CONFIG"
echo ""
echo "  2. Run full matrix on 3060 (sequential, will take ~40-60 hours):"
echo "     bash scripts/run_matrix.sh"
echo ""
echo "  3. Deploy to Vast for faster parallel execution"
echo ""
echo "Monitor at: https://wandb.ai"
echo ""

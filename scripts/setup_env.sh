#!/bin/bash
# setup_env.sh - One-time environment setup for v0.1 experiments
# Usage: bash scripts/setup_env.sh

set -e

echo "=================================="
echo "MidflowLM v0.1 Environment Setup"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this from the repo root directory"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "Step 2: Setup Weights & Biases (wandb)..."
echo "You'll need a wandb API key from: https://wandb.ai/authorize"
echo ""
if ! wandb login 2>/dev/null; then
    echo "Please run: wandb login"
    echo "Then re-run this script"
    exit 1
fi
echo "✓ wandb configured"
echo ""

echo "Step 3: Setup Hugging Face token..."
echo "You'll need a token from: https://huggingface.co/settings/tokens"
echo "(Token needs 'read' access for model downloads)"
echo ""
if ! huggingface-cli whoami 2>/dev/null; then
    echo "Running: huggingface-cli login"
    huggingface-cli login
else
    echo "Already logged in as: $(huggingface-cli whoami)"
fi
echo "✓ Hugging Face configured"
echo ""

echo "Step 4: Verify GPU access..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: No GPUs detected!')
    exit(1)
gpus = torch.cuda.device_count()
print(f'✓ Found {gpus} GPU(s):')
for i in range(gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
"
echo ""

echo "Step 5: Create output directories..."
mkdir -p outputs logs
chmod +x scripts/run_matrix.sh
echo "✓ Directories created"
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Review configs: ls configs/v0_1_matrix/"
echo "  2. Run experiments: bash scripts/run_matrix.sh"
echo "  3. Monitor at: https://wandb.ai"
echo ""

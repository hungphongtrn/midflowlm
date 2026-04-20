#!/bin/bash
# Setup and run sequential experiments on RTX 3090
# Run this on the rented 3090 instance

set -e

echo "=== MidflowLM v0.1 Experiment Matrix Setup ==="
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "Setting up Python environment..."
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch for CUDA 12.4 (compatible with CUDA 12.7 driver)
echo "Installing PyTorch and dependencies..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
uv pip install transformers datasets torchmetrics torchdiffeq einops pyyaml safetensors numpy tqdm accelerate wandb --upgrade

# Check for wandb authentication
echo ""
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set!"
    echo "Options to authenticate:"
    echo "  1. Set env var: export WANDB_API_KEY=your_key_here"
    echo "  2. Run: wandb login"
    echo ""
    echo "Continuing without wandb authentication..."
    echo "(experiments will still log to local files and tensorboard)"
else
    echo "WANDB_API_KEY detected - wandb will be enabled"
fi

echo ""
echo "=== Environment ready ==="
echo "Starting sequential experiment run..."
echo ""

# Run the matrix sequentially
bash scripts/run_matrix.sh --sequential

echo ""
echo "=== All experiments complete ==="
echo "Check logs/matrix_*/ for results"
echo "Check wandb for experiment tracking"

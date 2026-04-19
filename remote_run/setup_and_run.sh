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

# Install PyTorch for CUDA 12.1 (adjust for 3090)
echo "Installing PyTorch and dependencies..."
uv pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers>=4.57.0 datasets>=2.18.0 torchmetrics>=1.4.0
uv pip install torchdiffeq>=0.2.5 einops>=0.8.0 pyyaml>=6.0
uv pip install safetensors>=0.4.0 numpy>=1.26.0 tqdm accelerate

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

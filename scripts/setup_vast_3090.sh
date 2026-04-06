#!/bin/bash
# setup_vast_3090.sh - Optimize 3x3090 Vast instance for maximum performance
# Run this once when you first start your Vast instance

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║    Vast.ai 3x3090 Instance Optimization Setup             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect hardware
echo -e "${BLUE}Detecting hardware...${NC}"
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CPU_CORES=$(nproc)
TOTAL_RAM=$(free -g | grep Mem | awk '{print $2}')

echo "  GPUs: $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  CPU cores: $CPU_CORES"
echo "  RAM: ${TOTAL_RAM}GB"
echo ""

# Install essential tools
echo -e "${BLUE}Installing essential tools...${NC}"
apt-get update -qq
apt-get install -y -qq \
    git-lfs \
    tmux \
    htop \
    nvtop \
    iotop \
    rsync \
    aria2 \
    pigz

echo -e "${GREEN}✓ Tools installed${NC}"
echo ""

# Setup workspace directories
echo -e "${BLUE}Setting up workspace directories...${NC}"
mkdir -p /workspace/.cache/huggingface/datasets
mkdir -p /workspace/.cache/huggingface/modules
mkdir -p /workspace/.cache/pip
mkdir -p /workspace/.cache/torch
mkdir -p /workspace/outputs
mkdir -p /workspace/logs

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Configure environment variables for maximum parallelism
echo -e "${BLUE}Configuring environment for maximum parallelism...${NC}"

# Calculate optimal thread allocation
THREADS_PER_GPU=$((CPU_CORES / NUM_GPUS))

cat >> ~/.bashrc << 'EOF'

# ============================================
# MidflowLM Performance Optimization
# ============================================

# HuggingFace Datasets - Maximum parallelism
export HF_DATASETS_PARALLELISM=true
export HF_DATASETS_DOWNLOAD_PARALLELISM=8
export HF_DATASETS_DOWNLOAD_CONCURRENCY=8
export HF_DATASETS_DISABLE_TELEMETRY=1

# Tokenizers - Enable parallel processing
export TOKENIZERS_PARALLELISM=true

# PyTorch - Thread allocation per process
EOF

echo "export OMP_NUM_THREADS=$THREADS_PER_GPU" >> ~/.bashrc
echo "export MKL_NUM_THREADS=$THREADS_PER_GPU" >> ~/.bashrc
echo "export NUMEXPR_NUM_THREADS=$THREADS_PER_GPU" >> ~/.bashrc

cat >> ~/.bashrc << 'EOF'

# Cache locations (fast SSD)
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip
export TORCH_HOME=/workspace/.cache/torch

# Python optimization
export PYTHONUNBUFFERED=1

# WandB (set your key)
export WANDB_API_KEY=""
export WANDB_PROJECT="midflowlm-v0-1"

# Disable unnecessary warnings
export TOKENIZERS_PARALLELISM=true
EOF

echo -e "${GREEN}✓ Environment configured${NC}"
echo "  Threads per GPU: $THREADS_PER_GPU"
echo "  Cache location: /workspace/.cache/"
echo ""

# Clone repository if not exists
if [ ! -d "/workspace/midflowlm" ]; then
    echo -e "${BLUE}Cloning midflowlm repository...${NC}"
    cd /workspace
    git clone https://github.com/hungphongtrn/midflowlm.git
    echo -e "${GREEN}✓ Repository cloned${NC}"
else
    echo -e "${YELLOW}Repository already exists at /workspace/midflowlm${NC}"
fi
echo ""

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
cd /workspace/midflowlm

# Upgrade pip for faster installs
pip install --upgrade pip -q

# Install with cache in workspace (faster reinstalls)
pip install -r requirements.txt --cache-dir /workspace/.cache/pip -q

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Pre-download teacher model (saves time during experiments)
echo -e "${BLUE}Pre-downloading teacher model...${NC}"
python3 << 'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set cache dir
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

print("Downloading Qwen/Qwen3.5-0.8B...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype="auto",
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

print("Model cached successfully!")
PYEOF

echo -e "${GREEN}✓ Teacher model cached${NC}"
echo ""

# Generate configs with optimal settings
echo -e "${BLUE}Generating optimized experiment configs...${NC}"
python3 scripts/generate_v0_1_configs.py
echo -e "${GREEN}✓ Configs generated${NC}"
echo ""

# Display summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Setup Complete - Next Steps                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Your 3x3090 instance is now optimized!${NC}"
echo ""
echo "Next steps:"
echo "  1. Set your WandB API key:"
echo "     export WANDB_API_KEY=your_key_here"
echo ""
echo "  2. Start the experiment matrix:"
echo "     bash scripts/run_matrix.sh --parallel"
echo ""
echo "  3. Or run a single test experiment:"
echo "     python3 scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml"
echo ""
echo "Performance settings:"
echo "  - Dataloader workers: Auto-detected from CPU cores"
echo "  - Prefetch factor: 4 batches per worker"
echo "  - Persistent workers: Enabled"
echo "  - Pin memory: Enabled"
echo "  - HF datasets parallelism: Enabled"
echo ""
echo "Monitor with:"
echo "  - htop: CPU usage"
echo "  - nvtop: GPU usage"
echo "  - iotop: Disk I/O"
echo "  - wandb: https://wandb.ai (after setting API key)"
echo ""
echo "Note: Source ~/.bashrc or restart shell to load environment variables"

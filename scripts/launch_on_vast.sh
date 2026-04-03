#!/bin/bash
# launch_on_vast.sh - Template for launching on Vast.ai
# Usage: Copy commands from this file to Vast instance

set -e

echo "MidflowLM v0.1 - Vast Launch Template"
echo "======================================"
echo ""

# Configuration
REPO_URL="https://github.com/YOUR_USERNAME/midflowlm.git"  # UPDATE THIS
INSTANCE_TYPE="3x RTX 3090"  # Or your preferred GPU setup

cat << 'EOF'

## Step 1: Rent Instance on Vast

Go to: https://vast.ai/console/create/

Filter:
- GPU: RTX 3090 (3x)
- Image: pytorch/pytorch:latest
- Disk: 100 GB minimum
- On-start script: (leave blank, we'll set up manually)

## Step 2: SSH into Instance

Once instance is running:

```bash
ssh root@<INSTANCE_IP> -p <PORT>
```

## Step 3: Quick Setup Commands

Run these commands on the instance:

EOF

cat << EOF
# Update and install basics
apt-get update && apt-get install -y git tmux htop

# Clone repo
cd /workspace
git clone $REPO_URL
cd midflowlm

# Install dependencies
pip install -q -r requirements.txt

# Login to wandb and huggingface
wandb login
huggingface-cli login

# Test single GPU
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \\
    --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml \\
    --fast-dev-run

# If test passes, run full matrix
bash scripts/run_matrix.sh --parallel
EOF

cat << 'EOF'

## Step 4: Detach and Monitor

Use tmux to keep running after SSH disconnect:

```bash
tmux new -s midflow
bash scripts/run_matrix.sh --parallel
# Press Ctrl+B then D to detach
```

Reattach later:
```bash
tmux attach -t midflow
```

## Step 5: Monitor via wandb

Open: https://wandb.ai

All experiments log automatically with experiment names.

## Useful Commands

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# View running processes
ps aux | grep train.py

# Check experiment status
cat .experiment_status

# View logs
tail -f logs/*.log

# Restart failed experiments only
bash scripts/run_matrix.sh --parallel
```

## Estimated Costs

With 3x RTX 3090:
- ~$0.50-1.00/hour depending on provider
- ~15-20 hours for full matrix
- Total: ~$10-20

## Troubleshooting

**"No space left on device"**
```bash
# Clean up
rm -rf outputs/*/checkpoints/epoch_*.ckpt  # Keep only best
```

**"CUDA out of memory"**
```bash
# Reduce batch size in profile
sed -i 's/"batch_size": 2/"batch_size": 1/' profiles/v0_1_3090_profile.json
```

**Runner stopped unexpectedly**
```bash
# Check logs
ls -lt logs/ | head

# Resume
bash scripts/run_matrix.sh --parallel
```

EOF

echo ""
echo "======================================"
echo "Update REPO_URL in this script before using!"
echo "======================================"

#!/bin/bash
# Smoke test launcher: prepare data + train on RTX 3060
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="/home/hungphongtrn/Workspace/midflowlm/.venv/bin/python3"
cd "$WORKTREE"

export PYTHONPATH="$WORKTREE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/smoke_test_${TIMESTAMP}.log"
OUTPUT_DIR="outputs/issue-9/sft_flow_midblock_3060_smoke"
DATA_DIR="data/reasoning_sft_smoke"

mkdir -p logs "$DATA_DIR" "$OUTPUT_DIR"

echo "========== SMOKE TEST STARTED at $(date) ==========" | tee "$LOG_FILE"
echo "Worktree: $WORKTREE" | tee -a "$LOG_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 2: Prepare smoke test data (streaming for faster load)
echo "--- Step 1: Preparing smoke test data ---" | tee -a "$LOG_FILE"
$VENV_PYTHON -c "
import sys, os, time
from pathlib import Path

ds_name = 'Jackrong/GLM-5.1-Reasoning-1M-Cleaned'
max_length = 1024
num_proc = 4
output_dir = Path('data/reasoning_sft_smoke')
output_dir.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer
from src.data.reasoning_sft import create_reasoning_sft_datasets

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use streaming to grab 1000 samples quickly, then convert to regular dataset
print('Loading 1000 samples via streaming...')
ds_stream = load_dataset(ds_name, split='train', streaming=True)
samples = []
for i, ex in enumerate(ds_stream):
    if i >= 1000:
        break
    samples.append(ex)
print(f'Collected {len(samples)} samples')

from datasets import Dataset
ds = Dataset.from_list(samples)

print(f'Processing {len(ds)} samples...')
train_ds, eval_ds = create_reasoning_sft_datasets(
    ds, tokenizer,
    max_length=max_length,
    num_proc=num_proc,
    val_split=0.02,
    max_train_samples=500,
    max_eval_samples=50,
    seed=1337,
)

train_path = output_dir / 'train'
eval_path = output_dir / 'eval'
train_ds.save_to_disk(str(train_path))
eval_ds.save_to_disk(str(eval_path))
print(f'Saved: train={len(train_ds)} sequences, eval={len(eval_ds)} sequences')
" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "FATAL: Data preparation failed" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"

# Phase 3: Train
echo "--- Step 2: Starting training ---" | tee -a "$LOG_FILE"
$VENV_PYTHON scripts/train_sft.py \
    --config configs/issue-9/sft_flow_midblock_3060.yaml 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "========== SMOKE TEST FINISHED at $(date) (exit=$EXIT_CODE) ==========" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Check outputs: ls -la $OUTPUT_DIR/" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE

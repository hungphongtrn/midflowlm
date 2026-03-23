---
# midflowlm-bjm0
title: Create MMLU-Pro evaluation script
status: completed
type: task
created_at: 2026-03-19T08:43:51Z
updated_at: 2026-03-19T08:43:51Z
---

Create a script to evaluate models on downstream task using chat template instead of just perplexity and loss using a small subset of MMLU-Pro (around 100 samples are enough). Remember to use chat template attached with Qwen/Qwen3.5-0.8B.

## Tasks
- [x] Create MMLU-Pro evaluation script with chat template support
- [x] Load TIGER-Lab/MMLU-Pro validation set (70 samples)
- [x] Integrate Qwen chat template for prompting
- [x] Implement accuracy-based evaluation metrics
- [x] Add CLI argument parsing and logging
- [x] Test the evaluation script

## Summary of Changes

Created `scripts/eval_mmlu_pro.py` - A comprehensive downstream task evaluation script for MMLU-Pro:

### Features:
- **Dataset**: Loads TIGER-Lab/MMLU-Pro validation split with configurable sample count (default 70)
- **Chat Template**: Uses Qwen's chat template via `tokenizer.apply_chat_template()` for proper prompting
- **Evaluation Metrics**: Computes accuracy (correct predictions / total) instead of perplexity/loss
- **Model Support**: 
  - Student models with checkpoints
  - Baseline models (identity, t1_shared, simple_recurrent)
  - Teacher model (original Qwen)
- **CLI Arguments**:
  - `--config`: Path to config YAML
  - `--checkpoint`: Path to student checkpoint
  - `--baseline`: Evaluate specific baseline
  - `--num-samples`: Number of MMLU-Pro samples (default 70)
  - `--num-steps`: List of T values to test
  - `--output`: JSON output path
- **Output**: Saves results as JSON with accuracy, latency, and comparison table

### Usage Examples:
```bash
# Evaluate teacher model
python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml

# Evaluate student with checkpoint
python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --checkpoint ./checkpoints/best.ckpt

# Evaluate baselines
python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --baseline all

# Custom sample count and steps
python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --num-samples 100 --num-steps 4 8
```

### Key Implementation Details:
- Uses `datasets` library to load MMLU-Pro
- Applies proper chat formatting with system message for multiple-choice QA
- Extracts single-letter answers (A, B, C, etc.) from model output
- Measures per-question latency for performance comparison
- Generates side-by-side comparison table in logs

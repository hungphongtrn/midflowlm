# Training Monitoring Guide for MidFlowLM

## Quick Start: Training Command

To run training with your cache at `./cache/tinystories_qwen_boundary_states`:

```bash
# Basic training (cache location is already configured in v0_onemotif.yaml)
python scripts/train_v0.py --config configs/v0_onemotif.yaml

# Resume from checkpoint
python scripts/train_v0.py --config configs/v0_onemotif.yaml --resume-from-checkpoint ./outputs/v0_qwen_iterative_midblock/checkpoints/best.ckpt

# Fast dev run (1 train step + 1 val step for testing)
python scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run

# Limited batch run (for quick validation)
python scripts/train_v0.py --config configs/v0_onemotif.yaml --limit-train-batches 10 --limit-val-batches 5
```

**Note:** The cache directory is already configured in `configs/v0_onemotif.yaml` at line 45:
```yaml
teacher_cache:
  cache_dir: "./cache/tinystories_qwen_boundary_states"
```

---

## Perplexity Evaluation

### Current Status

**The training script currently does NOT compute perplexity during training.** The cache contains:
- `input_ids` and `attention_mask` (for computing perplexity)
- `h_start`, `trajectory_targets`, `h_target` (for distillation)
- `teacher_logits` (optional, currently disabled in config)

### Perplexity Before Training (Baseline)

To evaluate perplexity **before** layer replacement (teacher model):

```bash
# The teacher model is the original Qwen model without modifications
# You can compute baseline perplexity using the eval script on a baseline
python scripts/eval_v0.py --config configs/v0_onemotif.yaml --baseline identity --output results_baseline.json
```

**Note:** The baseline models (identity, t1_shared, simple_recurrent) do not produce logits - they only produce hidden states. For full perplexity evaluation, you need the student model with LM head.

### Perplexity During Training

To compute perplexity during training, you need to:

1. **Enable logits storage in cache** (rebuild cache if needed):
   ```yaml
   teacher_cache:
     store_logits: true  # Change from false to true
   ```

2. **Run evaluation during training** (in a separate terminal):
   ```bash
   # Evaluate current checkpoint
   python scripts/eval_v0.py \
       --config configs/v0_onemotif.yaml \
       --checkpoint ./outputs/v0_qwen_iterative_midblock/checkpoints/best.ckpt \
       --output eval_results.json
   ```

### Perplexity After Training

To evaluate the final trained model:

```bash
# Evaluate trained model with multiple T values
python scripts/eval_v0.py \
    --config configs/v0_onemotif.yaml \
    --checkpoint ./outputs/v0_qwen_iterative_midblock/checkpoints/final.ckpt \
    --num-steps 1 4 8 \
    --output final_eval.json
```

---

## Monitoring Training Progress

### 1. Console Logs

The trainer logs to console every `log_every_n_steps` (default: 10 steps):

```
Step 100: loss=0.8234, T=4, lr=0.000095
```

**Key metrics to watch:**
- `loss`: Total training loss (should decrease)
- `T`: Current step count being trained
- `lr`: Learning rate (should decrease with cosine schedule)

### 2. Validation Metrics

Validation runs every `val_check_interval` (default: 250 steps) and at epoch end:

```
Validation at step 250: val/loss=0.6543 val/endpoint_loss=0.4321 val/trajectory_loss=0.5432 val/kl_loss=0.1234 val/ce_loss=0.0000 val/T=4.5
```

**Key validation metrics:**
- `val/loss`: Total validation loss (monitor for overfitting)
- `val/endpoint_loss`: Hidden state endpoint matching error
- `val/trajectory_loss`: Trajectory matching error
- `val/kl_loss`: KL divergence between student and teacher

### 3. Checkpoint Saving

Best checkpoints are saved based on `val/total_loss`:

```
Saved best checkpoint to ./outputs/v0_qwen_iterative_midblock/checkpoints/best.ckpt
```

Checkpoints include:
- Model weights
- Optimizer state
- Scheduler state
- Training step/epoch

### 4. Log Directory Structure

```
./outputs/v0_qwen_iterative_midblock/
├── checkpoints/
│   ├── best.ckpt              # Best model based on val/loss
│   ├── epoch_1.ckpt           # End of epoch 1
│   ├── epoch_2.ckpt           # End of epoch 2
│   └── final.ckpt             # Final model after training
└── logs/                      # TensorBoard/JSON logs (if configured)
```

---

## What to Monitor

### Primary Metrics

1. **Training Loss** (`loss`)
   - Should steadily decrease
   - If plateaued: try increasing learning rate or adjusting loss weights
   - If NaN/Inf: reduce learning rate or check data

2. **Validation Loss** (`val/loss`)
   - Best indicator of model quality
   - Save checkpoints based on this
   - If training loss ↓ but val loss ↑ = overfitting

3. **Endpoint Loss** (`endpoint_loss`, `val/endpoint_loss`)
   - Measures how well h_end matches teacher
   - Weighted by `endpoint_weight` in config (default: 1.0)

4. **Trajectory Loss** (`trajectory_loss`, `val/trajectory_loss`)
   - Measures intermediate step matching
   - Weighted by `trajectory_weight` in config (default: 1.0)

### Secondary Metrics

5. **KL Divergence** (`kl_loss`, `val/kl_loss`)
   - Logits-level distillation loss
   - Weighted by `kl_weight` in config (default: 0.25)

6. **Learning Rate** (`lr`)
   - Should follow warmup + cosine decay schedule
   - Check that warmup (default: 100 steps) completes

7. **T Distribution** (`T`)
   - Shows which step counts are being trained
   - Should match train_T_weights distribution

---

## Expected Training Behavior

### Normal Training

- **Epoch 1**: Loss drops rapidly (from ~2.0 to ~0.5)
- **Epoch 2-3**: Loss plateaus gradually (0.5 to 0.3)
- **Validation**: Should track training loss closely
- **Best checkpoint**: Usually from epoch 2 or 3

### Warning Signs

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Loss NaN/Inf | Learning rate too high | Reduce LR to 5e-5 |
| Loss not decreasing | Cache issue or bad init | Check cache integrity |
| Val loss >> Train loss | Overfitting | Reduce epochs or add regularization |
| High endpoint loss | Low endpoint_weight | Increase to 2.0 |
| High trajectory loss | T mismatch | Check trajectory alignment config |

---

## Advanced Monitoring

### Custom Perplexity Evaluation Script

To add perplexity monitoring during training, create a custom evaluation callback:

```python
# scripts/eval_ppl_during_training.py
#!/usr/bin/env python3
"""Evaluate perplexity on validation set during training."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.tinystories import TinyStoriesDataset
from torch.utils.data import DataLoader

def compute_ppl(model, dataloader, device):
    """Compute perplexity on dataloader."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

# Usage during training:
# 1. Load teacher model as baseline
# 2. Load student model from checkpoint
# 3. Compare perplexities
```

### TensorBoard Logging (if implemented)

```bash
# If TensorBoard logging is enabled:
tensorboard --logdir ./outputs/v0_qwen_iterative_midblock/logs
```

---

## Troubleshooting

### Check Cache Integrity

```python
# Quick Python check
from src.training.data import get_cache_info
cache_info = get_cache_info("./cache/tinystories_qwen_boundary_states")
print(cache_info)
```

### Verify Model is Training

```bash
# Check that trainable parameters are being updated
python scripts/print_model_stats.py --config configs/v0_onemotif.yaml
```

Expected output should show:
- Trainable params: ~50M-100M (midblock only)
- Frozen params: ~800M (Qwen base model)

### Debug One Training Step

```bash
# Run single step and print detailed metrics
python -c "
import sys
sys.path.insert(0, '.')
from scripts.train_v0 import *

config = load_config('configs/v0_onemotif.yaml')
model = create_student_model(config, 'cuda')
loss_fn = create_loss_function(config)
train_dl, val_dl = create_dataloaders(config, './cache/tinystories_qwen_boundary_states')

trainer = Trainer(model, loss_fn, config, 'cuda', train_dl, val_dl)
batch = next(iter(train_dl))
metrics = trainer.train_step(batch)
print('Step metrics:', metrics)
"
```

---

## Summary

| Task | Command |
|------|---------|
| Start training | `python scripts/train_v0.py --config configs/v0_onemotif.yaml` |
| Resume training | Add `--resume-from-checkpoint <path>` |
| Quick test | Add `--fast-dev-run` |
| Evaluate | `python scripts/eval_v0.py --config configs/v0_onemotif.yaml --checkpoint <path>` |
| Check model stats | `python scripts/print_model_stats.py --config configs/v0_onemotif.yaml` |

**Perplexity monitoring** requires running evaluation separately - it's not computed during the training loop. Run evaluation before training (baseline), during (periodic checkpoints), and after training to track perplexity changes.

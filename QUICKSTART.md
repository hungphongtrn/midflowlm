# Quick Start Guide - MidflowLM v0.1

Run the full experiment matrix on a Vast 3×3090 machine with minimal setup.

## 1. Clone and Setup (One-time)

```bash
# Clone repo
git clone <your-repo-url>
cd midflowlm

# Run setup (interactive - will prompt for wandb and huggingface tokens)
bash scripts/setup_env.sh
```

This sets up:
- Python dependencies
- Weights & Biases login
- Hugging Face token
- Verifies GPU access

## 2. Run Experiments

### Option A: Parallel on All GPUs (Recommended for 3×3090)

```bash
bash scripts/run_matrix.sh --parallel
```

Runs 15 experiments across 3 GPUs automatically. Monitors with wandb.

### Option B: Sequential (One GPU)

```bash
bash scripts/run_matrix.sh
```

### Option C: Single GPU Only

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_matrix.sh
```

## 3. Monitor Progress

**Primary:** https://wandb.ai - All metrics, logs, and checkpoints logged automatically

**Local:**
```bash
# Check status
cat .experiment_status

# View logs
ls logs/
tail -f logs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.log

# See what's running
ps aux | grep train.py
```

## 4. Resume After Interruption

The runner tracks completed experiments. Just re-run:

```bash
bash scripts/run_matrix.sh --parallel
```

It will skip completed experiments and continue with pending ones.

## 5. Experiment Matrix

15 configs in `configs/v0_1_matrix/`:

| Phase | Configs | Purpose |
|-------|---------|---------|
| P1 (3) | `p1_a1_*`, `p1_a2_*`, `p1_a3_*` | Architecture comparison |
| P2 (4) | `p2_l1_*` to `p2_l4_*` | Loss ablation |
| P3 (3) | `p3_d1_*` to `p3_d3_*` | Data mix ablation |
| P4 (5) | `p4_*_evalT{1,2,4,8,12}` | T-step sweep |

All configs use:
- seq_len=1024
- batch_size=2, grad_accum=8
- 3 epochs each
- bf16-mixed precision

## Troubleshooting

**"No GPUs detected"**
```bash
# Check GPU availability
nvidia-smi
python3 -c "import torch; print(torch.cuda.device_count())"
```

**"wandb login failed"**
```bash
wandb login  # Re-run manually
```

**"Out of memory"**
Edit `profiles/v0_1_3090_profile.json` to reduce `batch_size` to 1.

**Resume specific experiment**
```bash
# Edit config to add checkpoint path, then run:
python3 scripts/train.py --config configs/v0_1_matrix/YOUR_CONFIG.yaml
```

## Expected Runtime

With 3×3090 and parallel mode:
- ~15-20 hours for full matrix
- Each experiment: ~3-4 hours

Monitor ETA in wandb dashboard.

## Files Created

```
logs/                    # Training logs (one per experiment)
outputs/                 # Checkpoints (per experiment)
.experiment_status       # Run status tracking
.runner_lock            # Prevents duplicate runs
```

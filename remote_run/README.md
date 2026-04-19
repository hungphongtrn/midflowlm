# Remote Experiment Execution on RTX 3090

## Quick Start

1. Rent a 3090 instance (RunPod, Vast.ai, etc.)
2. Upload this repository to the instance
3. Run: `bash remote_run/setup_and_run.sh`

## What This Does

The script will:
1. Install uv (Python package manager)
2. Create a Python 3.10 virtual environment
3. Install PyTorch with CUDA 12.1 support
4. Install all required dependencies
5. Run all 15 experiments **sequentially** on the single 3090

## Experiment Matrix (Sequential Order)

### Phase 1: Architecture Sanity (3 experiments)
- P1-A1: One-shot projector, Mix B, End + KL
- P1-A2: Shared recurrent residual, Mix B, End + KL  
- P1-A3: Flow midblock, Mix B, End + KL

### Phase 2: Loss Ablation (4 experiments)
- P2-L1: Flow, Mix B, End only
- P2-L2: Flow, Mix B, End + KL
- P2-L3: Flow, Mix B, End + Traj + KL
- P2-L4: Flow, Mix B, End + Traj + KL + CE

### Phase 3: Data Mix Ablation (3 experiments)
- P3-D1: Flow, Mix A, End + Traj + KL
- P3-D2: Flow, Mix B, End + Traj + KL
- P3-D3: Flow, Mix C, End + Traj + KL

### Phase 4: T Sweep Evaluation (5 configs)
- P4-E1: Eval at T=1
- P4-E2: Eval at T=2
- P4-E3: Eval at T=4
- P4-E4: Eval at T=8
- P4-E5: Eval at T=12

## Monitoring

- Real-time logs: `logs/matrix_*/`
- Experiment tracking: wandb.ai (if configured)
- Status tracking: `.experiment_status/status_*.log`

## Expected Duration

Each experiment runs ~30-60 minutes on 3090:
- Total estimated time: ~8-15 hours for all 15 experiments
- The script runs them sequentially as requested

## Resume Capability

If interrupted, resume with:
```bash
bash scripts/run_matrix.sh --sequential --resume
```

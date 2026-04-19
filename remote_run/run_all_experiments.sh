#!/bin/bash
# Individual experiment commands for RTX 3090
# Run these one at a time (sequentially)
#
# Setup first:
#   uv venv --python 3.10
#   source .venv/bin/activate
#   uv pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
#   uv pip install transformers>=4.57.0 datasets>=2.18.0 torchmetrics>=1.4.0
#   uv pip install torchdiffeq>=0.2.5 einops>=0.8.0 pyyaml>=6.0
#   uv pip install safetensors>=0.4.0 numpy>=1.26.0 tqdm accelerate

set -e  # Stop on error

# ============================================================
# PHASE 1: Architecture Sanity (3 experiments)
# ============================================================

echo "========================================"
echo "PHASE 1: Architecture Sanity"
echo "========================================"

# P1-A1: One-shot projector, Mix B, End + KL
echo "Running P1-A1: One-shot projector..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml

# P1-A2: Shared recurrent residual, Mix B, End + KL
echo "Running P1-A2: Shared recurrent residual..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml

# P1-A3: Flow midblock, Mix B, End + KL
echo "Running P1-A3: Flow midblock..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml

# ============================================================
# PHASE 2: Loss Ablation (4 experiments)
# ============================================================

echo ""
echo "========================================"
echo "PHASE 2: Loss Ablation"
echo "========================================"

# P2-L1: Flow, Mix B, End only
echo "Running P2-L1: End only..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p2_l1_flow_mixb_end_trainT_r2468.yaml

# P2-L2: Flow, Mix B, End + KL
echo "Running P2-L2: End + KL..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p2_l2_flow_mixb_endkl_trainT_r2468.yaml

# P2-L3: Flow, Mix B, End + Traj + KL
echo "Running P2-L3: End + Traj + KL..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p2_l3_flow_mixb_endtrajkl_trainT_r2468.yaml

# P2-L4: Flow, Mix B, End + Traj + KL + CE
echo "Running P2-L4: End + Traj + KL + CE..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p2_l4_flow_mixb_endtrajklce_trainT_r2468.yaml

# ============================================================
# PHASE 3: Data Mix Ablation (3 experiments)
# ============================================================

echo ""
echo "========================================"
echo "PHASE 3: Data Mix Ablation"
echo "========================================"

# P3-D1: Flow, Mix A, End + Traj + KL
echo "Running P3-D1: Mix A (FineWeb only)..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d1_flow_mixa_endtrajkl_trainT_r2468.yaml

# P3-D2: Flow, Mix B, End + Traj + KL
echo "Running P3-D2: Mix B (FineWeb + UltraChat)..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d2_flow_mixb_endtrajkl_trainT_r2468.yaml

# P3-D3: Flow, Mix C, End + Traj + KL
echo "Running P3-D3: Mix C (Full mix)..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p3_d3_flow_mixc_endtrajkl_trainT_r2468.yaml

# ============================================================
# PHASE 4: T Sweep Evaluation (5 configs)
# ============================================================

echo ""
echo "========================================"
echo "PHASE 4: T Sweep Evaluation"
echo "========================================"

# P4-E1: Eval at T=1
echo "Running P4-E1: Eval T=1..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT1.yaml

# P4-E2: Eval at T=2
echo "Running P4-E2: Eval T=2..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT2.yaml

# P4-E3: Eval at T=4
echo "Running P4-E3: Eval T=4..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT4.yaml

# P4-E4: Eval at T=8
echo "Running P4-E4: Eval T=8..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT8.yaml

# P4-E5: Eval at T=12
echo "Running P4-E5: Eval T=12..."
uv run python scripts/train.py --config configs/v0_1_matrix/midflow_qwen_8to11_p4_flow_mixc_endtrajkl_evalT12.yaml

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo "Check logs/matrix_*/ for detailed logs"
echo "Check wandb.ai for experiment tracking"

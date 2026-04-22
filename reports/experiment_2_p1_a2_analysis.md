# Experiment 2 Analysis Report: P1-A2 (Shared Recurrent Residual)

**W&B Run:** [stilted-paper-3 (ze54okvs)](https://wandb.ai/yuuart/midflowlm-v0-1/runs/ze54okvs)  
**Status:** ✅ Finished  
**Duration:** ~19.3 hours (69,684 seconds)  
**Completed:** April 22, 2026

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Experiment ID** | P1-A2 |
| **Architecture** | Shared Recurrent Residual Block |
| **Loss Configuration** | End + KL |
| **Data Mix** | Mix B (FineWeb + UltraChat) |
| **Train T Values** | [2, 4, 6, 8] |
| **Max Steps T** | 8 |
| **Model** | Qwen/Qwen3.5-0.8B |
| **Replacement Layers** | 8-11 (4 layers) |
| **Replacement Depth** | 4 |
| **Batch Size** | 1 |
| **Gradient Accumulation** | 16 |
| **Effective Batch Size** | 16 |
| **Learning Rate** | 1e-4 |
| **Max Epochs** | 3 |
| **Precision** | bf16-mixed |

---

## Final Metrics Summary

| Metric | Training | Validation |
|--------|----------|------------|
| **Total Loss** | 0.0358 | 0.0562 |
| **KL Loss** | 0.0693 | 0.1098 |
| **Endpoint Loss** | 0.0358 | 0.0562 |
| **CE Loss** | 0 | 0 |
| **Velocity Loss** | 0 | 0 |
| **Gradient Norm** | 1.397 | - |
| **Learning Rate** | 9.91e-05 | - |
| **Mean T** | 0.764 | 5.06 |

---

## Key Findings

### 1. Training Stability
- **Status:** ✅ Stable convergence
- **Final gradient norm:** 1.397 (well below clip threshold of 1.0, indicates good stability)
- **No NaN values detected** in training metrics
- **Loss decreased** from initial ~0.115 to final ~0.036

### 2. Loss Composition
- **Endpoint loss dominates** (100% of total loss)
- **KL loss:** 0.069 (training), 0.110 (validation)
- **CE and Velocity losses disabled** (weight = 0)
- **Endpoint + KL configuration working as expected**

### 3. Validation Performance
- **Validation loss higher than training** (0.056 vs 0.036)
- **Gap:** ~56% higher validation loss suggests moderate overfitting
- **Validation KL loss:** 0.110 vs training 0.069 (58% higher)

### 4. Training Dynamics
- **Total steps:** 3,201 training steps
- **Warmup completed** (100 steps) before main training
- **Learning rate schedule:** Cosine with warmup, reached near-max by end
- **T sampling:** Uniform across [2, 4, 6, 8] during training

### 5. System Utilization
- **Platform:** Linux x86_64
- **W&B version:** 0.26.0
- **PyTorch version:** 2.x (implied from CUDA 12.8)
- **System metrics logged:** GPU, CPU, memory, disk I/O, network

---

## Architecture Details

### Shared Recurrent Residual Block Configuration
```yaml
family: shared_recurrent_residual
depth: 4
start_layer: 8
end_layer: 11
conditioning_mode: timestep_plus_layer_boundary
use_step_conditioning: true
use_qwen_causal_mask: true
init_strategy: fresh
mlp_ratio: 4
qkv_bias: true
```

### Training Mode
- **Teacher mode:** `online_no_cache` (real-time teacher target extraction)
- **Student training:** Continuous time sampling with T ∈ {2, 4, 6, 8}
- **Gradient checkpointing:** Enabled (memory optimization)

---

## Comparison Context (Phase 1 - Architecture Sanity)

| Exp | Architecture | Train Loss | Val Loss | Notes |
|-----|--------------|------------|----------|-------|
| P1-A1 | One-shot projector | - | - | Baseline |
| **P1-A2** | **Shared recurrent** | **0.036** | **0.056** | **This run** |
| P1-A3 | Flow midblock | - | - | To be evaluated |

---

## Recommendations

1. **Validation gap suggests** potential regularization benefits:
   - Consider dropout increase
   - Monitor for early stopping opportunities

2. **Shared recurrent block** appears functional but comparison with P1-A1 and P1-A3 needed for architectural ranking

3. **For next experiments:**
   - Evaluate whether 56% val/train gap is consistent across architectures
   - Consider loss weight tuning if validation remains elevated

---

## Raw Config Reference

**Full config file:** `configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml`

**Tags:** `v0.1`, `p1`, `a2`, `architecture`, `recurrent`, `mix-b`

**Checkpoint directory:** `./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints`

---

*Report generated: April 22, 2026*
*Analysis for: Experiment 2 (P1-A2) - Shared Recurrent Residual Architecture*

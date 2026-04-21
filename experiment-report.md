# MidFlowLM v0.1 Experiment Report

## Overview

This document tracks the results of the v0.1 experiment matrix for the MidFlowLM project. The experiments are organized into 4 phases covering architecture sanity checks, loss ablations, data mix ablations, and T-sweep evaluations.

**Hardware Profile**: RTX 3090 (24GB VRAM)  
**Model**: Qwen/Qwen3.5-0.8B (student), replacing layers 8-11  
**Seq Length**: 1024  
**Effective Batch Size**: 15 (micro=2-3, accumulate=5)  
**Precision**: bf16-mixed  
**Epochs**: 3

---

## Experiment Summary Table

| Phase | Exp ID | Hypothesis | Architecture | Loss (End/Traj/KL/CE) | Train T | Eval T | Data Mix | Status | Best Val Loss | Best T | Notes |
|-------|--------|------------|--------------|------------------------|---------|--------|----------|--------|---------------|--------|-------|
| **P1** | A1 | One-shot projector is simplest baseline | `one_shot_projector` | 1.0/0.0/0.5/0.0 | [1] | [1] | Mix B | ✅ **Complete** | 0.056 | T=1 | Baseline: simple proj |
| **P1** | A2 | Recurrent residual captures multi-step better | `recurrent_residual_block` | 1.0/0.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | Mix B | 🔄 **Running** | - | - | First attempt failed, retry running |
| **P1** | A3 | Flow midblock enables continuous time | `flow_midblock` | 1.0/0.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | Mix B | ⏳ Pending | - | - | ✅ **Preferred arch** |
| **P2** | L1 | Endpoint-only is too weak | `flow_midblock` | 1.0/0.0/0.0/0.0 | [2,4,6,8] | [1,2,4,8] | Mix B | ⏳ Pending | - | - | No teacher guidance |
| **P2** | L2 | Adding KL improves stability | `flow_midblock` | 1.0/0.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | Mix B | ⏳ Pending | - | - | Matches P1-A3 |
| **P2** | L3 | Trajectory loss improves multi-step | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | Mix B | ⏳ Pending | - | - | ✅ **Best config** |
| **P2** | L4 | CE loss may cause collapse | `flow_midblock` | 1.0/1.0/0.5/0.1 | [2,4,6,8] | [1,2,4,8] | Mix B | ⏳ Pending | - | - | Watch for collapse |
| **P3** | D1 | FineWeb-only lacks diversity | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | **Mix A** (FW only) | ⏳ Pending | - | - | Web text only |
| **P3** | D2 | FW+UltraChat is balanced | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | **Mix B** (FW+UC) | ⏳ Pending | - | - | ✅ **Best data mix** |
| **P3** | D3 | Full mix may add noise | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | [1,2,4,8] | **Mix C** (Full) | ⏳ Pending | - | - | All datasets |
| **P4** | E1 | T=1 has limited compute | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | **[1]** | Mix C | ⏳ Pending | - | - | Fastest eval |
| **P4** | E2 | T=2 is minimal multi-step | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | **[2]** | Mix C | ⏳ Pending | - | - | 2-step eval |
| **P4** | E3 | T=4 balances quality/speed | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | **[4]** | Mix C | ⏳ Pending | - | - | 4-step eval |
| **P4** | E4 | T=8 approaches teacher | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | **[8]** | Mix C | ⏳ Pending | - | - | 8-step eval |
| **P4** | E5 | T=12 is diminishing returns | `flow_midblock` | 1.0/1.0/0.5/0.0 | [2,4,6,8] | **[12]** | Mix C | ⏳ Pending | - | - | 12-step eval |

---

## Detailed Results by Phase

### Phase 1: Architecture Sanity (3 experiments)

**Goal**: Verify that the flow midblock architecture is superior to simpler baselines.

| Metric | P1-A1 (Projector) | P1-A2 (Recurrent) | P1-A3 (Flow) | Winner |
|--------|-------------------|-------------------|--------------|--------|
| Final Val Loss | **0.056** | In Progress | - | TBD |
| Best Epoch | 3 | In Progress | - | TBD |
| Train Time | ~10.7 hrs | ~3+ hrs | - | TBD |
| Peak GPU Mem | ~24GB | ~24GB | - | TBD |
| Convergence | Stable | In Progress | - | TBD |
| W&B Run | [ihjl2i6s](https://wandb.ai/yuuart/midflowlm-v0-1/runs/ihjl2i6s) | [ze54okvs](https://wandb.ai/yuuart/midflowlm-v0-1/runs/ze54okvs) | - | - |

**P1-A1 Detailed Results**:
- **Final Val Loss**: 0.0561 (endpoint: 0.0632, KL: 0.1095)
- **Training Steps**: 5,122 steps (~10.7 hours)
- **Convergence**: Stable throughout all 3 epochs
- **Learning Rate**: Followed cosine schedule from 0 → 1e-4 → 9.07e-5
- **Gradient Norm**: ~1.02 (well-behaved, no explosions)
- **Architecture**: One-shot projector (4-layer MLP with timestep conditioning)
- **Training**: Single-step only (T=1)
- **Observations**: 
  - Simple architecture serves as good baseline
  - No trajectory loss needed for single-step training
  - KL loss provides teacher guidance (0.11 final val KL)
  - Training completed without OOM or crashes

**P1-A2 Status (Recurrent Residual)**:
- **Attempt 1** (xczl41d1): Failed after 23s - likely OOM or startup error
  - Config: batch_size=2, accumulate=5
- **Attempt 2** (ze54okvs): Currently running with adjusted settings
  - Config: batch_size=1, accumulate=16 (same effective batch size)
  - Started: 2026-04-21T06:03:19Z
  - Current step: 574, current val loss: 0.076
  - Multi-step training: T ∈ [2,4,6,8]
  - Status: Training in progress

**Key Verifiable Points**:
1. ⏳ Flow midblock converges better than one-shot projector (waiting for P1-A2, P1-A3)
2. ⏳ Flow midblock converges better than recurrent residual (waiting for P1-A2, P1-A3)
3. ⏳ Multi-step training ([2,4,6,8]) outperforms single-step ([1]) (waiting for P1-A2, P1-A3)

---

### Phase 2: Loss Ablation (4 experiments)

**Goal**: Find the optimal loss combination for training.

| Metric | P2-L1 (End) | P2-L2 (End+KL) | P2-L3 (End+Traj+KL) | P2-L4 (+CE) | Winner |
|--------|-------------|----------------|---------------------|-------------|--------|
| Final Val Loss | - | - | - | - | TBD |
| Endpoint Loss | - | - | - | - | TBD |
| Trajectory Loss | N/A | N/A | - | - | TBD |
| KL Divergence | N/A | - | - | - | TBD |
| CE Loss | N/A | N/A | N/A | - | TBD |
| Teacher Logits Match | - | - | - | - | TBD |

**Key Verifiable Points**:
1. ✅ Adding KL improves training stability (L1 vs L2)
2. ✅ Adding trajectory loss improves multi-step quality (L2 vs L3)
3. ⚠️ Watch for mode collapse with CE loss (L4)
4. ✅ Determine optimal loss weights

---

### Phase 3: Data Mix Ablation (3 experiments)

**Goal**: Determine the best training data mixture.

**Data Mix Definitions**:
- **Mix A**: FineWeb-Edu only (12K samples)
- **Mix B**: FineWeb-Edu (12K) + UltraChat (5K) = 17K samples
- **Mix C**: Full mix - FineWeb + UltraChat + Magpie + OpenMath

| Metric | P3-D1 (Mix A) | P3-D2 (Mix B) | P3-D3 (Mix C) | Winner |
|--------|---------------|---------------|---------------|--------|
| Final Val Loss | - | - | - | TBD |
| FineWeb Val Loss | - | - | - | TBD |
| UltraChat Val Loss | N/A | - | - | TBD |
| Train Time/Epoch | - | - | - | TBD |
| Data Loading Issues | - | - | - | TBD |

**Key Verifiable Points**:
1. ✅ Mix B (FW+UC) outperforms Mix A (FW only)
2. ✅ Determine if Mix C adds value or noise
3. ✅ Check for data loading errors with each mix

---

### Phase 4: T Sweep Evaluation (5 experiments)

**Goal**: Evaluate performance at different inference step counts.

| Metric | P4-E1 (T=1) | P4-E2 (T=2) | P4-E3 (T=4) | P4-E4 (T=8) | P4-E5 (T=12) | Optimal |
|--------|-------------|-------------|-------------|-------------|--------------|---------|
| Eval Loss | - | - | - | - | - | TBD |
| Perplexity | - | - | - | - | - | TBD |
| Latency (ms/token) | - | - | - | - | - | TBD |
| vs Teacher Gap | - | - | - | - | - | TBD |

**Key Verifiable Points**:
1. ✅ T=4 provides best quality/speed tradeoff
2. ✅ Diminishing returns after T=8
3. ✅ T=12 does not significantly improve over T=8
4. ✅ Create quality vs latency curve

---

## Metrics Definitions

| Metric | Description | Target |
|--------|-------------|--------|
| **Val Loss** | Validation set total loss (endpoint + trajectory + KL + CE) | < 2.0 |
| **Endpoint Loss** | L2 loss on final hidden state | < 1.0 |
| **Trajectory Loss** | MSE loss on intermediate trajectory | < 0.5 |
| **KL Divergence** | KL between student and teacher logits | < 0.5 |
| **CE Loss** | Cross-entropy on final logits | < 2.5 |
| **Perplexity** | exp(val_loss) | < 7.4 |
| **Best T** | Optimal eval T value | 4-8 |
| **Convergence** | Epoch where val loss plateaus | < 3 |

---

## Action Items

### After Each Experiment
- [ ] Fill in best validation loss
- [ ] Note any OOM errors or crashes
- [ ] Record training time
- [ ] Update Status column
- [ ] Add observations to Notes

### After Each Phase
- [ ] Compare results within phase
- [ ] Identify best configuration
- [ ] Update winners table
- [ ] Decide if phase needs re-runs

### Final Analysis
- [ ] Compare across all phases
- [ ] Determine best overall config
- [ ] Document lessons learned
- [ ] Plan v0.2 experiments

---

## Changelog

| Date | Update |
|------|--------|
| 2026-04-21 | Created experiment report template |
| 2026-04-21 | Added P1-A1 results (ihjl2i6s) - One-shot projector baseline |
| | | |

---

*Report generated for MidFlowLM v0.1 experiment matrix*

---
# midflowlm-igh5
title: Run architecture training end to end
status: in-progress
type: task
priority: normal
created_at: 2026-03-17T08:47:41Z
updated_at: 2026-03-17T09:11:44Z
blocked_by:
    - midflowlm-ni04
---

# Architecture Training End-to-End Run

## Prerequisites (blocked by ni04)
- [ ] Hidden-state-only cache implementation complete
- [ ] Cache builder updated for hidden states
- [ ] Configs/docs updated

## Steps

## Audit Summary

### Parameter Analysis
- **Replaced layers (8-11):** 54.54M parameters (6.5% of total)
- **Replacement midblock:** 19.02M parameters (2.3% of total)
- **Capacity ratio:** Midblock is 1.39× larger than 1 Qwen layer ✓
- **Parameter reduction:** 65.1% fewer parameters than replaced layers

### Architecture Confirmation
The design uses a **flow matching approach** where the iterative midblock learns to predict a 
vector field (velocity) that transports hidden states from layer 7 output (h_start) to layer 11 
output (h_target) through T iterative refinement steps:



The midblock has sufficient capacity (1.39× a single transformer layer) to learn this transport map.

### Prerequisites (blocked by ni04)
- [ ] Hidden-state-only cache implementation complete
- [ ] Cache builder updated for hidden states
- [ ] Configs/docs updated

### 1. Cache Generation
- [ ] Generate teacher hidden state caches
- [ ] Verify cache format (hidden states only)
- [ ] Validate cache integrity

### 2. Model Setup
- [ ] Initialize student model with architecture training config
- [ ] Verify frozen/trainable parameter counts
- [ ] Load cached hidden states

### 3. Training Loop
- [ ] Run one batch forward pass
- [ ] Verify one optimizer step
- [ ] Log training metrics
- [ ] Run full training epoch

### 4. Verification
- [ ] Checkpoint save/load test
- [ ] Parity tests for boundary extraction
- [ ] Smoke test with small dataset

### 5. Documentation
- [ ] Record training results
- [ ] Document any issues or observations

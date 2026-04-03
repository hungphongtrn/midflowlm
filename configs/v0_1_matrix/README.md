# v0.1 Experiment Matrix Configs

This directory contains all YAML configs for the v0.1 experiment matrix.

## Structure

- `p1_*`: Phase 1 - Architecture Sanity (3 configs)
  - A1: One-shot projector
  - A2: Shared recurrent residual
  - A3: Flow midblock
- `p2_*`: Phase 2 - Loss Ablations (4 configs)
  - L1: Endpoint only
  - L2: Endpoint + KL
  - L3: Endpoint + Trajectory + KL
  - L4: Endpoint + Trajectory + KL + CE
- `p3_*`: Phase 3 - Data Mix Ablations (3 configs)
  - D1: Mix A (FineWeb only)
  - D2: Mix B (FineWeb + UltraChat)
  - D3: Mix C (Full mix)
- `p4_*`: Phase 4 - T Sweep (5 configs)
  - Eval at T=1, 2, 4, 8, 12

## Regeneration

To regenerate all configs:

```bash
python scripts/generate_v0_1_configs.py
```

## Hardware Profile

All configs use the same hardware profile:
- seq_len: 1024
- batch_size: 2 (microbatch)
- accumulate_grad_batches: 8
- effective_batch_size: 16
- precision: bf16-mixed
- gradient_checkpointing: true

See `profiles/v0_1_3090_profile.json` for details.

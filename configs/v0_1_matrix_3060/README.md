# v0.1 Experiment Matrix - 3060 Adapted

These configs are adapted for RTX 3060 12GB.

Changes from base configs:
- batch_size: 2 → 1
- accumulate_grad_batches: 8 → 16
- num_workers: 2 → 0
- pin_memory: true → false

Effective batch size remains 16.

Usage:
```bash
# Single experiment
python3 scripts/train.py --config configs/v0_1_matrix_3060/CONFIG.yaml

# Full matrix (sequential on 3060 - ~40-60 hours)
for cfg in configs/v0_1_matrix_3060/*.yaml; do
  python3 scripts/train.py --config "$cfg"
done
```

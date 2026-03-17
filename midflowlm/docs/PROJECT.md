# midflowlm

A research project exploring iterative latent refinement for language model distillation. Replaces a middle span of Qwen3.5-0.8B with a trainable iterative block that refines hidden states over T steps, distilling both teacher endpoint states and intermediate trajectory.

## Goal

Replace layers 8-11 of Qwen3.5-0.8B (default span) with an iterative refinement block that:
- Starts from frozen layer 7 hidden state (`h_start`)
- Runs T residual refinement steps: `h_{k+1} = h_k + delta_k`
- Distills both teacher endpoint (`h_target`) and full trajectory
- Achieves competitive performance with reduced effective depth

## Architecture

```
Input tokens
    ↓
[Qwen Embeddings + Layers 0-7] (frozen)
    ↓
h_start ──→ [IterativeMidblock] (trainable, T steps)
    ↓
[Qwen Layers 12-23] (frozen)
    ↓
[Final Norm + LM Head] (frozen)
    ↓
Logits
```

- **Frozen**: All Qwen modules outside replacement span (~97% of parameters)
- **Trainable**: IterativeMidblock only (~3% of parameters, ~22M params)
- **Key feature**: Variable T without model rebuild

## Tech Stack

- **Base Model**: Qwen3.5-0.8B (Hugging Face Transformers)
- **Training Framework**: Raw PyTorch (no Lightning)
- **Data**: TinyStories dataset
- **Key Dependencies**: torch, transformers, datasets, safetensors

## Repository Structure

```
midflowlm/
├── configs/v0_onemotif.yaml     # Main experiment configuration
├── docs/
│   ├── PROJECT.md               # This file
│   ├── plans/                   # Implementation plans
│   └── flux-adaptation-notes.md # Research notes
├── external/minFM/              # Vendored training reference
├── scripts/
│   ├── build_teacher_cache.py   # Offline teacher inference
│   ├── train_v0.py              # Training script
│   ├── eval_v0.py               # Evaluation script
│   └── print_model_stats.py     # Model inspection
├── src/
│   ├── data/                    # Data loading & caching
│   ├── eval/                    # Evaluation baselines
│   ├── model/                   # Model components
│   └── training/                # Training loop & losses
└── tests/                       # Comprehensive test suite
```

## Quick Start

```bash
# View model statistics
python scripts/print_model_stats.py --config configs/v0_onemotif.yaml

# Build teacher cache
python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 100

# Train
python scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run

# Evaluate
python scripts/eval_v0.py --config configs/v0_onemotif.yaml --checkpoint outputs/checkpoint_best.pt
```

## Key Design Decisions

1. **Raw PyTorch over Lightning**: Faster iteration, explicit control
2. **Offline teacher caching**: Required before student training
3. **Mandatory trajectory supervision**: Not just endpoint distillation
4. **Variable T training**: Sample T values from distribution during training
5. **Parity-first**: Verify Qwen boundary extraction before building on it

## Trajectory Alignment

The training handles three T regimes:
- **T = depth(span)**: Exact layer-wise trajectory matching
- **T < depth(span)**: Compression via uniform/weighted sampling
- **T > depth(span)**: Expansion via linear interpolation

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_midblock.py -v

# Smoke tests only
pytest tests/test_train_smoke.py -v
```

## Go/No-Go Criteria (v0)

Do not scale unless:
1. Teacher-cache generation is deterministic
2. Parity tests pass
3. Raw PyTorch train/val smoke tests pass
4. Trajectory supervision is active
5. Iterative block beats identity baseline
6. Comparable to simple distilled baseline

## References

- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-0.8B
- minFM (training reference): external/minFM/
- Plan: docs/plans/2025-03-15-v0-flow-midblock.md

## License

MIT

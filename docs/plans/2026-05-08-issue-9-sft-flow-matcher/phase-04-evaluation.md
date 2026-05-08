# Phase 4: Evaluation — TBD

**Status:** Stub — will be detailed after Phase 3 completes.

## High-Level Scope
- Multi-T evaluation at T=1, 4, 8, 16, 32
- Metrics: fixed-T accuracy, oracle best-of-T accuracy, prediction-change rate, answer distribution
- Compare against original P3-D3 baseline and teacher/baseline where available
- Generate evaluation report

## Key Unknowns (to resolve)
- Which benchmarks to evaluate on (MMLU-Pro, ARC, etc.)
- How to compute oracle best-of-T (pass@k across T values)
- Prediction-change rate metric definition
- Answer distribution visualization format

## Rough File Plan
- `scripts/eval_sft_multi_t.py` — Multi-T evaluation script
- `src/eval/sft_metrics.py` — Metric computation utilities
- `reports/issue-9/eval_results.md` — Evaluation report template

**Depends on:** Phase 3

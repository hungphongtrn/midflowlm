# P3-D3 Diagnostic Report: T-Scaling Investigation

## 1. Executive Summary

**Q1: Does increasing T change hidden states before decoding?** — **PASS** (mean_delta=0.0124, p=0.0000, d=0.21 (small))

**Q2: Does increasing T change logits without changing parsed predictions?** — **WARN** (Logits do not change measurably between T values)

**Q3: Does increasing T fail to put probability mass on reachable answer labels?** — **FAIL** (Max mean prob on correct answer = 0.108 < 0.3 — answer-token supervision likely needed)

## 2. Flow Integration Analysis

### Endpoint Hidden Norm per T

| T | Probes | Mean Norm | Std Norm | Median Norm |
|---|--------|-----------|----------|-------------|
| 1 | 40 | 2.0797 | 0.0591 | 2.0767 |
| 2 | 40 | 2.0720 | 0.0584 | 2.0693 |
| 8 | 40 | 2.0682 | 0.0579 | 2.0657 |
| 64 | 40 | 2.0673 | 0.0578 | 2.0649 |

### Pairwise T-Test Results (Bonferroni-corrected)

| T_a | T_b | t-stat | p-value | Significant | Cohen's d | Interpretation | Mean Delta |
|-----|-----|--------|---------|-------------|-----------|---------------|------------|
| 1 | 2 | 41.265 | 0.0000 | YES | 0.132 | small | -0.0077 |
| 1 | 8 | 38.063 | 0.0000 | YES | 0.197 | small | -0.0115 |
| 1 | 64 | 37.127 | 0.0000 | YES | 0.212 | small | -0.0124 |
| 2 | 8 | 32.770 | 0.0000 | YES | 0.065 | small | -0.0038 |
| 2 | 64 | 31.714 | 0.0000 | YES | 0.080 | small | -0.0046 |
| 8 | 64 | 27.708 | 0.0000 | YES | 0.015 | small | -0.0009 |

### Velocity Norm per T

| T | Mean Velocity | Std Velocity | Nonzero Probes |
|---|---------------|--------------|----------------|
| 1 | 0.0000 | 0.0000 | 0 |
| 2 | 0.4338 | 0.0259 | 40 |
| 8 | 0.7617 | 0.0453 | 40 |
| 64 | 0.8577 | 0.0510 | 40 |

### Divergence from T1

| T | Mean Divergence | Proportion Diverged |
|---|-----------------|---------------------|
| 2 | 0.0671 | 100.00% |
| 8 | 0.1110 | 100.00% |
| 64 | 0.1231 | 100.00% |

## 3. Decoder/Readout Analysis

### Per-T Accuracy

| T | Probes | Correct | Accuracy |
|---|--------|---------|----------|
| 1 | 40 | 0 | 0.00% |
| 2 | 40 | 1 | 2.50% |
| 8 | 40 | 1 | 2.50% |
| 64 | 40 | 1 | 2.50% |

### KL/JS Divergence from Teacher per T

| T | Mean KL | Mean JS | Median KL | Median JS |
|---|---------|---------|-----------|-----------|
| 1 | 0.3000 | 0.0645 | 0.1988 | 0.0461 |
| 2 | 0.3017 | 0.0644 | 0.1931 | 0.0478 |
| 8 | 0.3041 | 0.0645 | 0.1925 | 0.0478 |
| 64 | 0.3050 | 0.0646 | 0.1925 | 0.0478 |

### Answer Coverage per T

| T | Mean Pr(correct) | Median Pr(correct) | Probes Pr > 50% |
|---|------------------|---------------------|-----------------|
| 1 | 0.1053 | 0.0142 | 0 |
| 2 | 0.1069 | 0.0166 | 0 |
| 8 | 0.1078 | 0.0182 | 0 |
| 64 | 0.1080 | 0.0186 | 0 |

### Prediction Stability Across T

- Always wrong (all T): 39
- Flipped prediction (changed between T values): 4
- Became correct at higher T: 1

## 4. Root Cause Decision Tree

```
Q1: Hidden states change with T?
└── YES → Q2: Logits change without prediction change?
    ├── NO → ROOT CAUSE: Flow integration too weak
    │        Logits stuck despite hidden-state changes — timestep conditioning or target scale too small
```

## 5. Recommendations

1. **Audit flow integration strength** → Check whether hidden-state deltas from T1→T64 are statistically significant; if borderline, increase flow block capacity → Expected: clearer separation between T values → Effort: Medium
2. **Review prediction parity** → Verify that logit shifts, when they occur, are in the direction of correct answer labels → Expected: flipped probes become correct more often → Effort: Low

---
*Report generated from traces at `results/diagnostic_p3d3/traces`. Probes loaded from `results/diagnostic_p3d3/probes.json`.*

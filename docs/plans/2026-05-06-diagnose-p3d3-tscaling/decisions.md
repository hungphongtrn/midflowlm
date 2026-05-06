# Decision Log

## 2026-05-06: Diagnostic Findings

### Root Cause: Flow Integration Too Weak

The P3-D3 flow matching pipeline produces statistically significant hidden-state changes
(Q1: PASS, p < 0.0001, Cohen's d = 0.21), but these changes don't reach the decoder
output (Q2: WARN — no measurable logit shift) and never put mass on correct answer labels
(Q3: FAIL — max Pr(correct) = 10.8%).

### Key Evidence

| Decision | Rationale |
|----------|-----------|
| Hidden states DO change with T | All pairwise t-tests significant after Bonferroni correction; 100% of probes diverge from T1 at higher T |
| Velocity increases with T | 0.43 → 0.86 mean velocity from T=2→T=64 — ODE is doing work, not dead |
| Hidden norm decreases slightly with T | 2.080 → 2.067 — small negative trend, opposite of expected growth |
| Logits are FROZEN across T | Mean pairwise KL delta ≈ 0; logit distributions don't shift despite hidden-state movement |
| Predictions are essentially random | 39/40 probes always wrong, only 1 became correct at any T |

### Recommendation Path

The next remediation issue (#7) should address the gap between observable
hidden-state changes and the frozen decoder output. Top candidates:

1. **Increase target scale** — Scale up h8–h11 teacher anchor targets to produce
   larger hidden-state deltas (aim for Cohen's d > 0.5)
2. **Add answer-token supervision** — Cross-entropy over A–J logits using teacher
   soft targets to directly train the readout pathway
3. **Investigate normalization barriers** — LayerNorm or RMSNorm may be absorbing
   the small midblock deltas before the lm_head projection

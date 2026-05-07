# Diagnose flat P3-D3 T-scaling and answer-space collapse - Strategy

## Goal

Build a deterministic diagnostic loop that isolates where T-scaling collapses in the P3-D3 flow matching pipeline, producing a reproducible probe, machine-readable artifacts, and a recommendation for the next remediation issue.

## Architecture

The diagnostic pipeline has three layers:

1. **Fixed Probe Set** — a curated subset of ARC-Easy (teacher-correct/student-wrong) and MMLU-Pro (correct labels E-J) examples from existing `results/stress_test/` outputs. These examples are loaded from JSON, not re-evaluated.

2. **Deterministic Trace Runner** — loads the P3-D3 checkpoint once, then runs forward passes at `T ∈ {1, 2, 8, 64}` with fixed seed and decoding config. Captures two trace families:
   - **Flow integration traces**: endpoint hidden-state deltas, per-step velocity norms, trajectory differences (when available).
   - **Decoder/readout traces**: full logits over A-J answer tokens, decoded answer text, parsed answer, and teacher answer distributions.

3. **Diagnostic Report** — programmatically generated report (markdown + JSON/CSV artifacts) answering three key questions:
   - Does increasing T change hidden states before decoding?
   - Does increasing T change logits without changing parsed predictions?
   - Does increasing T fail to put probability mass on reachable answer labels?

## Tech Stack

- Existing `FrozenQwenStudent` model and `FlowMidblock` (in `src/model/`)
- `torchdiffeq` for ODE trajectory capture
- Existing checkpoint loading (`torch.load` with `model_state_dict`)
- Existing tokenizer (`Qwen/Qwen3.5-0.8B`)
- No new dependencies

## Constraints & Assumptions

- **Deterministic**: fixed seed, no dropout, greedy decoding, pinned random state per probe
- **Checkpoint**: uses the P3-D3 checkpoint already producing the `results/stress_test/` data
- **No autoregressive loop**: to isolate flow integration from generation dynamics, the first pass uses a single forward call (next-token prediction) over the full prompt
- **Output artifacts**: JSON for traces, CSV for aggregate stats, markdown for the diagnostic report
- **Smoke command**: a single command that runs a minimal subset (2-4 examples, T=1 and T=8) and verifies T changes internals

## Phases (High-Level)

### Phase 1: Probe Selection & Trace Runner Skeleton
**Outcome:** Fixed probes are selected, checkpoint loads deterministically, and a single-example forward pass at T=1 produces trace output.
**Rough scope:** Probe curation from existing stress-test JSON; seed/determinism infrastructure; minimal trace runner that calls `model.forward()` with `return_dict=True` and captures `endpoint_hidden` and logits.

### Phase 2: Full Trace Capture Across T
**Outcome:** Runner captures all two trace families (flow integration + decoder/readout) across all T values for the full probe set. Artifacts saved as JSON.
**Rough scope:** ODE trajectory capture (`_forward_ode` with `return_trajectory=True`); per-step velocity norms; logit distribution extraction over answer tokens; teacher hidden-state comparison via `extract_teacher_targets()`.

### Phase 3: Diagnostic Report & Recommendation
**Outcome:** Report answers the three key questions; smoke command is documented; final recommendation identifies the next remediation path.
**Rough scope:** Programmatic report generation from traces; delta-visualization tables; smoke-test script; final recommendation distilled from evidence.
**Depends on:** Phase 2

## Open Questions

1. **Which specific P3-D3 checkpoint?** — Identify exact checkpoint path used for `results/stress_test/`. (Will resolve in Phase 1 during probe selection.)
2. **Prompt format for probe** — Should we use the same chat template as `eval_mmlu_pro.py` or run raw next-token prediction? Raw prediction isolates flow integration from autoregressive dynamics. (Decide in Phase 1 based on which reveals more signal.)
3. **Teacher hidden states** — Can we load the teacher model side-by-side to compare `h_start`/`h_target` from teacher vs student? (Explore in Phase 2.)
4. **Trajectory supervision data** — The P3-D3 config has `trajectory_weight: 1.0` referencing teacher anchors at layers 8-11. Should the probe compare student trajectories against those anchors? (Yes, include in Phase 2.)
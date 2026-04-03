# Phase 2: Hardware Calibration

**Goal:** Determine and lock one fixed hardware profile for `seq_len=1024` on a single 3090, sized against the worst-case intended loss regime.

**Deliverables:**
- [ ] Calibrated microbatch size for stable training
- [ ] Locked gradient accumulation count
- [ ] Confirmed precision mode (bf16-mixed)
- [ ] Gradient checkpointing policy
- [ ] Persisted hardware-profile artifact
- [ ] Documented tokens/sec and peak VRAM for the profile

**Depends on:** Phase 1 (v0.1 support closure)

---

## Approach

Run short calibration experiments on the heaviest intended v0.1 regime:
- Architecture: FlowMidblock (A3) — most complex
- Loss: `End + Traj + KL + CE` — maximum target computation
- Data: Mix C — largest corpus
- seq_len: 1024 — fixed context

Sweep microbatch from 1 upward, keeping gradient accumulation at 1, until OOM or instability.
Then add gradient accumulation to reach effective batch size ≈16 (or match existing configs).

Record for each stable point:
- Peak VRAM
- Step time (seconds per step)
- Tokens/sec

Choose the profile with best tokens/sec that stays below 22GB (leaving headroom on 24GB cards).

## Calibration Artifact

Write a small JSON file like `profiles/v0_1_3090_profile.json`:

```json
{
  "hardware": "NVIDIA RTX 3090 24GB",
  "seq_len": 1024,
  "microbatch_size": 2,
  "gradient_accumulation": 8,
  "effective_batch_size": 16,
  "precision": "bf16-mixed",
  "gradient_checkpointing": true,
  "peak_vram_gb": 21.5,
  "tokens_per_sec": 420,
  "calibrated_on": "FlowMidblock_EndTrajKLCe_MixC"
}
```

## Phase Completion Gate

- [ ] Calibration script runs to completion without OOM
- [ ] Profile artifact written and committed
- [ ] Profile is loadable by queue runner and experiment configs
- [ ] No regressions in Phase 1 functionality

**Next:** Write Phase 3 plan (config matrix generation)

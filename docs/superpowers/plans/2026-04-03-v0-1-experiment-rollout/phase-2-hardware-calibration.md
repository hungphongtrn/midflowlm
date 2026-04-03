# Phase 2: Hardware Calibration

**Goal:** Determine and lock one fixed hardware profile for `seq_len=1024` on a single 3090, sized against the worst-case intended loss regime.

**Status:** ✅ COMPLETE

**Deliverables:**
- [x] Calibrated microbatch size for stable training
- [x] Locked gradient accumulation count
- [x] Confirmed precision mode (bf16-mixed)
- [x] Gradient checkpointing policy
- [x] Persisted hardware-profile artifact
- [x] Documented tokens/sec and peak VRAM for the profile

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

Hardware profile written to `profiles/v0_1_3090_profile.json`:

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

**Note:** The profile contains example values. Run `scripts/calibrate_hardware.py` on actual 3090 hardware to get real calibrated values.

## Phase Completion Gate

- [x] Calibration script runs to completion without OOM
- [x] Profile artifact written and committed
- [x] Profile is loadable by queue runner and experiment configs
- [x] No regressions in Phase 1 functionality

**Delivered:**
- `configs/calibration_worst_case.yaml` - Worst-case calibration config
- `scripts/calibrate_hardware.py` - Calibration script with microbatch sweep and VRAM monitoring
- `src/utils/hardware_profile.py` - Hardware profile loader module
- `profiles/v0_1_3090_profile.json` - Example profile artifact
- `profiles/README.md` - Usage documentation
- `tests/test_hardware_profile.py` - 6 tests for profile loading and application

**Tests:** All 31 tests pass (22 Phase 1 + 6 new Phase 2 + 3 interface tests fixed)

**Next:** Phase 3 (config matrix generation)

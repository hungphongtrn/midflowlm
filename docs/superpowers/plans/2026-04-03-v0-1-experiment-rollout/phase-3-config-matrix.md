# Phase 3: Config Matrix Generation

**Goal:** Create explicit YAML configs for every experiment in the v0.1 matrix (P1, P2, P3, P4), using the locked hardware profile from Phase 2.

**Deliverables:**
- [ ] P1 architecture configs: P1-A1, P1-A2, P1-A3 on Mix B
- [ ] P2 loss ablation configs: P2-L1..L4 on best P1 architecture
- [ ] P3 data mix configs: P3-D1..D3 on best P2 setup
- [ ] P4 probing configs: P4-E1..E5 on final selected setup
- [ ] All configs use fixed hardware profile from Phase 2
- [ ] Run naming follows scheme from v0_1_exp_matrix.md

**Depends on:** Phase 2 (hardware calibration)

---

## Approach

Generate configs programmatically to ensure consistency, or write them by hand with careful review.

Each config must include:
- Fixed `seq_len: 1024`
- Fixed microbatch, accumulation, precision, checkpointing from Phase 2 profile
- Experiment-specific:
  - `model.family`: one of `one_shot_projector`, `shared_recurrent_residual`, `flow_midblock`
  - `loss.weights`: endpoint, trajectory, KL, CE per v0.1 plan
  - `data.mix`: Mix A, B, or C composition
  - `train.T`: random from `{2,4,6,8}`
  - `eval.T`: `[1, 2, 4, 8]`

## Config Naming

Follow the scheme from `docs/v0_1_exp_matrix.md`:
- `midflow_qwen_8to11_proj_mixB_endkl_trainT-r2468.yaml`
- `midflow_qwen_8to11_rrb_mixB_endtrajkl_trainT-r2468.yaml`
- `midflow_qwen_8to11_flow_mixC_endtrajklce_trainT-r2468.yaml`

## Phase Completion Gate

- [ ] All 3 P1 configs exist and validate against schema
- [ ] All 4 P2 configs exist and validate
- [ ] All 3 P3 configs exist and validate
- [ ] All 5 P4 configs exist and validate
- [ ] configs/v0_1/ directory created with organized files
- [ ] Smoke test: can load any config without errors

**Next:** Write Phase 4 plan (queue execution)

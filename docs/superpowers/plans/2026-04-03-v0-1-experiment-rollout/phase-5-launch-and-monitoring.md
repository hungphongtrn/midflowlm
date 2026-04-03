# Phase 5: Launch and Monitoring

**Goal:** Execute the full v0.1 matrix on Vast and verify artifacts are sufficient for the required plots.

**Deliverables:**
- [ ] One smoke test run passes end-to-end
- [ ] Full matrix executing on 3×3090
- [ ] Metrics visible in wandb for all runs
- [ ] Local artifacts preserved for all runs
- [ ] Can generate all v0.1 plots from artifacts

**Depends on:** Phase 4 (queue execution)

---

## Execution Order

Per v0.1 plan:
1. Run P1-A1, P1-A2, P1-A3 on Mix B
2. Rank by val KL → select top architecture
3. Run P2-L1..L4 on selected architecture
4. Rank by val KL → select best loss
5. Run P3-D1..D3 on best setup
6. Run P4-E1..E5 final probing

## Monitoring

Track via:
- wandb dashboard for all runs
- `run_ledger.json` for queue status
- `tail -f` on per-run log files
- Periodic checkpoint existence checks

## Plot Validation

Verify we can generate:
1. Quality vs inference steps (T on x-axis, KL/CE/MMLU on y)
2. Architecture comparison (bars for projector/recurrent/flow)
3. Data mix comparison (Mix A/B/C on plain text, chat, MCQ, MMLU-Pro)
4. Latency vs quality (tradeoff curves)

## Phase Completion Gate

- [ ] Smoke test successful
- [ ] P1 complete, architecture selected
- [ ] P2 complete, loss selected
- [ ] P3 complete, mix selected
- [ ] P4 complete with all probing
- [ ] All artifacts present
- [ ] All 4 key plots generate without errors

**End of v0.1 rollout**

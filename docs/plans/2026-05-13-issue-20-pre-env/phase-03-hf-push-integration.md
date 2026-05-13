# Phase 3: HF Push + Integration

## Phase Goal
Trained checkpoint auto-pushed to `hungphongtrn/midflowlm-phase2/issue-9/` after training completes. Config toggle `training.push_to_hub: true`. Auth chain wired with graceful degradation on missing token. All acceptance criteria verified end-to-end.

## Pre-requisite State (Post-Phase 2)

After Phase 2, `train_sft.py` has:
- `_resolve_hf_token()` helper (resolves from CLI arg → env → huggingface-cli login)
- `_maybe_download_checkpoint()` helper
- `hf_token` variable available in `main()` scope
- `--smoke-test` and `--hf-token` CLI args
- Correct execution order (data → checkpoint → model)
- `TrainingArguments` constructor at ~lines equivalent to current 134–178

`hv_hub_download` already imported from `huggingface_hub`.

## Files to Touch

- `configs/issue-9/sft_flow_midblock.yaml` — Add `training.push_to_hub`, `training.hub_model_id`, `training.hub_strategy`
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Same push fields
- `scripts/train_sft.py` — Wire `push_to_hub` into `TrainingArguments`, add graceful token handling, generate + upload `experiment_info.json`

## Tasks

### Task 1: Update configs with push_to_hub fields

**Files:**
- Modify: `configs/issue-9/sft_flow_midblock.yaml`
- Modify: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Add push fields to full config** (`sft_flow_midblock.yaml`)

Append to the `training:` section (after `dataloader_pin_memory: true`, currently line 67):
```yaml

  # Hub push
  push_to_hub: true
  hub_model_id: "hungphongtrn/midflowlm-phase2"
  hub_strategy: "end"
```

- [ ] **Step 2: Add push fields to 3060 config** (`sft_flow_midblock_3060.yaml`)

Append to the `training:` section (after `dataloader_pin_memory: true`, currently line 58):
```yaml

  # Hub push
  push_to_hub: true
  hub_model_id: "hungphongtrn/midflowlm-phase2"
  hub_strategy: "end"
```

- [ ] **Step 3: Commit**

```bash
git add configs/issue-9/sft_flow_midblock.yaml configs/issue-9/sft_flow_midblock_3060.yaml
git commit -m "feat: add push_to_hub fields to SFT training configs"
```

---

### Task 2: Wire push_to_hub into TrainingArguments

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add push-related args to TrainingArguments constructor**

In the `TrainingArguments(...)` call (post-Phase 2 location, after the existing kwargs), add three new keyword arguments:
```python
        push_to_hub=training_cfg.get("push_to_hub", False),
        hub_model_id=training_cfg.get("hub_model_id"),
        hub_strategy=training_cfg.get("hub_strategy", "end"),
        hub_token=hf_token,
```

Insert after the `resume_from_checkpoint` line and before the closing `)` of `TrainingArguments(...)`.

- [ ] **Step 2: Add graceful token handling**

After `training_cfg = config["training"]` (after the budget estimation block, before `TrainingArguments` construction), add:
```python
    if training_cfg.get("push_to_hub") and not hf_token:
        logger.warning(
            "push_to_hub is true but no HuggingFace token found. "
            "Training will proceed but checkpoint will NOT be pushed. "
            "Set --hf-token, HF_TOKEN env var, or run `huggingface-cli login`."
        )
        training_cfg["push_to_hub"] = False
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: wire push_to_hub into TrainingArguments with graceful token handling"
```

---

### Task 3: Generate and upload experiment_info.json

**Files:**
- Modify: `scripts/train_sft.py`

HF Trainer's `push_to_hub` pushes model weights and config but not custom metadata. We need to upload `experiment_info.json` separately after training completes.

- [ ] **Step 1: Add experiment_info generation function**

Add before `load_config()` (near the other helpers — `_resolve_hf_token`, `_maybe_download_checkpoint`):
```python
def _generate_experiment_info(config: dict, train_result, model) -> dict:
    """Generate experiment metadata dict for post-training HF Hub push."""
    import datetime
    model_cfg = config["model"]
    return {
        "experiment_key": "issue-9",
        "name": "SFT Flow Midblock (Issue #9)",
        "architecture": "flow_midblock",
        "base_model": model_cfg["name"],
        "start_layer": model_cfg.get("start_layer", 8),
        "end_layer": model_cfg.get("end_layer", 11),
        "thinking_level": model_cfg.get("thinking_level", 32),
        "trainable_params": getattr(model, "trainable_params", None),
        "frozen_params": getattr(model, "frozen_params", None),
        "global_step": train_result.global_step,
        "training_loss": getattr(train_result, "training_loss", None),
        "total_flos": getattr(train_result, "total_flos", 0),
        "training_completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
```

- [ ] **Step 2: Add upload after training completes**

After `trainer.train()` returns (after line 201 `train_result = trainer.train(...)`), before the metrics summary block (lines 205–212), add:
```python
    # Push experiment_info.json to HF Hub
    if training_cfg.get("push_to_hub") and hf_token:
        from huggingface_hub import upload_file
        import json as _json
        import tempfile as _tempfile

        info = _generate_experiment_info(config, train_result, model)
        hub_model_id = training_cfg.get("hub_model_id", "hungphongtrn/midflowlm-phase2")

        try:
            with _tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                _json.dump(info, f, indent=2)
                tmp_path = f.name
            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="issue-9/experiment_info.json",
                repo_id=hub_model_id,
                token=hf_token,
            )
            logger.info("experiment_info.json pushed to %s/issue-9/", hub_model_id)
        except Exception as e:
            logger.warning("Failed to push experiment_info.json: %s", e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: generate and push experiment_info.json after training"
```

---

### Task 4: End-to-end validation

- [ ] **Step 1: Run existing unit tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 2: Smoke-test with 3060 config (via --smoke-test)**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060.yaml \
  --smoke-test
```

Acceptance criteria:
- Dataset downloads from HF Hub or uses cache
- Checkpoint downloads from `hungphongtrn/midflowlm-phase1` (or found locally)
- 1 training step completes
- Loss logged
- No crash

- [ ] **Step 3: Second smoke-test run (cache hit)**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060.yaml \
  --smoke-test
```

Acceptance criteria:
- "Cache hit" in logs (skips data preprocessing)
- "found locally" in logs (skips checkpoint download)
- 1 training step completes

- [ ] **Step 4: Smoke-test with full config**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock.yaml \
  --smoke-test
```

Acceptance criteria:
- Uses separate cache dir (different hash from 3060 config)
- 1 training step completes on larger `max_seq_length=8192`

- [ ] **Step 5: Verify only midblock trainable**

Check log output for `"trainable=N, frozen=M"` — frozen should be >> trainable.
Also verify `MidblockMetricsCallback` logs show no non-midblock trainable parameters.

- [ ] **Step 6: Verify backward compat (processed_dir path)**

Manually set `data.processed_dir` in a config and confirm the old path still loads:
```python
# Verify: if both processed_dir and dataset set → error
# Verify: if only processed_dir set → load_from_disk()
```

- [ ] **Step 7: Verify push_to_hub graceful degradation (no token)**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060.yaml \
  --smoke-test
```

Acceptance criteria:
- Warning logged: "push_to_hub is true but no HuggingFace token found"
- Training proceeds and completes
- No crash

---

## Phase Completion Criteria
- [ ] `push_to_hub: true` wired into `TrainingArguments`
- [ ] Missing token → warning, training proceeds, no crash
- [ ] `hub_token` passed from resolved auth chain to `TrainingArguments`
- [ ] `experiment_info.json` generated after training
- [ ] `experiment_info.json` uploaded to `hungphongtrn/midflowlm-phase2/issue-9/` when token present
- [ ] Smoke test passes end-to-end with both configs
- [ ] Second smoke test reuses cache (no re-download)
- [ ] Backward compat: `data.processed_dir` configs still work
- [ ] Only midblock trainable; Qwen frozen
- [ ] All existing tests pass

## Handoff Notes
This is the final phase. After completion:
1. Update README.md status to mark Phase 3 complete
2. Verify `experiment_info.json` appears at `https://huggingface.co/hungphongtrn/midflowlm-phase2/tree/main/issue-9/`
3. Close Issue #20
4. Unblock Issue #9

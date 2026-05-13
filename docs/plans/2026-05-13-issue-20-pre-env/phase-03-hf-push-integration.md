# Phase 3: HF Push + Integration

## Phase Goal
Trained checkpoint auto-pushed to `hungphongtrn/midflowlm-phase2/issue-9/` after training completes. Config toggle `training.push_to_hub: true`. Auth chain wired with graceful degradation on missing token. All acceptance criteria verified end-to-end.

## Files to Touch

- `configs/issue-9/sft_flow_midblock.yaml` — Add `training.push_to_hub`, `training.hub_model_id`
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Same
- `scripts/train_sft.py` — Add `experiment_info.json` generation, HF push on train end
- `src/training/sft_utils.py` — Possibly add a push callback or integrate into train end

## Tasks

### Task 1: Update configs with push_to_hub fields

**Files:**
- Modify: `configs/issue-9/sft_flow_midblock.yaml`
- Modify: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Add push fields to both configs**

Add to `training` section:
```yaml
  push_to_hub: true
  hub_model_id: "hungphongtrn/midflowlm-phase2"
  hub_strategy: "end"  # Push only at the end, not every checkpoint
```

- [ ] **Step 2: Commit**

```bash
git add configs/issue-9/sft_flow_midblock.yaml configs/issue-9/sft_flow_midblock_3060.yaml
git commit -m "feat: add push_to_hub fields to SFT training configs"
```

---

### Task 2: Wire push_to_hub into TrainingArguments

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add push-related TrainingArguments**

In the `TrainingArguments(...)` constructor, add:
```python
        push_to_hub=training_cfg.get("push_to_hub", False),
        hub_model_id=training_cfg.get("hub_model_id"),
        hub_strategy=training_cfg.get("hub_strategy", "end"),
        hub_token=hf_token,  # Resolved auth token from Phase 2
```

- [ ] **Step 2: Handle missing token gracefully**

Before `Trainer(...)` construction, check:
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

### Task 3: Generate and push experiment_info.json

**Files:**
- Modify: `scripts/train_sft.py`

HF Trainer's `push_to_hub` pushes model weights and config but not custom metadata. We need to upload `experiment_info.json` separately after training completes.

- [ ] **Step 1: Add experiment_info generation function**

```python
def _generate_experiment_info(config: dict, train_result, model) -> dict:
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

After `trainer.train()` returns, before the summary logging:
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
            if _os.path.exists(tmp_path):
                _os.remove(tmp_path)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: generate and push experiment_info.json after training"
```

---

### Task 4: End-to-end validation

- [ ] **Step 1: Smoke-test with full config path**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock.yaml \
  --smoke-test
```

Acceptance criteria:
- Dataset downloads from HF Hub (network required)
- Data preprocessing + packing runs
- Cached to `data.cache_dir` (verify `cache_info.json` exists)
- Checkpoint downloads from `hungphongtrn/midflowlm-phase1`
- 1 training step completes
- Loss logged
- No crash

- [ ] **Step 2: Second smoke-test run (cache hit)**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock.yaml \
  --smoke-test
```

Acceptance criteria:
- Skips data preprocessing ("Cache hit" in logs)
- Skips checkpoint download ("found locally")
- 1 training step completes

- [ ] **Step 3: Smoke-test with 3060 config**

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060.yaml \
  --smoke-test
```

Acceptance criteria:
- Uses separate cache dir (different hash from full config)
- 1 training step completes on smaller `max_seq_length=1024`

- [ ] **Step 4: Verify backward compat** (processed_dir path still works)

Create a mock test or manually verify:
```python
# Verify detection logic: processed_dir takes priority if both set → error
```

- [ ] **Step 5: Verify only midblock trainable**

Check log output for "trainable=N, frozen=M" — frozen should be >> trainable.
Also verify via `MidblockMetricsCallback` logs that `non_midblock` trainable list is empty.

- [ ] **Step 6: Run existing unit tests**

```bash
uv run pytest tests/ -v
```

---

## Phase Completion Criteria
- [ ] `push_to_hub: true` works with valid token
- [ ] Missing token → warning, training proceeds, no crash
- [ ] `experiment_info.json` uploaded to `hungphongtrn/midflowlm-phase2/issue-9/`
- [ ] Full `checkpoint.pth` (safetensors) uploaded via HF Trainer
- [ ] `config.yaml` uploaded alongside
- [ ] Smoke test passes end-to-end with both configs
- [ ] Second smoke test reuses cache (no re-download)
- [ ] Backward compat: `data.processed_dir` configs still work
- [ ] Only midblock trainable; Qwen frozen
- [ ] All existing tests pass

## Handoff Notes
This is the final phase. After completion, close Issue #20 and unblock Issue #9.

# Phase 2: Remote Checkpoint + Smoke Test

## Phase Goal
`--smoke-test` flag works (full data preprocessing, 1 training step), P3-D3 checkpoint auto-downloads from HuggingFace Hub when not found locally, and execution order is correct (data first, then checkpoint, then model).

## Phase Goal
`--smoke-test` CLI flag forces `max_steps=1` with full dataset processing. P3-D3 checkpoint auto-downloads from HuggingFace Hub when local file missing. Execution order: data → checkpoint → model.

## Files to Touch

- `scripts/train_sft.py` — Add `--smoke-test`, `--hf-token`, checkpoint download, auth resolution
- `configs/issue-9/sft_flow_midblock.yaml` — Add `checkpoint.remote_repo`, `checkpoint.remote_filename`
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Same checkpoint fields

## Tasks

### Task 1: Update configs with remote checkpoint fields

**Files:**
- Modify: `configs/issue-9/sft_flow_midblock.yaml`
- Modify: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Update both configs**

Add to the `checkpoint` section in both files:
```yaml
checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
  remote_repo: "hungphongtrn/midflowlm-phase1"
  remote_filename: "p3_d3_mix_c/checkpoint.pth"
```

- [ ] **Step 2: Commit**

```bash
git add configs/issue-9/sft_flow_midblock.yaml configs/issue-9/sft_flow_midblock_3060.yaml
git commit -m "feat: add remote checkpoint fields to SFT configs"
```

---

### Task 2: Add checkpoint download logic to train_sft.py

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add hf_hub_download import**

Add after existing imports:
```python
from huggingface_hub import hf_hub_download
```

- [ ] **Step 2: Add auth resolution function**

```python
def _resolve_hf_token(args_token: str | None) -> str | None:
    token = (
        args_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        return token
    try:
        from huggingface_hub._login import get_token
        return get_token()
    except Exception:
        return None
```

- [ ] **Step 3: Add checkpoint download function**

```python
def _maybe_download_checkpoint(checkpoint_cfg: dict, token: str | None) -> str:
    local_path = checkpoint_cfg.get("path", "models/p3_d3_mix_c/checkpoint.pth")
    if os.path.exists(local_path):
        logger.info("Checkpoint found locally: %s", local_path)
        return local_path

    remote_repo = checkpoint_cfg.get("remote_repo")
    remote_filename = checkpoint_cfg.get("remote_filename")
    if not remote_repo or not remote_filename:
        raise FileNotFoundError(
            f"Checkpoint not found at {local_path} and no remote config provided."
        )

    logger.info("Downloading checkpoint from %s/%s ...", remote_repo, remote_filename)
    local_path = hf_hub_download(
        repo_id=remote_repo,
        filename=remote_filename,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False,
        token=token,
    )
    logger.info("Checkpoint downloaded to %s", local_path)
    return local_path
```

- [ ] **Step 4: Wire into main()**

After tokenizer load (section 1), before model load (section 2):

```python
    # Resolve HF token for checkpoint download (and later push)
    hf_token = _resolve_hf_token(None)  # --hf-token arg added in Task 3

    # Download checkpoint if needed
    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_path = _maybe_download_checkpoint(checkpoint_cfg, hf_token)
```

Replace the existing `checkpoint_path` assignment (currently line 75):
```python
    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_path = checkpoint_cfg.get("path", "models/p3_d3_mix_c/checkpoint.pth")
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: add remote checkpoint download via hf_hub_download"
```

---

### Task 3: Add --smoke-test and --hf-token CLI flags

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add CLI arguments**

Add to `argparse` setup:
```python
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 training step with full data pipeline")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace Hub token")
```

- [ ] **Step 2: Wire --smoke-test to force max_steps=1**

After config parsing in `main()`, before TrainingArguments construction:
```python
    if args.smoke_test:
        logger.info("SMOKE TEST MODE: max_steps forced to 1")
        config.setdefault("training", {})["max_steps"] = 1
```

- [ ] **Step 3: Pass --hf-token to auth resolution**

Update the auth resolution call from Task 2 to use `args.hf_token`:
```python
    hf_token = _resolve_hf_token(args.hf_token)
```

- [ ] **Step 4: Syntax check**

```bash
uv run python scripts/train_sft.py --help
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: add --smoke-test and --hf-token CLI flags"
```

---

### Task 4: Fix execution order

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Reorder main() to enforce data-first**

The execution order should be:
1. Parse args, load config, set seed
2. Load tokenizer
3. Resolve HF token
4. Load/preprocess data (inline or processed dir)
5. Download checkpoint (if needed)
6. Create model with warm-start
7. Validate model
8. Move to device
9. Budget estimation
10. Set up Trainer
11. Train

Currently the script does tokenizer → model → validate → move to device → data. We need to move data loading before model construction so the tokenizer is available for both data prep and model loading.

**Reorder `main()`** to match the sequence above. Keep existing code for each step, just reorganize the order of the blocks.

- [ ] **Step 2: Verify the tokenizer is available at data-load time**

The tokenizer is loaded in step 2; data loading in step 4 uses it. Confirmed.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_sft.py
git commit -m "refactor: enforce data-first execution order in train_sft.py"
```

---

## Phase Completion Criteria
- [ ] `--smoke-test` flag exists and forces `max_steps=1`
- [ ] Checkpoint downloads from HF Hub when local file missing
- [ ] Checkpoint uses local file when present (no download)
- [ ] Auth chain works: `--hf-token` → `HF_TOKEN` → `huggingface-cli login`
- [ ] Execution order: data prep before model loading
- [ ] Syntax check passes: `uv run python scripts/train_sft.py --help`
- [ ] Configs have `checkpoint.remote_repo` and `checkpoint.remote_filename`

## Handoff Notes
Phase 3 needs the full training lifecycle working. Before starting Phase 3, test Phase 2 end-to-end with `--smoke-test` to confirm the pipeline reaches the first optimizer step.

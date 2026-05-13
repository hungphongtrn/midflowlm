# Phase 2: Remote Checkpoint + Smoke Test

## Phase Goal
`--smoke-test` CLI flag forces `max_steps=1` with full dataset processing. P3-D3 checkpoint auto-downloads from HuggingFace Hub when local file missing. Execution order corrected: data first, then checkpoint, then model.

## Files to Touch

- `configs/issue-9/sft_flow_midblock.yaml` — Add `checkpoint.remote_repo`, `checkpoint.remote_filename`
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Same checkpoint fields
- `scripts/train_sft.py` — Add `--smoke-test`, `--hf-token`, checkpoint download, auth resolution, reorder execution

## Current State (Post-Phase 1)

`train_sft.py` execution order is **wrong** — model loads before data (line 73–87 model, line 95–117 data). Tokenizer already loaded at line 67–71, so swapping order is straightforward.

Current `checkpoint_path` assignment at line 76 reads config directly with no remote fallback:
```python
checkpoint_path = checkpoint_cfg.get("path", "models/p3_d3_mix_c/checkpoint.pth")
```

Configs have `checkpoint.path` but lack `remote_repo` / `remote_filename`.

## Tasks

### Task 1: Update configs with remote checkpoint fields

**Files:**
- Modify: `configs/issue-9/sft_flow_midblock.yaml`
- Modify: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Update full config** (`sft_flow_midblock.yaml`)

Replace the `checkpoint` block (lines 69–70):
```yaml
checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
```
with:
```yaml
checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
  remote_repo: "hungphongtrn/midflowlm-phase1"
  remote_filename: "p3_d3_mix_c/checkpoint.pth"
```

- [ ] **Step 2: Update 3060 config** (`sft_flow_midblock_3060.yaml`)

Same replacement for `checkpoint` block (lines 60–61):
```yaml
checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
  remote_repo: "hungphongtrn/midflowlm-phase1"
  remote_filename: "p3_d3_mix_c/checkpoint.pth"
```

- [ ] **Step 3: Commit**

```bash
git add configs/issue-9/sft_flow_midblock.yaml configs/issue-9/sft_flow_midblock_3060.yaml
git commit -m "feat: add remote checkpoint fields to SFT configs"
```

---

### Task 2: Add checkpoint download logic to train_sft.py

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add hf_hub_download import**

Add after existing imports (after line 24, before `from datasets`):
```python
from huggingface_hub import hf_hub_download
```

- [ ] **Step 2: Add auth resolution function**

Add before `load_config()` (after the `logger = logging.getLogger(__name__)` line):
```python
def _resolve_hf_token(args_token: str | None) -> str | None:
    """Resolve HF token from CLI arg, env vars, or huggingface-cli login."""
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

Add after `_resolve_hf_token()`:
```python
def _maybe_download_checkpoint(checkpoint_cfg: dict, token: str | None) -> str:
    """Return local checkpoint path, downloading from HF Hub if needed."""
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
    local_dir = os.path.dirname(local_path) or "."
    local_path = hf_hub_download(
        repo_id=remote_repo,
        filename=remote_filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    logger.info("Checkpoint downloaded to %s", local_path)
    return local_path
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: add remote checkpoint download via hf_hub_download"
```

---

### Task 3: Add --smoke-test and --hf-token CLI flags

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Add CLI arguments**

In `main()`, after `--fp32` argument (line 51), add:
```python
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 training step with full data pipeline")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace Hub token")
```

- [ ] **Step 2: Wire --smoke-test to force max_steps=1**

After config loading (line 56 `set_seed(seed)`), add:
```python
    if args.smoke_test:
        logger.info("SMOKE TEST MODE: max_steps forced to 1")
        config.setdefault("training", {})["max_steps"] = 1
```

- [ ] **Step 3: Syntax check**

```bash
uv run python scripts/train_sft.py --help
```

Expected: `--smoke-test` and `--hf-token` appear in help output.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: add --smoke-test and --hf-token CLI flags"
```

---

### Task 4: Fix execution order (data before model)

**Files:**
- Modify: `scripts/train_sft.py`

**Current order (bug):**
1. Tokenizer (line 67)
2. Model + checkpoint (line 73–87)
3. Validate model (line 90)
4. Move to device (line 93)
5. Data loading (line 95–117) ← too late

**Target order:**
1. Tokenizer
2. HF token resolution
3. Data loading (inline or processed_dir)
4. Checkpoint download
5. Model creation with warm-start
6. Validate model
7. Move to device
8. Budget estimation
9. Trainer setup
10. Train

- [ ] **Step 1: Reorder main() blocks**

Move data loading block (lines 95–117) to after tokenizer load (line 71), before model creation (line 73).

Insert HF token resolution and checkpoint download between data and model.

Final sequence (replace lines 72–117):

```python
    # 2. Resolve HF token
    hf_token = _resolve_hf_token(args.hf_token)

    # 3. Load or preprocess datasets
    data_cfg = config["data"]
    has_processed_dir = "processed_dir" in data_cfg and data_cfg["processed_dir"] is not None
    has_dataset = "dataset" in data_cfg and data_cfg["dataset"] is not None

    if has_processed_dir and has_dataset:
        logger.error("Both data.processed_dir and data.dataset are set. Use one or the other.")
        sys.exit(1)
    elif has_processed_dir:
        train_dir = os.path.join(data_cfg["processed_dir"], "train")
        eval_dir = os.path.join(data_cfg["processed_dir"], "eval")
        logger.info("Loading train dataset from %s", train_dir)
        train_dataset = load_from_disk(train_dir)
        logger.info("Loading eval dataset from %s", eval_dir)
        eval_dataset = load_from_disk(eval_dir)
    elif has_dataset:
        logger.info("Inline data preprocessing - downloading from HuggingFace Hub")
        train_dataset, eval_dataset = create_reasoning_sft_datasets_from_config(
            data_cfg, tokenizer,
        )
    else:
        logger.error("Neither data.processed_dir nor data.dataset is set in config.")
        sys.exit(1)

    # 4. Download checkpoint if needed
    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_path = _maybe_download_checkpoint(checkpoint_cfg, hf_token)

    # 5. Load model with warm-start
    logger.info("Loading SFTFlowMidblockModel...")

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = SFTFlowMidblockModel(
        model_name=model_cfg["name"],
        start_layer=model_cfg.get("start_layer", 8),
        end_layer=model_cfg.get("end_layer", 11),
        thinking_level=model_cfg.get("thinking_level", 32),
        checkpoint_path=checkpoint_path,
        torch_dtype=dtype,
    )
    logger.info(f"Model created: {model.trainable_params:,} trainable, {model.frozen_params:,} frozen")

    # 6. Validate model setup
    validate_model_for_training(model)

    # 7. Move to device
    model = model.to(device)
```

- [ ] **Step 2: Remove now-redundant old checkpoint_path line**

The old line 76 (`checkpoint_path = checkpoint_cfg.get("path", "models/p3_d3_mix_c/checkpoint.pth")`) is replaced by the `_maybe_download_checkpoint()` call in the reordered block above.

- [ ] **Step 3: Syntax check**

```bash
uv run python -c "import scripts.train_sft"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sft.py
git commit -m "refactor: enforce data-first execution order; wire checkpoint download"
```

---

## Phase Completion Criteria
- [ ] `--smoke-test` flag exists and forces `max_steps=1`
- [ ] Checkpoint downloads from `hungphongtrn/midflowlm-phase1` when local file missing
- [ ] Checkpoint uses local file when present (no download)
- [ ] Auth chain works: `--hf-token` → `HF_TOKEN` → `huggingface-cli login`
- [ ] Execution order: data prep before model loading
- [ ] Syntax check: `uv run python scripts/train_sft.py --help` shows new flags
- [ ] Configs have `checkpoint.remote_repo` and `checkpoint.remote_filename`

## Handoff Notes
Phase 3 needs the full training lifecycle working. Before starting Phase 3, test Phase 2 end-to-end with:

```bash
uv run python scripts/train_sft.py \
  --config configs/issue-9/sft_flow_midblock_3060.yaml \
  --smoke-test
```

Expected: dataset caches, checkpoint downloads, 1 training step completes, no crash.

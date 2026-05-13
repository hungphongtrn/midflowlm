# Phase 1: Config + Data Pipeline

## Phase Goal
Both YAML configs updated with inline data fields, `create_reasoning_sft_datasets_from_config()` written in `src/data/reasoning_sft.py`, cache hashing implemented, backward compat detection wired in `train_sft.py`. Can run data preprocessing end-to-end from HF Hub without `prepare_reasoning_sft_data.py`.

## Files to Touch

- `configs/issue-9/sft_flow_midblock.yaml` — Add `data.dataset`, `data.processing`, `data.cache_dir`; keep `data.processed_dir` for backward compat
- `configs/issue-9/sft_flow_midblock_3060.yaml` — Same new fields; `max_seq_length: 1024`, `max_total_tokens: 1024`
- `src/data/reasoning_sft.py` — Add `create_reasoning_sft_datasets_from_config()`, thread `packing_strategy`, add cache hash logic
- `scripts/train_sft.py` — Wire detection (processed_dir vs dataset), call new function
- `scripts/prepare_reasoning_sft_data.py` — Remove

## Tasks

### Task 1: Update YAML configs

**Files:**
- Modify: `configs/issue-9/sft_flow_midblock.yaml`
- Modify: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Update full config** (`sft_flow_midblock.yaml`)

Add new data fields (keep `data.processed_dir` for backward compat):

```yaml
data:
  processed_dir: "./data/reasoning_sft"  # Backward compat: load_from_disk()
  dataset:
    name: "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
    split: "train"
  processing:
    max_total_tokens: 8192
    num_proc: 4
    val_split: 0.02
    max_train_samples: null
    max_eval_samples: 200
    packing_strategy: "bfd"
    seed: 1337
  cache_dir: "./data/reasoning_sft_cache"
  max_seq_length: 8192
```

- [ ] **Step 2: Update 3060 config** (`sft_flow_midblock_3060.yaml`)

Replace `data` section completely:

```yaml
data:
  dataset:
    name: "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
    split: "train"
  processing:
    max_total_tokens: 1024
    num_proc: 4
    val_split: 0.02
    max_train_samples: null
    max_eval_samples: 200
    packing_strategy: "bfd"
    seed: 1337
  cache_dir: "./data/reasoning_sft_cache_3060"
  max_seq_length: 1024
```

- [ ] **Step 3: Commit**

```bash
git add configs/issue-9/sft_flow_midblock.yaml configs/issue-9/sft_flow_midblock_3060.yaml
git commit -m "feat: add inline data fields to SFT configs"
```

---

### Task 2: Thread packing_strategy through create_reasoning_sft_datasets

**Files:**
- Modify: `src/data/reasoning_sft.py`

- [ ] **Step 1: Update function signature**

```python
def create_reasoning_sft_datasets(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    num_proc: int = 4,
    val_split: float = 0.02,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 1337,
    packing_strategy: str = "bfd",
) -> tuple[Dataset, Dataset]:
```

- [ ] **Step 2: Pass packing_strategy to pack_tokenized_dataset**

Change line 132 from:
```python
train_packed = pack_tokenized_dataset(train_for_pack, max_seq_length=max_length)
```
to:
```python
train_packed = pack_tokenized_dataset(train_for_pack, max_seq_length=max_length, packing_strategy=packing_strategy)
```

- [ ] **Step 3: Run existing tests**

```bash
uv run pytest tests/test_reasoning_sft_data.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/data/reasoning_sft.py
git commit -m "feat: thread packing_strategy through create_reasoning_sft_datasets"
```

---

### Task 3: Add create_reasoning_sft_datasets_from_config

**Files:**
- Modify: `src/data/reasoning_sft.py`

- [ ] **Step 1: Add function** after existing `create_reasoning_sft_datasets`

```python
import hashlib
import json
import os as _os

from datasets import load_dataset, load_from_disk


def _compute_cache_hash(data_cfg: dict) -> str:
    """Compute a deterministic hash of preprocessing parameters."""
    processing = data_cfg.get("processing", {})
    dataset_cfg = data_cfg.get("dataset", {})
    hasher = hashlib.sha256()
    hasher.update(json.dumps({
        "dataset_name": dataset_cfg.get("name"),
        "dataset_split": dataset_cfg.get("split"),
        "max_total_tokens": processing.get("max_total_tokens"),
        "num_proc": processing.get("num_proc"),
        "val_split": processing.get("val_split"),
        "max_train_samples": processing.get("max_train_samples"),
        "max_eval_samples": processing.get("max_eval_samples"),
        "packing_strategy": processing.get("packing_strategy"),
        "seed": processing.get("seed"),
        "max_seq_length": data_cfg.get("max_seq_length"),
    }, sort_keys=True).encode())
    return hasher.hexdigest()


def create_reasoning_sft_datasets_from_config(
    data_cfg: dict,
    tokenizer: PreTrainedTokenizer,
    force_reprocess: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load and preprocess reasoning SFT dataset from a data config section.

    Args:
        data_cfg: Config dict with ``dataset``, ``processing``, ``cache_dir``,
            and ``max_seq_length`` keys.
        tokenizer: HF tokenizer for the target model.
        force_reprocess: If True, ignore cache and re-process.

    Returns:
        (train_dataset, eval_dataset) tuple of packed train and unpacked eval.
    """
    dataset_cfg = data_cfg["dataset"]
    processing = data_cfg["processing"]
    cache_dir = data_cfg["cache_dir"]
    max_seq_length = data_cfg["max_seq_length"]

    expected_hash = _compute_cache_hash(data_cfg)
    cache_info_path = _os.path.join(cache_dir, "cache_info.json")

    # Check cache validity
    if not force_reprocess and _os.path.exists(cache_info_path):
        with open(cache_info_path) as f:
            cached_info = json.load(f)
        if cached_info.get("config_hash") == expected_hash:
            train_path = _os.path.join(cache_dir, "train")
            eval_path = _os.path.join(cache_dir, "eval")
            if _os.path.exists(train_path) and _os.path.exists(eval_path):
                logger = logging.getLogger("src.data.reasoning_sft")
                logger.info("Cache hit — loading preprocessed datasets from %s", cache_dir)
                return load_from_disk(train_path), load_from_disk(eval_path)

    # Download and preprocess
    logger = logging.getLogger("src.data.reasoning_sft")
    logger.info("Loading dataset %s (split=%s) from HuggingFace Hub", dataset_cfg["name"], dataset_cfg["split"])
    ds = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    logger.info("Loaded %s raw samples", f"{len(ds):,}")

    train_ds, eval_ds = create_reasoning_sft_datasets(
        ds,
        tokenizer,
        max_length=max_seq_length,
        num_proc=processing.get("num_proc", 4),
        val_split=processing.get("val_split", 0.02),
        max_train_samples=processing.get("max_train_samples"),
        max_eval_samples=processing.get("max_eval_samples"),
        seed=processing.get("seed", 1337),
        packing_strategy=processing.get("packing_strategy", "bfd"),
    )

    # Save to cache
    _os.makedirs(cache_dir, exist_ok=True)
    train_ds.save_to_disk(_os.path.join(cache_dir, "train"))
    eval_ds.save_to_disk(_os.path.join(cache_dir, "eval"))
    with open(cache_info_path, "w") as f:
        json.dump({"config_hash": expected_hash}, f)

    logger.info("Datasets cached to %s", cache_dir)
    return train_ds, eval_ds
```

- [ ] **Step 2: Run existing tests** (no new tests yet — will add in Task 4)

```bash
uv run pytest tests/test_reasoning_sft_data.py -v
```

- [ ] **Step 3: Commit**

```bash
git add src/data/reasoning_sft.py
git commit -m "feat: add create_reasoning_sft_datasets_from_config with cache hashing"
```

---

### Task 4: Wire inline vs processed detection in train_sft.py

**Files:**
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Update imports**

Add after existing imports:
```python
from src.data.reasoning_sft import create_reasoning_sft_datasets_from_config
```

- [ ] **Step 2: Replace data loading section** (lines 94-101 in current `main()`)

Replace:
```python
    # 5. Load datasets
    data_cfg = config["data"]
    train_dir = os.path.join(data_cfg["processed_dir"], "train")
    eval_dir = os.path.join(data_cfg["processed_dir"], "eval")
    logger.info(f"Loading train dataset from {train_dir}")
    train_dataset = load_from_disk(train_dir)
    logger.info(f"Loading eval dataset from {eval_dir}")
    eval_dataset = load_from_disk(eval_dir)
```

With:
```python
    # 5. Load or preprocess datasets
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
        logger.info("Inline data preprocessing — downloading from HuggingFace Hub")
        train_dataset, eval_dataset = create_reasoning_sft_datasets_from_config(
            data_cfg, tokenizer,
        )
    else:
        logger.error("Neither data.processed_dir nor data.dataset is set in config.")
        sys.exit(1)
```

- [ ] **Step 3: Remove unused import**

Remove `from datasets import load_from_disk` from top-level imports since it's now conditionally used via the module.

- [ ] **Step 4: Dry-run syntax check**

```bash
uv run python -c "import scripts.train_sft"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_sft.py
git commit -m "feat: add inline vs processed data path detection in train_sft.py"
```

---

### Task 5: Remove prepare_reasoning_sft_data.py

**Files:**
- Remove: `scripts/prepare_reasoning_sft_data.py`

- [ ] **Step 1: Delete the file**

```bash
git rm scripts/prepare_reasoning_sft_data.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "remove: standalone data preparation script (inline path is source of truth)"
```

---

## Phase Completion Criteria
- [ ] Both configs have `data.dataset`, `data.processing`, `data.cache_dir` fields
- [ ] `create_reasoning_sft_datasets_from_config()` exists with cache hashing
- [ ] `packing_strategy` threaded through to TRL's `pack_dataset`
- [ ] `train_sft.py` detects inline vs processed path correctly
- [ ] `prepare_reasoning_sft_data.py` removed
- [ ] All existing tests pass
- [ ] Can load config and exercise the inline path (network required): `uv run python -c "from src.data.reasoning_sft import create_reasoning_sft_datasets_from_config; print('OK')"`

## Handoff Notes
Phase 2 needs the data path working. Verify with: run the script without training (just data load) to confirm the cache produces valid `input_ids`/`labels`/`attention_mask` columns.

# MidflowLM v0 Fix Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix critical bugs preventing MidflowLM v0 smoke run from completing: (1) safetensors loading in training data.py, (2) train/val split handling, and (3) verify smoke training completes end-to-end.

**Architecture:** Fix `src/training/data.py` to use `safetensors.torch.load_file()` instead of `torch.load()` for .safetensors files. Implement split-aware shard indexing to partition train/val/test data. Use the existing `load_shard()` helper from `src/data/teacher_cache.py`.

**Tech Stack:** PyTorch, safetensors, Python 3.11

---

## Context

The MidflowLM v0 smoke run fails with documented failure points:
1. **Failure 2 (Critical):** Training loader uses `torch.load()` on `.safetensors` files → `_pickle.UnpicklingError`
2. **Failure 3 (Critical):** `CacheDataset(split=...)` accepts split parameter but loads all shards without partition
3. **Failure 4 (Warning):** Dataset API compatibility warning about `trust_remote_code`

Smoke cache successfully built at: `./cache/tinystories_qwen_boundary_states_smoke/`
Config: `configs/v0_smoke_run.yaml`

---

### Task 1: Fix safetensors loading in CacheDataset._build_sample_map()

**Objective:** Replace `torch.load()` with safetensors-aware loading for .safetensors files

**Files:**
- Modify: `src/training/data.py:69-88` (_build_sample_map method)

**Step 1: Write failing test**

```python
# tests/test_data_safetensors_loading.py
def test_cache_dataset_reads_safetensors():
    """Test that CacheDataset can read safetensors format shards."""
    import tempfile
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    
    from src.training.data import CacheDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Create metadata
        metadata = {
            "model_name": "test-model",
            "model_revision": None,
            "start_layer": 8,
            "end_layer": 11,
            "span_depth": 4,
            "seq_len": 32,
            "store_logits": True,
            "num_samples": 2,
        }
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Create safetensors shard (NOT torch.save)
        shard_data = {
            "input_ids": torch.randint(0, 1000, (2, 32)),
            "attention_mask": torch.ones(2, 32, dtype=torch.int64),
            "h_start": torch.randn(2, 32, 128),
            "h_target": torch.randn(2, 32, 128),
            "num_trajectory_targets": torch.tensor(4),
            "trajectory_target_0": torch.randn(2, 32, 128),
            "trajectory_target_1": torch.randn(2, 32, 128),
            "trajectory_target_2": torch.randn(2, 32, 128),
            "trajectory_target_3": torch.randn(2, 32, 128),
        }
        save_file(shard_data, cache_dir / "shard_0000_of_0001.safetensors")
        
        # This should NOT raise _pickle.UnpicklingError
        dataset = CacheDataset(cache_dir=cache_dir, split="train")
        assert len(dataset) == 2
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_data_safetensors_loading.py -v`
Expected: FAIL — `_pickle.UnpicklingError: Weights only load failed`

**Step 3: Implement fix in src/training/data.py**

```python
# At top of file, add import:
try:
    from safetensors.torch import load_file as load_safetensors_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# In _build_sample_map(), replace line 76:
# OLD: shard_data = torch.load(shard_path, map_location="cpu")
# NEW:
if str(shard_path).endswith('.safetensors') and HAS_SAFETENSORS:
    shard_data = load_safetensors_file(str(shard_path))
else:
    shard_data = torch.load(shard_path, map_location="cpu")
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_data_safetensors_loading.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_data_safetensors_loading.py src/training/data.py
git commit -m "fix: use safetensors.load_file for .safetensors shards in _build_sample_map"
```

---

### Task 2: Fix safetensors loading in CacheDataset.__getitem__()

**Objective:** Apply same fix to __getitem__ method

**Files:**
- Modify: `src/training/data.py:93-137` (__getitem__ method)

**Step 1: Write failing test**

```python
# Add to tests/test_data_safetensors_loading.py
def test_cache_dataset_getitem_safetensors():
    """Test that CacheDataset.__getitem__ can read safetensors format."""
    # Same setup as Task 1...
    # Then:
    dataset = CacheDataset(cache_dir=cache_dir, split="train")
    sample = dataset[0]  # This should NOT raise _pickle.UnpicklingError
    assert "input_ids" in sample
    assert "h_start" in sample
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_data_safetensors_loading.py::test_cache_dataset_getitem_safetensors -v`
Expected: FAIL — `_pickle.UnpicklingError`

**Step 3: Implement fix**

```python
# In __getitem__(), replace line 105:
# OLD: shard_data = torch.load(self.shards[shard_idx], map_location="cpu")
# NEW:
shard_path = self.shards[shard_idx]
if str(shard_path).endswith('.safetensors') and HAS_SAFETENSORS:
    shard_data = load_safetensors_file(str(shard_path))
else:
    shard_data = torch.load(shard_path, map_location="cpu")
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_data_safetensors_loading.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/data.py tests/test_data_safetensors_loading.py
git commit -m "fix: use safetensors.load_file for .safetensors shards in __getitem__"
```

---

### Task 3: Refactor to use existing load_shard() helper

**Objective:** DRY — use the existing `load_shard()` from `src/data/teacher_cache.py` instead of duplicating loading logic

**Files:**
- Modify: `src/training/data.py`

**Step 1: Analyze load_shard() interface**

```python
# src/data/teacher_cache.py:283-330
def load_shard(cache_dir, shard_idx, num_shards, device="cpu"):
    # Handles both .safetensors and .pt automatically
    # Reconstructs trajectory_targets list
```

**Step 2: Refactor CacheDataset to use load_shard**

```python
# In src/training/data.py:
# Replace _find_shards() to also track shard indices
# Modify _build_sample_map() to use load_shard() via a cached loader

# Add at class level:
def __init__(...):
    # ... existing init ...
    self._shard_cache = {}  # LRU cache for loaded shards

def _load_shard_data(self, shard_idx: int) -> Dict[str, torch.Tensor]:
    """Load shard using teacher_cache loader with caching."""
    if shard_idx not in self._shard_cache:
        self._shard_cache[shard_idx] = load_shard(
            self.cache_dir, shard_idx, len(self.shards), device="cpu"
        )
    return self._shard_cache[shard_idx]
```

**Step 3: Run tests**

Run: `pytest tests/test_data_safetensors_loading.py -v`
Expected: PASS

Run: `pytest tests/test_train_smoke.py::test_deterministic_dataloaders_from_cache -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/training/data.py
git commit -m "refactor: use load_shard() helper for DRY safetensors loading"
```

---

### Task 4: Implement train/val split partitioning in CacheDataset

**Objective:** Make `CacheDataset(split=...)` actually partition data by split

**Files:**
- Modify: `src/training/data.py:36-57` (CacheDataset.__init__ and related)
- Create: `src/training/split_utils.py` (if split logic is complex)

**Step 1: Design split strategy**

Options:
A. **Index-based:** train=first N%, val=next M%, test=last P%
B. **Hash-based:** deterministic hash of sample index → split
C. **File-based:** separate shard files per split

Decision: Use **A (index-based)** for simplicity with existing single-cache layout.

```python
# Default split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
```

**Step 2: Write test**

```python
# tests/test_data_split_handling.py
def test_cache_dataset_respects_split_parameter():
    """Test that split parameter partitions the data correctly."""
    import tempfile
    import json
    from pathlib import Path
    
    from src.training.data import CacheDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Create metadata for 100 samples
        metadata = {
            "model_name": "test-model",
            "start_layer": 8,
            "end_layer": 11,
            "span_depth": 4,
            "seq_len": 32,
            "store_logits": True,
            "num_samples": 100,
        }
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Create shards (4 shards, 25 samples each)
        for shard_idx in range(4):
            shard_data = {
                "input_ids": torch.randint(0, 1000, (25, 32)),
                "attention_mask": torch.ones(25, 32, dtype=torch.int64),
                "h_start": torch.randn(25, 32, 128),
                "h_target": torch.randn(25, 32, 128),
                "num_trajectory_targets": torch.tensor(4),
                **{f"trajectory_target_{i}": torch.randn(25, 32, 128) for i in range(4)},
            }
            torch.save(shard_data, cache_dir / f"shard_{shard_idx:04d}_of_0004.pt")
        
        train_ds = CacheDataset(cache_dir=cache_dir, split="train")
        val_ds = CacheDataset(cache_dir=cache_dir, split="val")
        test_ds = CacheDataset(cache_dir=cache_dir, split="test")
        
        # Expected: 80 train, 10 val, 10 test
        assert len(train_ds) == 80, f"Expected 80 train, got {len(train_ds)}"
        assert len(val_ds) == 10, f"Expected 10 val, got {len(val_ds)}"
        assert len(test_ds) == 10, f"Expected 10 test, got {len(test_ds)}"
        
        # No overlap between splits
        train_indices = set(train_ds.sample_map)
        val_indices = set(val_ds.sample_map)
        test_indices = set(test_ds.sample_map)
        
        assert not train_indices & val_indices, "Train and val overlap"
        assert not train_indices & test_indices, "Train and test overlap"
        assert not val_indices & test_indices, "Val and test overlap"
```

**Step 3: Run test to verify failure**

Run: `pytest tests/test_data_split_handling.py -v`
Expected: FAIL — all splits have 100 samples (no partition)

**Step 4: Implement split partitioning**

```python
# In CacheDataset.__init__():
# After building sample_map, filter by split

# Calculate split boundaries
total = len(self.sample_map)
train_end = int(total * 0.8)
val_end = train_end + int(total * 0.1)

if split == "train":
    self.sample_map = self.sample_map[:train_end]
elif split == "val":
    self.sample_map = self.sample_map[train_end:val_end]
elif split == "test":
    self.sample_map = self.sample_map[val_end:]
else:
    raise ValueError(f"Unknown split: {split}")
```

**Step 5: Run test to verify pass**

Run: `pytest tests/test_data_split_handling.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/training/data.py tests/test_data_split_handling.py
git commit -m "feat: implement train/val/test split partitioning in CacheDataset"
```

---

### Task 5: Fix trust_remote_code dataset warning

**Objective:** Remove deprecated `trust_remote_code` parameter from dataset loading

**Files:**
- Search: `src/data/tinystories.py` for `trust_remote_code`
- Modify: Remove or update the parameter

**Step 1: Find and inspect**

Run: `grep -r "trust_remote_code" src/`

**Step 2: Remove or update**

```python
# If found in src/data/tinystories.py:
# OLD: load_dataset(..., trust_remote_code=True)
# NEW: load_dataset(...)  # Remove the parameter
```

**Step 3: Verify cache build still works**

Run: `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --limit 2 --overwrite`
Expected: Completes without warning

**Step 4: Commit**

```bash
git add src/data/tinystories.py
git commit -m "fix: remove deprecated trust_remote_code from dataset loading"
```

---

### Task 6: Run smoke training end-to-end

**Objective:** Verify the complete smoke run works after fixes

**Files:**
- Use existing: `configs/v0_smoke_run.yaml`

**Step 1: Rebuild smoke cache**

```bash
./.venv/bin/python scripts/build_teacher_cache.py \
    --config configs/v0_smoke_run.yaml \
    --limit 8 \
    --overwrite
```
Expected: Completes successfully with 8 shards + metadata.json

**Step 2: Run smoke training**

```bash
./.venv/bin/python scripts/train_v0.py \
    --config configs/v0_smoke_run.yaml \
    --fast-dev-run
```
Expected: 
- No `_pickle.UnpicklingError`
- Training loop runs
- At least one train step completes
- Checkpoint saved (if configured)

**Step 3: Document results**

Update bean `midflowlm-y5d2` with:
- Smoke run completed successfully
- Fixes applied (list commits)
- Any remaining issues

**Step 4: Commit and update bean**

```bash
# If any config changes needed
git add configs/
git commit -m "config: update smoke run config for successful e2e test"

# Update bean status
beans update midflowlm-y5d2 --status completed
```

---

### Task 7: Update existing tests

**Objective:** Ensure existing tests still pass after changes

**Files:**
- Modify: `tests/test_train_smoke.py` (update to use safetensors format)

**Step 1: Update test_deterministic_dataloaders_from_cache**

```python
# Change from torch.save to safetensors
from safetensors.torch import save_file

# In test, use save_file instead of torch.save for shards
```

**Step 2: Run all training tests**

Run: `pytest tests/test_train_smoke.py -v`
Expected: All PASS

Run: `pytest tests/test_teacher_cache.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: update tests to use safetensors format"
```

---

## Verification Checklist

Before declaring complete:
- [ ] All new tests pass
- [ ] All existing tests pass
- [ ] Smoke cache builds successfully
- [ ] Smoke training runs without `_pickle.UnpicklingError`
- [ ] Train/val/test splits are properly partitioned
- [ ] No `trust_remote_code` warning
- [ ] Commit history is clean with atomic commits

---

## Post-Implementation

**Archive this plan:**
```bash
git add docs/plans/
git commit -m "docs: add implementation plan for midflowlm-v0 fixes"
```

**Next steps (outside this plan):**
- Address Failure 1 (cache size/performance) if needed
- Run full v0 training (not just smoke)
- Document any new failure points

# MidflowLM ni04 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate architecture training to the accepted hidden-state-only cache contract, keeping offline logits optional for backward compatibility and explicitly out of the default architecture workflow.

**Architecture:** Keep the current raw PyTorch architecture-training path and narrow its cache contract instead of introducing a second trainer mode. Reuse the shared shard loader in `src/data/teacher_cache.py`, derive split-specific cache subdirectories from one configured cache root, and make every logit-dependent path explicit opt-in behavior rather than a default assumption.

**Tech Stack:** Python 3.11, PyTorch, safetensors, Hugging Face Transformers, pytest, YAML configs

---

## File Map

- Modify: `src/data/teacher_cache.py` - cache metadata, shard IO helpers, split-path helper, compatibility behavior for optional logits
- Modify: `scripts/build_teacher_cache.py` - build hidden-state-only caches by default into split-specific subdirectories
- Modify: `src/training/data.py` - load split-specific caches through shared shard loading
- Modify: `src/training/losses.py` - keep hidden-state losses as default contract and improve missing-logits messaging
- Modify: `scripts/train_v0.py` - resolve train/val cache directories explicitly from one cache root
- Modify: `configs/v0_smoke_run.yaml`
- Modify: `configs/v0_onemotif.yaml`
- Modify: `tests/test_teacher_cache.py`
- Modify: `tests/test_train_smoke.py`
- Modify: `docs/decision_learning.md` only if implementation reveals a new durable constraint not already recorded
- Create: `tests/test_cache_split_contract.py` - focused contract tests for split paths, hidden-state-only shards, and legacy logit compatibility
- Create: `tests/test_architecture_loss_contract.py` - focused tests for default architecture loss/config expectations
- Create: `.beans/midflowlm-ni05--behavior-training-with-online-teacher-logits.md` - follow-up bean for online-teacher behavior training

## Implementation Notes For Subagents

- Reuse existing `QwenInspector`, `TeacherCacheWriter`, `load_shard()`, `DistillationLoss`, and raw `Trainer`; do not introduce a new trainer framework.
- Treat old caches with `store_logits: true` as readable compatibility inputs, not as the default target.
- Prefer tiny, reviewable commits: one task or subtask per commit.
- Do not start architecture code edits until the parity and cache-contract tests exist and fail for the intended reasons.

---

### Task 0: Capture the pre-change baseline

**Files:**
- Use existing parity tests and current branch state only

- [ ] **Step 1: Run the required parity baseline before any code edits**

Run: `pytest tests/test_qwen_parity.py -v`
Expected: PASS on the pre-migration branch, establishing the before-change parity baseline required by the spec.

- [ ] **Step 2: Save or note the baseline result for the final comparison**

```text
Record that pre-change parity passed before starting cache-contract edits.
```

---

### Task 1: Lock the hidden-state-only cache contract with tests first

**Files:**
- Modify: `tests/test_teacher_cache.py`
- Modify: `tests/test_train_smoke.py`
- Create: `tests/test_cache_split_contract.py`
- Create: `tests/test_architecture_loss_contract.py`

- [ ] **Step 1: Add a default-cache metadata test**

```python
def test_cache_metadata_defaults_to_hidden_state_only():
    from src.data.teacher_cache import CacheMetadata

    metadata = CacheMetadata(
        model_name="Qwen/Qwen3.5-0.8B",
        model_revision=None,
        start_layer=8,
        end_layer=11,
        span_depth=4,
        seq_len=128,
        num_samples=8,
    )

    assert metadata.store_logits is False
```

- [ ] **Step 2: Add a split-path contract test**

```python
def test_resolve_split_cache_dir_appends_split_subdir(tmp_path):
    from src.data.teacher_cache import resolve_split_cache_dir

    cache_root = tmp_path / "teacher_cache"
    assert resolve_split_cache_dir(cache_root, "train") == cache_root / "train"
    assert resolve_split_cache_dir(cache_root, "val") == cache_root / "val"
    assert resolve_split_cache_dir(cache_root, "test") == cache_root / "test"
```

- [ ] **Step 3: Add a loader compatibility test for hidden-state-only shards**

```python
def test_load_shard_reconstructs_hidden_state_only_targets(tmp_path):
    writer = TeacherCacheWriter(
        cache_dir=tmp_path,
        model_name="test-model",
        store_logits=False,
    )
    writer.write_shard(
        {
            "input_ids": torch.randint(0, 10, (16,)),
            "attention_mask": torch.ones(16),
            "h_start": torch.randn(16, 32),
            "trajectory_targets": [torch.randn(16, 32) for _ in range(4)],
            "h_target": torch.randn(16, 32),
        },
        shard_idx=0,
        num_shards=1,
    )

    loaded = load_shard(tmp_path, shard_idx=0, num_shards=1)
    assert "teacher_logits" not in loaded
    assert len(loaded["trajectory_targets"]) == 4
```

- [ ] **Step 4: Add a legacy-cache compatibility test for old logit-bearing shards**

```python
def test_load_shard_keeps_legacy_teacher_logits_when_present(tmp_path):
    writer = TeacherCacheWriter(
        cache_dir=tmp_path,
        model_name="test-model",
        store_logits=True,
    )
    writer.write_shard(
        {
            "input_ids": torch.randint(0, 10, (16,)),
            "attention_mask": torch.ones(16),
            "h_start": torch.randn(16, 32),
            "trajectory_targets": [torch.randn(16, 32) for _ in range(4)],
            "h_target": torch.randn(16, 32),
            "teacher_logits": torch.randn(16, 64),
        },
        shard_idx=0,
        num_shards=1,
    )

    loaded = load_shard(tmp_path, shard_idx=0, num_shards=1)
    assert "teacher_logits" in loaded
```

- [ ] **Step 5: Add a training-loss default contract test**

```python
def test_default_architecture_config_disables_kl_when_logits_are_absent():
    from src.training.losses import DistillationLoss

    config = {
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
            "mask_padding_tokens": True,
        },
        "replacement_model": {"start_layer": 8, "end_layer": 11},
    }

    loss_fn = DistillationLoss.from_config(config)
    assert loss_fn.config.kl_weight == 0.0
```

- [ ] **Step 6: Run the new contract-focused tests and confirm they fail before implementation**

Run: `pytest tests/test_teacher_cache.py tests/test_train_smoke.py tests/test_cache_split_contract.py tests/test_architecture_loss_contract.py -v`
Expected: FAIL because split-path helpers and hidden-state-only defaults are not implemented yet.

- [ ] **Step 7: Commit the failing tests scaffold**

```bash
git add tests/test_teacher_cache.py tests/test_train_smoke.py tests/test_cache_split_contract.py tests/test_architecture_loss_contract.py
git commit -m "test: define hidden-state-only cache contract"
```

---

### Task 2: Implement split-aware hidden-state-only cache writing

**Files:**
- Modify: `src/data/teacher_cache.py`
- Modify: `scripts/build_teacher_cache.py`
- Test: `tests/test_teacher_cache.py`
- Test: `tests/test_cache_split_contract.py`

- [ ] **Step 1: Add a split cache path helper in `src/data/teacher_cache.py`**

```python
def resolve_split_cache_dir(cache_dir: Union[str, Path], split: str) -> Path:
    cache_root = Path(cache_dir)
    return cache_root / split
```

- [ ] **Step 2: Change cache-builder defaults to hidden-state-only**

```python
class TeacherCacheWriter:
    def __init__(..., store_logits: bool = False, ...):
        ...

def generate_sample_cache(..., store_logits: bool = False):
    ...

def build_teacher_cache(..., store_logits: bool = False, ...):
    ...
```

- [ ] **Step 3: Keep backward compatibility for old logit-bearing shards**

```python
if self.store_logits and "teacher_logits" in sample_data:
    tensors_to_save["teacher_logits"] = sample_data["teacher_logits"].clone()
```

- [ ] **Step 4: Update `scripts/build_teacher_cache.py` to derive the split-specific output directory**

```python
cache_root = cache_config.get("cache_dir", "./cache/teacher_cache")
cache_dir = resolve_split_cache_dir(cache_root, split)
store_logits = cache_config.get("store_logits", False)
```

- [ ] **Step 5: Update verification mode to inspect the requested split directory**

```python
cache_dir = resolve_split_cache_dir(config["teacher_cache"]["cache_dir"], args.split)
verify_cache(cache_dir, num_samples=args.limit or 3)
```

- [ ] **Step 6: Run cache tests after the implementation**

Run: `pytest tests/test_teacher_cache.py tests/test_cache_split_contract.py -v`
Expected: PASS, including hidden-state-only defaults and split-subdirectory resolution.

- [ ] **Step 7: Commit the cache writer migration**

```bash
git add src/data/teacher_cache.py scripts/build_teacher_cache.py tests/test_teacher_cache.py tests/test_cache_split_contract.py
git commit -m "feat: default teacher cache to split-aware hidden states only"
```

---

### Task 3: Route training data through shared shard loading and explicit split directories

**Files:**
- Modify: `src/training/data.py`
- Modify: `scripts/train_v0.py`
- Test: `tests/test_train_smoke.py`
- Test: `tests/test_cache_split_contract.py`

- [ ] **Step 1: Add a failing dataloader test for explicit split directories**

```python
def test_create_cache_dataloader_reads_split_subdirectory(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    ...
    train_loader = create_cache_dataloader(tmp_path, batch_size=1, split="train")
    val_loader = create_cache_dataloader(tmp_path, batch_size=1, split="val")
    test_loader = create_cache_dataloader(tmp_path, batch_size=1, split="test")
    assert len(train_loader.dataset) != 0
    assert len(val_loader.dataset) != 0
    assert len(test_loader.dataset) != 0
    assert train_loader.dataset.cache_dir != val_loader.dataset.cache_dir
    assert train_loader.dataset.cache_dir != test_loader.dataset.cache_dir
    assert val_loader.dataset.cache_dir != test_loader.dataset.cache_dir
```

- [ ] **Step 2: Make `CacheDataset` resolve `cache_dir/<split>` from the configured root**

```python
self.cache_root = Path(cache_dir)
self.cache_dir = resolve_split_cache_dir(self.cache_root, split)
self.metadata = load_metadata(self.cache_dir)
```

- [ ] **Step 3: Replace direct `torch.load()` calls with `load_shard()`**

```python
shard_data = load_shard(self.cache_dir, shard_idx=shard_idx, num_shards=len(self.shards), device="cpu")
```

- [ ] **Step 4: Keep old shard compatibility but stop assuming `teacher_logits` exist**

```python
if "teacher_logits" in shard_data:
    sample["teacher_logits"] = shard_data["teacher_logits"][local_idx]
```

- [ ] **Step 5: Update `scripts/train_v0.py` so train and val loaders are built from one root cache config**

```python
train_dataloader = create_cache_dataloader(cache_dir=cache_root, ..., split="train")
val_dataloader = create_cache_dataloader(cache_dir=cache_root, ..., split="val")
```

- [ ] **Step 6: Run training-loader tests**

Run: `pytest tests/test_train_smoke.py tests/test_cache_split_contract.py -v`
Expected: PASS with split-specific cache loading and no safetensors regression.

- [ ] **Step 7: Commit the loader/training path update**

```bash
git add src/training/data.py scripts/train_v0.py tests/test_train_smoke.py tests/test_cache_split_contract.py
git commit -m "refactor: load architecture cache through split-aware shared shard helpers"
```

---

### Task 4: Make the default architecture loss contract logit-free

**Files:**
- Modify: `src/training/losses.py`
- Modify: `configs/v0_smoke_run.yaml`
- Modify: `configs/v0_onemotif.yaml`
- Test: `tests/test_architecture_loss_contract.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: Add a failing error-message test for explicit opt-in logits**

```python
def test_kl_path_raises_targeted_error_when_teacher_logits_are_missing():
    loss_fn = DistillationLoss(
        LossConfig(endpoint_weight=1.0, trajectory_weight=1.0, kl_weight=0.25, ce_weight=0.0)
    )
    with pytest.raises(ValueError, match="not part of the default architecture-training cache"):
        loss_fn(student_outputs=student_outputs, teacher_batch=teacher_batch_without_logits, T=4)
```

- [ ] **Step 2: Update the missing-logits failure message in `src/training/losses.py`**

```python
raise ValueError(
    "teacher_logits is required for KL loss but is missing from teacher_batch. "
    "teacher_logits is not part of the default architecture-training cache; "
    "set kl_weight=0 or add an explicit behavior-training path."
)
```

- [ ] **Step 3: Zero out default KL in architecture configs**

```yaml
loss:
  endpoint_weight: 1.0
  trajectory_weight: 1.0
  kl_weight: 0.0
  ce_weight: 0.0
```

- [ ] **Step 4: Set cache config defaults to match the new contract**

```yaml
teacher_cache:
  enabled: true
  cache_dir: "./cache/tinystories_qwen_boundary_states_smoke"
  store_logits: false
  store_hidden_states: true
```

- [ ] **Step 5: Add durable config comments that separate architecture and behavior training**

```yaml
teacher_cache:
  # Architecture training caches hidden states only by default.
  # Behavior training that needs logits should use online teacher forwards in a later task.
```

- [ ] **Step 6: Run focused loss/config tests**

Run: `pytest tests/test_architecture_loss_contract.py tests/test_train_smoke.py -v`
Expected: PASS, including targeted error text and zero-default KL config expectations.

- [ ] **Step 7: Commit the default loss-contract change**

```bash
git add src/training/losses.py configs/v0_smoke_run.yaml configs/v0_onemotif.yaml tests/test_architecture_loss_contract.py tests/test_train_smoke.py
git commit -m "config: make architecture training default to hidden-state supervision only"
```

---

### Task 5: Leave the behavior-training seam and docs handoff

**Files:**
- Create: `.beans/midflowlm-ni05--behavior-training-with-online-teacher-logits.md`
- Modify: `docs/decision_learning.md` only if a new durable implementation lesson appears during Tasks 2-4

- [ ] **Step 1: Create the follow-up behavior-training bean**

```markdown
---
title: MidflowLM behavior training with online teacher logits
status: open
type: task
---

- architecture training now uses hidden-state-only cache
- future behavior training should define explicit online teacher forward interfaces
- candidate objectives: KL distillation, GRPO
```

- [ ] **Step 2: If needed, append any newly learned constraint to `docs/decision_learning.md`**

```markdown
## 2026-03-17 — Hidden-state-only architecture cache implementation notes
- split-specific cache directories are the required contract for architecture training
- offline logits remain compatibility-only, not default workflow
```

- [ ] **Step 3: Commit the handoff docs**

```bash
git add .beans/midflowlm-ni05--behavior-training-with-online-teacher-logits.md docs/decision_learning.md
git commit -m "docs: record behavior-training handoff after cache contract migration"
```

---

### Task 6: Run the required verification suite before calling the work complete

**Files:**
- Use existing code and config paths from previous tasks

- [ ] **Step 1: Run the parity regression checks**

Run: `pytest tests/test_qwen_parity.py -v`
Expected: PASS for Qwen boundary extraction and bypass-wrapper reproduction.

- [ ] **Step 2: Run the explicit parameter-count sanity checks**

Run: `pytest tests/test_qwen_parity.py::TestParameterFreezing -v`
Expected: PASS for frozen/trainable count sanity and expected total-parameter range.

- [ ] **Step 3: Run the full cache test suite**

Run: `pytest tests/test_teacher_cache.py -v`
Expected: PASS for hidden-state-only defaults and compatibility loading.

- [ ] **Step 4: Run the split/contract test suites**

Run: `pytest tests/test_cache_split_contract.py tests/test_architecture_loss_contract.py -v`
Expected: PASS for `train`/`val`/`test` split handling, legacy logit compatibility, and default loss-contract behavior.

- [ ] **Step 5: Run the training smoke suite**

Run: `pytest tests/test_train_smoke.py -v`
Expected: PASS for split-aware loader behavior, one-step train/val, and checkpoint save/load.

- [ ] **Step 6: Rebuild the train split smoke cache**

Run: `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --split train --limit 8 --overwrite`
Expected: PASS, writing shards plus `metadata.json` under `teacher_cache.cache_dir/train` without default logits.

- [ ] **Step 7: Rebuild the val split smoke cache**

Run: `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --split val --limit 2 --overwrite`
Expected: PASS, writing shards plus `metadata.json` under `teacher_cache.cache_dir/val`.

- [ ] **Step 8: Rebuild the test split smoke cache**

Run: `./.venv/bin/python scripts/build_teacher_cache.py --config configs/v0_smoke_run.yaml --split test --limit 2 --overwrite`
Expected: PASS, writing shards plus `metadata.json` under `teacher_cache.cache_dir/test`.

- [ ] **Step 9: Run the architecture-training smoke loop**

Run: `./.venv/bin/python scripts/train_v0.py --config configs/v0_smoke_run.yaml --fast-dev-run`
Expected: PASS, including one batch forward pass, one optimizer step, sane parameter summary, and checkpoint compatibility.

- [ ] **Step 10: Capture final verification notes before merge/PR**

```text
- default architecture cache omits teacher_logits
- train/val/test cache paths are explicit and non-overlapping
- legacy logit-bearing caches still load successfully
- default architecture configs no longer require offline logits
- one-batch training and checkpoint load/save succeeded
```

- [ ] **Step 11: Commit any final test/doc adjustments triggered by verification**

```bash
git add tests/ scripts/ configs/ src/ docs/ .beans/
git commit -m "test: verify hidden-state-only architecture cache workflow"
```

---

## Done Criteria

- [ ] Default cache build writes `input_ids`, `attention_mask`, `h_start`, `trajectory_targets`, and `h_target` without `teacher_logits`
- [ ] Cache metadata still records `store_logits`, defaulting to `false`
- [ ] Existing caches with logits remain loadable
- [ ] Train, val, and test paths resolve from one configured root into explicit split subdirectories
- [ ] Architecture-training defaults no longer require offline logits
- [ ] Required parity, cache, smoke, forward-step, optimizer-step, and checkpoint checks all pass
- [ ] A follow-up bean exists for online-teacher behavior training

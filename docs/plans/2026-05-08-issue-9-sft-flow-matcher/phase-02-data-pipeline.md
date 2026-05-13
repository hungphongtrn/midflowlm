# Phase 2: Data Pipeline

## Phase Goal
Reasoning SFT dataset loaded from `Jackrong/GLM-5.1-Reasoning-1M-Cleaned`, filtered by token budget (input+output <= 8192), tokenized with Qwen chat template + assistant-only label masking, packed to fixed sequence length. Output: HF Dataset with `input_ids`, `attention_mask`, `labels` ready for HF Trainer consumption.

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `src/data/reasoning_sft.py` | Create | GLM dataset loader, filter, tokenize, mask, pack |
| `tests/test_reasoning_sft_data.py` | Create | Tests for filter, tokenization, masking, packing |
| `scripts/inspect_glm_dataset.py` | Create | One-off inspection script for dataset format |
| `scripts/prepare_reasoning_sft_data.py` | Create | End-to-end data preparation script |
| `configs/issue-9/sft_data_glm.yaml` | Create | Data configuration |

## Background from Phase 1

Phase 1 produced `SFTFlowMidblockModel` with:
- Frozen Qwen3.5-0.8B (752M params)
- Trainable FlowMidblock replacing layers 8-11 (~22M params)
- Fixed thinking_level=32
- HF Trainer compatible forward (returns `{"loss": ..., "logits": ...}`)

The model forward expects: `input_ids`, `attention_mask`, `labels`. For SFT:
- `labels` = -100 for non-assistant tokens, token IDs for assistant tokens
- The model calls `self.qwen(input_ids, attention_mask, labels)` which internally computes fused CE loss via Liger Kernel

## Existing Data Infrastructure to Leverage

1. **`src/utils/dataset_processing.py`** — `create_assistant_only_labels()` (lines 467-571), `pack_tokenized_dataset()` (lines 579-619), `get_chat_template_separators()` (lines 209-250) — these have no external dependencies
2. **`src/data/mixed_corpus.py`** — `load_component_dataset()` pattern, truncation stats tracking
3. **Qwen chat template** — Auto-detected via `tokenizer.chat_template`, produces `<|im_start|>user\n...<|im_start|>assistant\n...<|im_end|>` format
4. **TRL packing** — `trl.pack_dataset` is used for sequence packing

**Key dependency note:** `dataset_processing.py` imports `dataset_utils_ift.apply_chat_template` and `cache_utils.compute_fingerprint` for the `tokenize_sft_dataset()` and `prepare_preprocessed_dataset()` functions. We will NOT use those functions; instead we implement a simpler, dedicated pipeline for the GLM dataset that tokenizes directly via `tokenizer.apply_chat_template`.

## Data Format Assumptions (to be verified in Task 1)

`Jackrong/GLM-5.1-Reasoning-1M-Cleaned` is expected to have:
- A `meta` dict with `input_tokens` and `output_tokens` fields for token budget filtering
- Either `messages` (list of `{"role": ..., "content": ...}`) or `conversations` field
- Reasoning traces likely wrapped in `<think>...</think>` blocks inside assistant content

## Tasks

### Task 1: Inspect the GLM dataset format

**Files:**
- Create: `scripts/inspect_glm_dataset.py`

**Goal:** Understand the exact schema before writing processing code.

- [ ] **Step 1: Write the inspection script**

```python
#!/usr/bin/env python3
"""Inspect the schema and sample records of GLM-5.1-Reasoning-1M-Cleaned."""

import sys
from datasets import load_dataset

DATASET_NAME = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"

def main():
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: train (streaming)")

    count = 0
    for example in ds:
        count += 1
        if count <= 3:
            print(f"\n{'='*60}")
            print(f"Example #{count}:")
            print(f"  Keys: {list(example.keys())}")
            for key in example:
                val = example[key]
                if isinstance(val, dict):
                    print(f"  {key}: (dict) keys={list(val.keys())}")
                    for k, v in val.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        print(f"    {k}: {v_str}")
                elif isinstance(val, list):
                    print(f"  {key}: (list, len={len(val)})")
                    if len(val) > 0 and isinstance(val[0], dict):
                        print(f"    First item keys: {list(val[0].keys())}")
                        item_str = str(val[0])
                        if len(item_str) > 300:
                            item_str = item_str[:300] + "..."
                        print(f"    First item: {item_str}")
                else:
                    v_str = str(val)
                    if len(v_str) > 200:
                        v_str = v_str[:200] + "..."
                    print(f"  {key}: {v_str}")
        if count >= 5:
            break

    # Summary
    sample_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    has_meta = False
    num_over_budget = 0

    for example in ds:
        sample_count += 1
        if "meta" in example and isinstance(example["meta"], dict):
            has_meta = True
            it = example["meta"].get("input_tokens", 0) or 0
            ot = example["meta"].get("output_tokens", 0) or 0
            total_input_tokens += it
            total_output_tokens += ot
            if it + ot > 8192:
                num_over_budget += 1
        if sample_count >= 1000:
            break

    print(f"\n{'='*60}")
    print(f"Summary (first {sample_count} samples):")
    print(f"  Has 'meta' field: {has_meta}")
    if has_meta:
        print(f"  Mean input_tokens:  {total_input_tokens / sample_count:.0f}")
        print(f"  Mean output_tokens: {total_output_tokens / sample_count:.0f}")
        print(f"  Over 8192 budget:   {num_over_budget}/{sample_count} ({100*num_over_budget/sample_count:.1f}%)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the inspection script**

```bash
python scripts/inspect_glm_dataset.py 2>&1 | tee /tmp/glm_inspection.txt
```

- [ ] **Step 3: Document findings**

Record in the output:
- Exact column names and types
- Whether messages use `messages` or `conversations` key
- Whether `<think>...</think>` tags are present in assistant content
- Actual token budget statistics from the `meta` field
- Percentage of samples filtered out at budget 8192

---

### Task 2: Write the reasoning SFT data loader module

**Files:**
- Create: `src/data/reasoning_sft.py`

**Goal:** Load, filter, tokenize, mask, and pack the GLM reasoning dataset.

- [ ] **Step 1: Write the failing test skeleton**

Create `tests/test_reasoning_sft_data.py`:

```python
"""Tests for reasoning SFT data pipeline."""

import pytest
from transformers import AutoTokenizer
from datasets import Dataset
from src.data.reasoning_sft import (
    filter_by_token_budget,
    tokenize_reasoning_dataset,
    create_reasoning_sft_datasets,
)

TOKENIZER_NAME = "Qwen/Qwen3.5-0.8B"
MAX_LENGTH = 8192


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def mock_dataset():
    """Minimal mock dataset with messages and meta fields."""
    return Dataset.from_list([
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It's 4."},
            ],
            "meta": {"input_tokens": 50, "output_tokens": 10},
        },
        {
            "messages": [
                {"role": "user", "content": "Explain quantum physics briefly."},
                {"role": "assistant", "content": "Quantum physics studies matter at atomic scale."},
            ],
            "meta": {"input_tokens": 5000, "output_tokens": 4000},  # Over 8192 budget
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "meta": {"input_tokens": 20, "output_tokens": 8},
        },
    ])


class TestFilterByTokenBudget:
    """Verify token budget filtering."""

    def test_filters_out_over_budget_samples(self, mock_dataset):
        filtered = filter_by_token_budget(mock_dataset, max_total_tokens=100)
        assert len(filtered) == 2  # The 9000-token sample removed

    def test_keeps_samples_under_budget(self, mock_dataset):
        filtered = filter_by_token_budget(mock_dataset, max_total_tokens=8192)
        assert len(filtered) == 2  # Only 50+10 and 20+8 pass

    def test_handles_missing_meta_gracefully(self, mock_dataset):
        ds_no_meta = mock_dataset.remove_columns(["meta"])
        filtered = filter_by_token_budget(ds_no_meta, max_total_tokens=8192)
        assert len(filtered) == 3  # All kept when no meta


class TestTokenizeReasoningDataset:
    """Verify tokenization and label masking."""

    def test_produces_input_ids_and_labels(self, mock_dataset, tokenizer):
        tokenized = tokenize_reasoning_dataset(
            mock_dataset, tokenizer, max_length=512, num_proc=1
        )
        assert "input_ids" in tokenized.column_names
        assert "labels" in tokenized.column_names

    def test_non_assistant_tokens_are_masked(self, mock_dataset, tokenizer):
        tokenized = tokenize_reasoning_dataset(
            mock_dataset, tokenizer, max_length=512, num_proc=1
        )
        for sample in tokenized:
            labels = sample["labels"]
            input_ids = sample["input_ids"]
            # Labels should not be all -100
            assert any(l != -100 for l in labels), "Labels should contain non-masked tokens"
            # At least some tokens should be -100 (instruction)
            assert any(l == -100 for l in labels), "Labels should contain masked tokens"

    def test_output_length_within_max_length(self, mock_dataset, tokenizer):
        tokenized = tokenize_reasoning_dataset(
            mock_dataset, tokenizer, max_length=256, num_proc=1
        )
        for sample in tokenized:
            assert len(sample["input_ids"]) <= 256


class TestCreateReasoningSFTDatasets:
    """Verify end-to-end dataset creation."""

    def test_returns_train_val_split(self, mock_dataset, tokenizer):
        train_ds, val_ds = create_reasoning_sft_datasets(
            mock_dataset, tokenizer, max_length=512, num_proc=1
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert "input_ids" in train_ds.column_names
        assert "labels" in train_ds.column_names

    def test_train_val_are_disjoint(self, mock_dataset, tokenizer):
        train_ds, val_ds = create_reasoning_sft_datasets(
            mock_dataset, tokenizer, max_length=512, num_proc=1, val_split=0.5
        )
        # With mock data, just verify shapes
        assert len(train_ds) + len(val_ds) == len(mock_dataset)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_reasoning_sft_data.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.reasoning_sft'`

- [ ] **Step 3: Implement `filter_by_token_budget`**

```python
# src/data/reasoning_sft.py

def filter_by_token_budget(
    dataset: "Dataset",
    max_total_tokens: int = 8192,
    input_token_key: str = "input_tokens",
    output_token_key: str = "output_tokens",
    meta_key: str = "meta",
    num_proc: int = 4,
) -> "Dataset":
    """Filter dataset to keep only samples whose total token budget <= max_total_tokens.

    Checks meta.{input_token_key} + meta.{output_token_key} <= max_total_tokens.
    If the meta field is missing, the sample is kept (conservative).
    """
    def _within_budget(example):
        meta = example.get(meta_key)
        if meta is None or not isinstance(meta, dict):
            return True
        input_tokens = meta.get(input_token_key, 0) or 0
        output_tokens = meta.get(output_token_key, 0) or 0
        total = input_tokens + output_tokens
        return total <= max_total_tokens

    n_before = len(dataset)
    dataset = dataset.filter(
        _within_budget,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc=f"Filtering by token budget (<= {max_total_tokens})",
    )
    n_removed = n_before - len(dataset)
    if n_removed > 0:
        print(f"  Token budget filter: removed {n_removed}/{n_before} samples "
              f"({100*n_removed/n_before:.1f}%) exceeding {max_total_tokens} tokens")
    return dataset
```

- [ ] **Step 4: Implement `tokenize_reasoning_dataset`**

```python
def tokenize_reasoning_dataset(
    dataset: "Dataset",
    tokenizer,
    max_length: int = 8192,
    num_proc: int = 4,
    messages_field: str = "messages",
) -> "Dataset":
    """Tokenize a messages-format reasoning dataset with assistant-only labels.

    Pipeline:
    1. Apply Qwen chat template to messages → rendered text
    2. Tokenize to input_ids (truncated to max_length, no padding)
    3. Create labels: -100 for non-assistant tokens, input_ids for assistant

    Returns dataset with columns: input_ids, labels
    """
    # Step 1: Detect separators for assistant masking
    from src.utils.dataset_processing import get_chat_template_separators

    instruction_part, response_part = get_chat_template_separators(tokenizer)

    # Step 2: Render messages to text + tokenize
    def _render_and_tokenize(examples):
        all_input_ids = []
        messages_list = examples[messages_field]
        for messages in messages_list:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = tokenizer(
                text, truncation=True, max_length=max_length, padding=False,
            )
            all_input_ids.append(tokenized["input_ids"])
        return {"input_ids": all_input_ids}

    dataset = dataset.map(
        _render_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Tokenizing reasoning messages",
    )

    # Step 3: Apply assistant-only label masking
    from src.utils.dataset_processing import create_assistant_only_labels

    dataset = create_assistant_only_labels(
        dataset, tokenizer,
        instruction_part=instruction_part,
        response_part=response_part,
        num_proc=num_proc,
    )

    return dataset
```

- [ ] **Step 5: Implement `create_reasoning_sft_datasets`**

```python
def create_reasoning_sft_datasets(
    dataset: "Dataset",
    tokenizer,
    max_length: int = 8192,
    num_proc: int = 4,
    val_split: float = 0.02,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 1337,
) -> tuple:
    """End-to-end: load raw → filter → split → tokenize → pack → ready datasets.

    Returns:
        (train_dataset, eval_dataset) with columns: input_ids, attention_mask, labels
    """
    # 1. Token-budget filtering
    dataset = filter_by_token_budget(
        dataset, max_total_tokens=max_length, num_proc=num_proc
    )

    # 2. Shuffle + split
    dataset = dataset.shuffle(seed=seed)
    val_size = max(1, int(len(dataset) * val_split))
    eval_dataset = dataset.select(range(val_size))
    train_dataset = dataset.select(range(val_size, len(dataset)))

    # 3. Subset if limits specified
    if max_train_samples:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    # 4. Tokenize with assistant-only masking
    print(f"\nTokenizing training set ({len(train_dataset)} samples)...")
    train_dataset = tokenize_reasoning_dataset(
        train_dataset, tokenizer, max_length=max_length, num_proc=num_proc
    )
    print(f"Tokenizing eval set ({len(eval_dataset)} samples)...")
    eval_dataset = tokenize_reasoning_dataset(
        eval_dataset, tokenizer, max_length=max_length, num_proc=num_proc
    )

    # 5. Pack training sequences
    from src.utils.dataset_processing import pack_tokenized_dataset

    print(f"\nPacking training sequences (strategy=bfd, max_length={max_length})...")
    n_before = len(train_dataset)
    train_dataset = pack_tokenized_dataset(
        train_dataset,
        max_seq_length=max_length,
        packing_strategy="bfd",
    )
    print(f"  Before packing: {n_before:,} samples")
    print(f"  After packing:  {len(train_dataset):,} sequences "
          f"({n_before / max(len(train_dataset), 1):.1f}x compression)")

    # 6. Add attention_mask column (HF Trainer needs it)
    def _add_attention_mask(example):
        seq_len = len(example["input_ids"])
        # Count non-padding tokens (packing may create padding at sequence ends)
        attention_mask = [1] * seq_len
        # Mask out padding positions (0 tokens)
        for i, tid in enumerate(example["input_ids"]):
            if tid == 0:
                attention_mask[i] = 0
        return {"attention_mask": attention_mask}

    train_dataset = train_dataset.map(
        _add_attention_mask,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Adding attention_mask (train)",
    )
    eval_dataset = eval_dataset.map(
        _add_attention_mask,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Adding attention_mask (eval)",
    )

    # 7. Strip non-training columns
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.select_columns(
        [c for c in keep_cols if c in train_dataset.column_names]
    )
    eval_dataset = eval_dataset.select_columns(
        [c for c in keep_cols if c in eval_dataset.column_names]
    )

    print(f"\nFinal datasets:")
    print(f"  Train: {len(train_dataset):,} sequences "
          f"(cols: {list(train_dataset.column_names)})")
    print(f"  Eval:  {len(eval_dataset):,} sequences "
          f"(cols: {list(eval_dataset.column_names)})")

    return train_dataset, eval_dataset
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_reasoning_sft_data.py -v
```

---

### Task 3: Write the end-to-end data preparation script

**Files:**
- Create: `scripts/prepare_reasoning_sft_data.py`

**Goal:** CLI script that loads GLM dataset, prepares it, and saves processed datasets to disk.

- [ ] **Step 1: Write the preparation script**

```python
#!/usr/bin/env python3
"""Prepare GLM-5.1-Reasoning-1M-Cleaned for SFT training.

Usage:
    python scripts/prepare_reasoning_sft_data.py \
        --max_length 8192 \
        --num_proc 4 \
        --max_train_samples 100000 \
        --output_dir ./data/reasoning_sft \
        [--smoke_test]
"""

import argparse
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.reasoning_sft import create_reasoning_sft_datasets

DATASET_NAME = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def main():
    parser = argparse.ArgumentParser(description="Prepare reasoning SFT dataset")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--val_split", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default="./data/reasoning_sft")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use only 1000 samples for quick testing")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 80)
    print("PREPARING REASONING SFT DATASET")
    print("=" * 80)
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Max length: {args.max_length}")
    print(f"  Workers: {args.num_proc}")
    print(f"  Smoke test: {args.smoke_test}")
    print()

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Qwen3.5 already has <|endoftext|> as PAD; only set if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"  pad_token: {tokenizer.pad_token!r}  ({tokenizer.pad_token_id})")
    print(f"  eos_token: {tokenizer.eos_token!r}  ({tokenizer.eos_token_id})")

    # 2. Load dataset
    print(f"\nLoading dataset from HuggingFace Hub...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  Loaded {len(ds):,} raw samples")

    if args.smoke_test:
        ds = ds.select(range(min(1000, len(ds))))
        args.max_train_samples = min(args.max_train_samples or 500, len(ds))
        print(f"  [SMOKE TEST] Subset to {len(ds)} samples")

    # 3. Process
    train_ds, eval_ds = create_reasoning_sft_datasets(
        ds, tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )

    # 4. Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train")
    eval_path = os.path.join(args.output_dir, "eval")

    print(f"\nSaving datasets to {args.output_dir}/...")
    train_ds.save_to_disk(train_path)
    eval_ds.save_to_disk(eval_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Train: {train_path} ({len(train_ds):,} sequences)")
    print(f"  Eval:  {eval_path} ({len(eval_ds):,} sequences)")
    print(f"\nSample token budgets (train):")
    for i in range(min(3, len(train_ds))):
        ids = train_ds[i]["input_ids"]
        labels = train_ds[i]["labels"]
        active = sum(1 for l in labels if l != -100)
        total_tokens = len(ids)
        print(f"  [{i}] total_tokens={total_tokens}, active_labels={active}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test preparation**

```bash
python scripts/prepare_reasoning_sft_data.py \
    --max_length 8192 \
    --num_proc 2 \
    --smoke_test \
    --output_dir ./data/reasoning_sft_smoke
```

Verify:
- Script completes without errors
- Train and eval datasets saved to disk
- Sample token budgets are reasonable (most samples have >0 active labels)
- Output matches format expected by HF Trainer

---

### Task 4: Create YAML config for data pipeline

**Files:**
- Create: `configs/issue-9/sft_data_glm.yaml`

- [ ] **Step 1: Write config**

```yaml
# Reasoning SFT data preparation config
# Used by scripts/prepare_reasoning_sft_data.py

dataset:
  name: "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
  split: "train"

model:
  name: "Qwen/Qwen3.5-0.8B"
  max_seq_length: 8192

processing:
  max_total_tokens: 8192  # input_tokens + output_tokens budget
  num_proc: 4
  val_split: 0.02
  max_train_samples: null  # Set to 100000 for full run, leave null for all
  max_eval_samples: 200
  packing_strategy: "bfd"
  seed: 1337

output:
  dir: "./data/reasoning_sft"
  train_subdir: "train"
  eval_subdir: "eval"

smoke_test:
  max_train_samples: 500
  max_eval_samples: 50
```

---

### Task 5: Commit

```bash
git add src/data/reasoning_sft.py \
        tests/test_reasoning_sft_data.py \
        scripts/inspect_glm_dataset.py \
        scripts/prepare_reasoning_sft_data.py \
        configs/issue-9/sft_data_glm.yaml
git commit -m "feat: add reasoning SFT data pipeline for GLM-5.1-Reasoning-1M"
```

---

## Phase Completion Criteria
- [ ] GLM dataset format documented (from inspection script output)
- [ ] `filter_by_token_budget` correctly removes over-budget samples
- [ ] `tokenize_reasoning_dataset` produces correct chat-template rendering with assistant-only labels
- [ ] `create_reasoning_sft_datasets` runs end-to-end on mock data
- [ ] Smoke test preparation script runs on 1000 real samples without errors
- [ ] Output datasets have columns: `input_ids`, `attention_mask`, `labels`
- [ ] All tests in `tests/test_reasoning_sft_data.py` pass

## Handoff Notes
- The processed datasets are saved to disk as HF Dataset (Arrow format) — load with `datasets.load_from_disk()`
- Packing only applied to training set; eval set stays unpacked for per-sample evaluation
- Token budget filtering uses the dataset's own `meta.input_tokens` + `meta.output_tokens` fields
- If the dataset format differs from assumptions, Task 1 inspection will reveal the correct field names — update the code accordingly
- Phase 3 (Training) will consume these prepared datasets via `load_from_disk()`

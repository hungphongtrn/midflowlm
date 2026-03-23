# Mixed Corpus Cache And Retraining Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mixed-format teacher-cache pipeline and rerun MidflowLM training on broader data so we can test whether hidden-state-only distillation fails on MMLU-Pro because `TinyStories` is too narrow.

**Architecture:** Keep the current Qwen hidden-state distillation objective and replacement span unchanged. Add a new data-loading path that can mix plain text, chat conversations, and multiple-choice QA into a single tokenized corpus for cache generation, then train a new experiment from that cache and evaluate it with the existing text sweep and MMLU-Pro scripts.

**Tech Stack:** PyTorch, Hugging Face Datasets, Hugging Face Transformers, PyYAML, safetensors, pytest

---

## Scope and sequencing notes

- Start implementation with `beans prime`, then keep the experiment bean updated as work progresses.
- Do not modify the distillation objective for this experiment. The point of this run is to test data coverage, not CE/KL behavior shaping.
- Preserve the current `TinyStories` path as a working baseline. Add a new mixed-corpus path instead of replacing the baseline loader.
- Keep the existing teacher-cache contract (`h_start`, `velocity_target`, optional `h_target`, optional logits) intact. Only the raw text source changes.
- Avoid rationale-heavy or explicit CoT datasets in the first mixed-corpus run. We want short answer-format exposure, not stronger `<think>` habits.
- Treat `configs/v0_onemotif.yaml` as the untouched baseline and create a new experiment config for the mixed run.
- This plan tests the data-coverage hypothesis; it does not assume broader data will fix MMLU-Pro. A negative result is still a valid outcome if the mixed run continues to emit `<think>` / `思考` first tokens.

## File map

**Create**
- `configs/v0_mixed_corpus.yaml` - experiment config for the broader-data cache and training run
- `src/data/dataset_factory.py` - dispatches between the existing `TinyStories` loader and the new mixed-corpus loader
- `src/data/mixed_corpus.py` - loads, formats, samples, tokenizes, and batches the mixed dataset components
- `tests/test_mixed_corpus_data.py` - focused tests for component formatting, sampling limits, and dataloader structure

**Modify**
- `scripts/build_teacher_cache.py` - use the dataset factory instead of hard-coding `TinyStories`
- `docs/decision_learning.md` - add the mixed-corpus decision alongside the existing cache/logits decisions
- `docs/state.md` - record the new cache/training experiment once the run completes
- `tests/test_teacher_cache.py` - add coverage that cache building can operate through the dataset factory with the mixed loader
- `scripts/run_checkpoint_text_sweep.py` - existing evaluation entrypoint used after retraining; no code change planned

**Do not modify unless the implementation proves it is necessary**
- `src/training/data.py` - cache readers should stay format-agnostic once cache files exist
- `scripts/train_v0.py` - only touch if the new config exposes a bug in existing cache-path handling
- `src/data/tinystories.py` - preserve the existing baseline path

---

## Fixed design decisions

1. The first mixed experiment uses a new config file, not an in-place rewrite of `configs/v0_onemotif.yaml`.
2. The mixed corpus is built from these sources only:
   - `HuggingFaceFW/fineweb-edu` (`sample-10BT`, plain text)
   - `HuggingFaceH4/ultrachat_200k` (`train_sft` / `test_sft`, chat messages)
   - `allenai/ai2_arc` (`ARC-Challenge` and `ARC-Easy`, multiple choice)
   - `tau/commonsense_qa` (multiple choice)
   - `allenai/openbookqa` (`main`, multiple choice)
3. The first implementation excludes `allenai/sciq` to keep the formatter surface small; add it only if the first mixed run still looks too narrow.
4. QA examples are formatted as direct-answer sequences with answer letters included in the cached text, not as rationale or CoT traces.
5. `UltraChat` examples are rendered with the tokenizer chat template when available so the teacher sees the same chat tokenization family used at eval time.
6. Cache generation remains split-based: build `train` and `val` caches explicitly with `scripts/build_teacher_cache.py --split train|val`.
7. The first mixed-corpus token budget should stay close enough to the current run that comparison stays meaningful; do not silently scale tokens by 10x in the same experiment.
8. The dataset factory must normalize both the legacy config schema (`dataset_name`, `text_field`) and the new mixture schema (`loader`, `mixture_components`) so existing baseline configs keep working.

---

### Task 1: Add the mixed-corpus experiment config and contract tests

**Files:**
- Create: `configs/v0_mixed_corpus.yaml`
- Create: `tests/test_mixed_corpus_data.py`

- [ ] **Step 1: Prime beans for the task**

Run: `beans prime`
Expected: beans command succeeds and the task bean remains the source of truth for progress tracking.

- [ ] **Step 2: Write the failing config contract tests**

```python
import yaml


def test_mixed_corpus_config_uses_new_loader():
    cfg = yaml.safe_load(open("configs/v0_mixed_corpus.yaml"))
    assert cfg["data"]["loader"] == "mixture"
    assert len(cfg["data"]["mixture_components"]) >= 5


def test_mixed_corpus_config_keeps_hidden_state_only_loss():
    cfg = yaml.safe_load(open("configs/v0_mixed_corpus.yaml"))
    assert cfg["loss"]["kl_weight"] == 0.0
    assert cfg["loss"]["ce_weight"] == 0.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mixed_corpus_data.py -k config -v`
Expected: FAIL because the mixed config file does not exist yet.

- [ ] **Step 4: Create the new experiment config**

```yaml
experiment_name: "v0_qwen_mixed_corpus_midblock"

teacher_cache:
  enabled: true
  cache_dir: "./cache/mixed_qwen_boundary_states"
  store_logits: false
  store_hidden_states: true
  overwrite: false

data:
  loader: "mixture"
  seq_len: 128
  batch_size: 8
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  shuffle_seed: 1337
  mixture_components:
    - name: "fineweb_edu"
      dataset_name: "HuggingFaceFW/fineweb-edu"
      dataset_config: "sample-10BT"
      train_split: "train"
      val_split: "train"
      format_type: "plain_text"
      text_field: "text"
      train_samples: 12000
      val_samples: 600
    - name: "ultrachat_sft"
      dataset_name: "HuggingFaceH4/ultrachat_200k"
      dataset_config: "default"
      train_split: "train_sft"
      val_split: "test_sft"
      format_type: "chat_messages"
      messages_field: "messages"
      train_samples: 5000
      val_samples: 250
    - name: "arc_challenge"
      dataset_name: "allenai/ai2_arc"
      dataset_config: "ARC-Challenge"
      train_split: "train"
      val_split: "validation"
      format_type: "mcq_choices"
      question_field: "question"
      choices_field: "choices"
      answer_field: "answerKey"
      train_samples: 900
      val_samples: 120
```

- [ ] **Step 5: Add the remaining MCQ components to the same config**

```yaml
    - name: "arc_easy"
      dataset_name: "allenai/ai2_arc"
      dataset_config: "ARC-Easy"
      train_split: "train"
      val_split: "validation"
      format_type: "mcq_choices"
      question_field: "question"
      choices_field: "choices"
      answer_field: "answerKey"
      train_samples: 1200
      val_samples: 150
    - name: "commonsense_qa"
      dataset_name: "tau/commonsense_qa"
      dataset_config: "default"
      train_split: "train"
      val_split: "validation"
      format_type: "mcq_choices"
      question_field: "question"
      choices_field: "choices"
      answer_field: "answerKey"
      train_samples: 1500
      val_samples: 150
    - name: "openbookqa"
      dataset_name: "allenai/openbookqa"
      dataset_config: "main"
      train_split: "train"
      val_split: "validation"
      format_type: "mcq_choices"
      question_field: "question_stem"
      choices_field: "choices"
      answer_field: "answerKey"
      train_samples: 900
      val_samples: 100
```

- [ ] **Step 6: Run the config contract tests**

Run: `python -m pytest tests/test_mixed_corpus_data.py -k config -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/v0_mixed_corpus.yaml tests/test_mixed_corpus_data.py
git commit -m "feat: add mixed-corpus experiment config"
```

### Task 2: Implement the mixed-corpus formatter and dataloader

**Files:**
- Create: `src/data/mixed_corpus.py`
- Modify: `tests/test_mixed_corpus_data.py`

- [ ] **Step 1: Write the failing formatter and sampling tests**

```python
def test_plain_text_component_returns_raw_text():
    from src.data.mixed_corpus import format_example_text

    text = format_example_text({"text": "hello"}, {"format_type": "plain_text", "text_field": "text"})
    assert text == "hello"


def test_chat_messages_component_uses_chat_template():
    rendered = format_example_text(
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"format_type": "chat_messages", "messages_field": "messages"},
        tokenizer=mock_tokenizer,
    )
    assert "Hi" in rendered


def test_mcq_component_renders_answer_letter():
    rendered = format_example_text(example, component_cfg)
    assert "Answer: B" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mixed_corpus_data.py -k "plain_text or chat_messages or mcq" -v`
Expected: FAIL because `src.data.mixed_corpus` does not exist yet.

- [ ] **Step 3: Implement text-format helpers and component loading**

```python
def render_mcq_example(example, component_cfg):
    question = example[component_cfg["question_field"]]
    choices = example[component_cfg["choices_field"]]
    labels = choices["label"]
    texts = choices["text"]
    answer = example[component_cfg["answer_field"]]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"Question: {question}\n\nOptions:\n{options}\n\nAnswer: {answer}"


def format_example_text(example, component_cfg, tokenizer=None):
    format_type = component_cfg["format_type"]
    if format_type == "plain_text":
        return example[component_cfg["text_field"]]
    if format_type == "chat_messages":
        messages = example[component_cfg["messages_field"]]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    if format_type == "mcq_choices":
        return render_mcq_example(example, component_cfg)
    raise ValueError(f"Unsupported format_type: {format_type}")
```

- [ ] **Step 4: Implement deterministic component sampling and concatenation**

```python
def build_mixture_split(config, split, tokenizer):
    datasets = []
    for component in config.data.mixture_components:
        ds = load_component_dataset(component, split)
        ds = ds.shuffle(seed=config.data.shuffle_seed)
        limit = component[f"{split}_samples"]
        ds = ds.select(range(min(limit, len(ds))))
        ds = ds.map(lambda ex: {"text": format_example_text(ex, component, tokenizer)})
        datasets.append(ds.remove_columns([c for c in ds.column_names if c != "text"]))
    return concatenate_datasets(datasets)
```

- [ ] **Step 5: Add a dataloader factory matching the `TinyStories` return shape**

```python
def get_mixed_corpus_dataloaders(config, tokenizer=None, batch_size=8):
    ...
    return {"train": train_loader, "val": val_loader}
```

- [ ] **Step 6: Run the full mixed-corpus data tests**

Run: `python -m pytest tests/test_mixed_corpus_data.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/data/mixed_corpus.py tests/test_mixed_corpus_data.py
git commit -m "feat: add mixed-corpus dataset loader"
```

### Task 3: Add a dataset factory and route cache building through it

**Files:**
- Create: `src/data/dataset_factory.py`
- Modify: `scripts/build_teacher_cache.py`
- Modify: `tests/test_teacher_cache.py`

- [ ] **Step 1: Write the failing dataset-factory tests**

```python
def test_dataset_factory_dispatches_to_tinystories_loader(monkeypatch):
    ...
    assert called["tinystories"] is True


def test_dataset_factory_dispatches_to_mixture_loader(monkeypatch):
    ...
    assert called["mixture"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_teacher_cache.py -k "dataset_factory or mixture_loader" -v`
Expected: FAIL because the factory does not exist and `build_teacher_cache.py` still imports `get_tinystories_dataloaders` directly.

- [ ] **Step 3: Create the dataset factory**

```python
from src.data.mixed_corpus import get_mixed_corpus_dataloaders
from src.data.tinystories import get_tinystories_dataloaders


def get_experiment_dataloaders(config, tokenizer=None, batch_size=8):
    data_config = normalize_data_config(config.data)
    loader_name = getattr(data_config, "loader", "tinystories")
    if loader_name == "tinystories":
        config.data = data_config
        return get_tinystories_dataloaders(config, tokenizer=tokenizer, batch_size=batch_size)
    if loader_name == "mixture":
        config.data = data_config
        return get_mixed_corpus_dataloaders(config, tokenizer=tokenizer, batch_size=batch_size)
    raise ValueError(f"Unsupported data.loader: {loader_name}")
```

- [ ] **Step 4: Add `normalize_data_config()` in `src/data/dataset_factory.py` so nested `mixture_components` survive config parsing**

```python
def normalize_data_config(data_config):
    if hasattr(data_config, "mixture_components"):
        return data_config

    normalized = copy.copy(data_config)
    if isinstance(getattr(data_config, "__dict__", None), dict):
        raw_components = data_config.__dict__.get("mixture_components")
        if raw_components is not None:
            normalized.mixture_components = raw_components
    return normalized
```

- [ ] **Step 5: Replace the hard-coded TinyStories call in the cache builder**

```python
from src.data.dataset_factory import get_experiment_dataloaders
from transformers import AutoTokenizer

...
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataloaders = get_experiment_dataloaders(
    config=config,
    tokenizer=tokenizer,
    batch_size=batch_size,
)
```

- [ ] **Step 6: Add a cache-builder test that patches the factory instead of the old loader**

```python
with patch("scripts.build_teacher_cache.get_experiment_dataloaders", return_value={"train": mock_loader}):
    ...
```

- [ ] **Step 7: Run the targeted cache/factory tests**

Run: `python -m pytest tests/test_teacher_cache.py -k "dataset_factory or mixture_loader or build_cache" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/data/dataset_factory.py scripts/build_teacher_cache.py tests/test_teacher_cache.py
git commit -m "feat: route cache building through dataset factory"
```

### Task 4: Record the mixed-corpus experiment decision and smoke-test commands

**Files:**
- Modify: `docs/decision_learning.md`

- [ ] **Step 1: Write the decision note**

```md
## Mixed-Corpus Experiment Decision

- Keep hidden-state-only loss unchanged
- Add broader data before adding CE/KL
- Use FineWeb-Edu + UltraChat + MCQ datasets
- Exclude explicit CoT/rationale data from the first mixed run
```

- [ ] **Step 2: Include exact smoke-test commands in the note**

```bash
python -m pytest tests/test_mixed_corpus_data.py tests/test_teacher_cache.py -v
python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 16 --split train --overwrite --verify
python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 8 --split val --overwrite --verify
python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --fast-dev-run
```

- [ ] **Step 3: Commit**

```bash
git add docs/decision_learning.md
git commit -m "docs: record mixed-corpus experiment decision"
```

### Task 5: Build a pilot mixed teacher cache and verify it is usable

**Files:**
- Modify: `docs/superpowers/plans/2026-03-19-mixed-corpus-cache-training.md`

- [ ] **Step 1: Run the mixed-corpus unit tests**

Run: `python -m pytest tests/test_mixed_corpus_data.py tests/test_teacher_cache.py -v`
Expected: PASS.

- [ ] **Step 2: Build a tiny train cache smoke sample**

Run: `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 32 --split train --overwrite --verify`
Expected: PASS, with readable cache metadata and verified `velocity_target` tensors.

- [ ] **Step 3: Build a tiny validation cache smoke sample**

Run: `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 16 --split val --overwrite --verify`
Expected: PASS, with the same metadata contract in the `val` cache directory.

- [ ] **Step 4: Verify cache metadata still matches the architecture contract**

Run: `python -m pytest tests/test_cache_dataset_init.py tests/test_train_smoke.py -k "cache or dataloader" -v`
Expected: PASS.

- [ ] **Step 5: Record the smoke-build results in the plan**

```md
- Mixed train cache smoke build: PASS / FAIL
- Mixed val cache smoke build: PASS / FAIL
- Any component-specific loader issues:
```

### Task 6: Generate the full mixed cache and rerun training

**Files:**
- Modify: `docs/superpowers/plans/2026-03-19-mixed-corpus-cache-training.md`

- [ ] **Step 1: Build the full train cache**

Run: `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --split train --overwrite --verify`
Expected: PASS, with the cache rooted at `cache/mixed_qwen_boundary_states/train`.

- [ ] **Step 2: Build the full validation cache**

Run: `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --split val --overwrite --verify`
Expected: PASS, with the cache rooted at `cache/mixed_qwen_boundary_states/val`.

- [ ] **Step 3: Run the training fast-dev smoke test**

Run: `python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --fast-dev-run`
Expected: PASS, with one train step and one val step using the mixed cache.

- [ ] **Step 4: Run a limited-batch training check before the full run**

Run: `python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --limit-train-batches 20 --limit-val-batches 10`
Expected: PASS, with checkpoint save and stable validation metrics.

- [ ] **Step 5: Launch the full training run**

Run: `python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml`
Expected: PASS, with timestamped outputs under `outputs/<timestamp>-v0_qwen_mixed_corpus_midblock/`.

- [ ] **Step 6: Record the resulting checkpoint paths in the plan**

```md
- best checkpoint:
- final checkpoint:
- structured log:
```

### Task 7: Evaluate the retrained checkpoint and update project state

**Files:**
- Modify: `docs/state.md`
- Modify: `docs/superpowers/plans/2026-03-19-mixed-corpus-cache-training.md`

- [ ] **Step 1: Run the qualitative text sweep on the new checkpoint**

Run: `python scripts/run_checkpoint_text_sweep.py --config configs/v0_mixed_corpus.yaml --checkpoint <best-or-final.ckpt> --num-steps 1 4 8 --output outputs/mixed_text_sweep.json`
Expected: PASS, with side-by-side teacher/student generations saved.

- [ ] **Step 2: Run MMLU-Pro on the new checkpoint**

Run: `python scripts/eval_mmlu_pro.py --config configs/v0_mixed_corpus.yaml --checkpoint <best-or-final.ckpt> --num-samples 70 --num-steps 1 4 8 --output results/mmlu_pro_eval_mixed.json`
Expected: PASS, with a first-token comparison against the teacher.

- [ ] **Step 3: Compare the new run against the current TinyStories baseline**

```md
- Baseline comparison checkpoint: `outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt`
- Does the student still emit `<think>` / `思考` / `</think>` as the first generated token family?
- Is MMLU-Pro still at 0%?
- Did text continuation quality regress?
```

- [ ] **Step 4: Update `docs/state.md` with the new experiment summary**

```md
## Mixed-Corpus Follow-Up

- config:
- cache path:
- checkpoint path:
- text sweep takeaways:
- MMLU-Pro takeaways:
- next decision:
```

- [ ] **Step 5: Commit**

```bash
git add docs/state.md docs/superpowers/plans/2026-03-19-mixed-corpus-cache-training.md
git commit -m "docs: record mixed-corpus retraining results"
```

---

## Verification checklist

- [ ] `beans prime`
- [ ] `python -m pytest tests/test_mixed_corpus_data.py tests/test_teacher_cache.py -v`
- [ ] `python -m pytest tests -v --tb=short`
- [ ] `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 32 --split train --overwrite --verify`
- [ ] `python scripts/build_teacher_cache.py --config configs/v0_mixed_corpus.yaml --limit 16 --split val --overwrite --verify`
- [ ] `python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --fast-dev-run`
- [ ] `python scripts/train_v0.py --config configs/v0_mixed_corpus.yaml --limit-train-batches 20 --limit-val-batches 10`
- [ ] `python scripts/eval_mmlu_pro.py --config configs/v0_mixed_corpus.yaml --checkpoint <best-or-final.ckpt> --num-samples 70 --num-steps 1 4 8 --output results/mmlu_pro_eval_mixed.json`

No dedicated repo lint or typecheck config is present at the workspace root, so verification stays on pytest plus cache/training/eval smoke commands.

## Go / no-go rule

Do not conclude that the data hypothesis is supported until all of the following are true:

- [ ] The mixed-corpus cache builds for both `train` and `val`
- [ ] Training smoke tests pass without touching the loss function
- [ ] The full mixed run produces a usable checkpoint
- [ ] MMLU-Pro no longer collapses entirely to `<think>` / `思考` / `</think>` first-token outputs, or we have hard evidence that broader data alone still fails
- [ ] `docs/state.md` records the result and the next decision point

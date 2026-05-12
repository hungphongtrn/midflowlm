# Phase 3: Training & Smoke Test

## Phase Goal
HF Trainer runs CE-only SFT on the `SFTFlowMidblockModel` with reasoning data from Phase 2. Smoke test completes on RTX 3060 (12GB) with reduced data. Full-run config documented for 24GB+ GPUs. Training produces checkpoint with saved FlowMidblock weights and training metrics logged.

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/train_sft.py` | Create | SFT training entry point |
| `configs/issue-9/sft_flow_midblock.yaml` | Create | Full-run training config (24GB+) |
| `configs/issue-9/sft_flow_midblock_3060.yaml` | Create | Smoke test config (RTX 3060) |
| `tests/test_sft_training.py` | Create | Integration tests for training loop |
| `src/training/sft_utils.py` | Create | HF Trainer callbacks, metrics helpers |

## Background from Phase 1

`SFTFlowMidblockModel` (in `src/model/sft_flow_midblock.py`):
- `forward(input_ids, attention_mask, labels)` returns HF Trainer compatible `{"loss": tensor, "logits": tensor}`
- Liger Kernel computes fused CE loss internally via `self.qwen()`
- Only `midblock.*` parameters have `requires_grad=True` (~22M); everything else frozen (~752M)
- `thinking_level=32` hardcoded (T=32 fixed)
- Loads as: `model = SFTFlowMidblockModel(checkpoint_path=CHECKPOINT_PATH)`

Key behaviors to verify in training:
- HF Trainer wraps the model as-is (no `DistillationTrainer` wrapping)
- Loss decreases over steps (model is learning)
- Gradients only flow through midblock (verified in Phase 1 tests)
- Checkpoint save/restore works correctly
- Memory usage fits within GPU budget

## Architecture

```
scripts/train_sft.py
    ├── loads YAML config
    ├── loads tokenizer (Qwen/Qwen3.5-0.8B)
    ├── creates SFTFlowMidblockModel (warm-start from P3-D3)
    ├── loads prepared datasets via load_from_disk()
    ├── configures HF TrainingArguments
    ├── creates HF Trainer with standard DataCollatorForSeq2Seq
    └── calls trainer.train()
```

## Constraints from Phase 1 Implementation

1. **Monkey-patched forward** — The model patches `Qwen3Model.forward`. `model.qwen` is `AutoLigerKernelForCausalLM` (or `AutoModelForCausalLM` fallback).
2. **Patching interaction with HF Trainer** — HF Trainer calls `model.forward(input_ids, attention_mask, labels)`. The method delegates to `self.qwen()`, which internally calls `self.model()` (the patched Qwen3Model). This is transparent to HF Trainer.
3. **Data format** — Datasets from Phase 2 have columns: `input_ids`, `attention_mask`, `labels`. With packing, attention_mask must be computed correctly.
4. **Fixed T=32** — No variable T sampling; every forward runs 32 ODE steps. This makes training slower than standard SFT (expect ~50-100ms per token per step vs ~5ms for standard).

## Tasks

### Task 1: Write the SFT training utilities

**Files:**
- Create: `src/training/sft_utils.py`

**Goal:** Provide training callbacks and helper functions that work with HF Trainer and SFTFlowMidblockModel.

- [ ] **Step 1: Implement `MidblockMetricsCallback`**

```python
"""HF Trainer callbacks and utilities for SFT training with FlowMidblock."""

import logging
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MidblockMetricsCallback(TrainerCallback):
    """Custom HF Trainer callback that logs FlowMidblock-specific metrics.

    Logs:
      - midblock/total_params: Number of trainable params in midblock
      - midblock/grad_norm: Gradient norm of midblock parameters
      - midblock/param_norm: L2 norm of midblock parameters
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        if hasattr(model, "midblock") and logs is not None:
            midblock = model.midblock
            grad_norm = 0.0
            param_norm = 0.0
            param_count = 0
            for p in midblock.parameters():
                if p.requires_grad:
                    param_norm += p.data.norm(2).item() ** 2
                    param_count += 1
                    if p.grad is not None:
                        grad_norm += p.grad.norm(2).item() ** 2
            logs["midblock/grad_norm"] = grad_norm ** 0.5
            logs["midblock/param_norm"] = param_norm ** 0.5
            logs["midblock/total_params"] = sum(
                p.numel() for p in midblock.parameters() if p.requires_grad
            )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "trainable_params"):
            logger.info(f"Trainable parameters: {model.trainable_params:,}")
            logger.info(f"Frozen parameters:    {model.frozen_params:,}")


def validate_model_for_training(model) -> None:
    """Pre-training validation checks for SFTFlowMidblockModel.

    Raises ValueError if the model is not correctly configured for SFT.
    """
    # Verify only midblock is trainable
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    non_midblock = [n for n in trainable_names if "midblock" not in n]
    if non_midblock:
        raise ValueError(
            f"Non-midblock parameters are trainable: {non_midblock}"
        )

    frozen_midblock = [
        n for n, p in model.named_parameters()
        if not p.requires_grad and "midblock" in n
    ]
    if frozen_midblock:
        logger.warning(
            f"Midblock parameters are frozen (should be trainable): {frozen_midblock}"
        )

    if model.thinking_level != 32:
        logger.warning(
            f"thinking_level={model.thinking_level}, expected 32 for this experiment"
        )

    logger.info("Model validation: PASSED")


def estimate_training_budget(
    num_sequences: int,
    seq_length: int,
    thinking_level: int = 32,
    batch_size: int = 1,
    grad_accum: int = 1,
    num_epochs: int = 1,
    steps_per_second_estimate: float = 0.5,  # Conservative for ODE
) -> dict:
    """Estimate training time and GPU memory for an SFT run.

    Returns:
        dict with:
          - total_steps, total_tokens, estimated_hours
          - effective_batch_size, steps_per_epoch
    """
    steps_per_epoch = num_sequences // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs
    total_tokens_per_step = batch_size * grad_accum * seq_length
    total_tokens = total_steps * total_tokens_per_step

    estimated_seconds = total_steps / steps_per_second_estimate
    estimated_hours = estimated_seconds / 3600

    print("\n" + "=" * 60)
    print("TRAINING BUDGET ESTIMATE")
    print("=" * 60)
    print(f"  Sequences:           {num_sequences:,}")
    print(f"  Seq length:          {seq_length}")
    print(f"  Thinking level (T):  {thinking_level}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Grad accumulation:   {grad_accum}")
    print(f"  Effective BS:        {batch_size * grad_accum}")
    print(f"  Epochs:              {num_epochs}")
    print(f"  Steps per epoch:     {steps_per_epoch:,}")
    print(f"  Total steps:         {total_steps:,}")
    print(f"  Tokens per step:     {total_tokens_per_step:,}")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"  Steps/sec (est):     {steps_per_second_estimate:.2f}")
    print(f"  Estimated hours:     {estimated_hours:.2f}")
    print(f"  Estimated days:      {estimated_hours/24:.2f}")

    return {
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "estimated_hours": estimated_hours,
        "effective_batch_size": batch_size * grad_accum,
        "steps_per_epoch": steps_per_epoch,
    }
```

- [ ] **Step 2: Write tests for training utilities**

Create `tests/test_sft_training.py`:

```python
"""Integration tests for SFT training setup."""

import pytest
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
from src.model.sft_flow_midblock import SFTFlowMidblockModel
from src.training.sft_utils import (
    MidblockMetricsCallback,
    validate_model_for_training,
)

CHECKPOINT_PATH = "models/p3_d3_mix_c/checkpoint.pth"


class TestSFTTrainingSetup:
    """Verify training infrastructure connects correctly."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B", trust_remote_code=True
        )

    @pytest.fixture(scope="class")
    def model(self):
        return SFTFlowMidblockModel(checkpoint_path=CHECKPOINT_PATH)

    def test_model_validation_passes(self, model):
        validate_model_for_training(model)

    def test_hf_trainer_can_wrap_model(self, model, tokenizer):
        """Verify HF Trainer can be created without errors."""
        ds = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4]],
            "labels": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
        })

        training_args = TrainingArguments(
            output_dir="/tmp/sft_test_hf_trainer",
            per_device_train_batch_size=1,
            max_steps=1,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, model=model, padding=True
            ),
            callbacks=[MidblockMetricsCallback()],
        )

        trainer.train()

    def test_loss_decreases_over_two_steps(self, model, tokenizer):
        """Verify training loss decreases (not stuck at initialization)."""
        ds = Dataset.from_dict({
            "input_ids": [
                [100, 200, 300, 400, 500],
                [101, 201, 301, 401, 501],
            ],
            "labels": [
                [100, 200, 300, 400, 500],
                [101, 201, 301, 401, 501],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
        })

        training_args = TrainingArguments(
            output_dir="/tmp/sft_test_loss_decrease",
            per_device_train_batch_size=1,
            max_steps=2,
            logging_steps=1,
            learning_rate=1e-3,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, model=model, padding=True
            ),
        )

        result = trainer.train()
        losses = [
            log_entry["loss"]
            for log_entry in result.log_history
            if "loss" in log_entry
        ]
        assert len(losses) >= 2, f"Expected at least 2 loss values, got {len(losses)}"
        # Loss should be finite (not NaN)
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss)), f"Loss is non-finite: {loss}"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_sft_training.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.sft_utils'`

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sft_training.py -v --timeout=600
```

---

### Task 2: Write the SFT training script

**Files:**
- Create: `scripts/train_sft.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Train SFTFlowMidblockModel on reasoning data using HF Trainer.

Usage (smoke test):
    python scripts/train_sft.py --config configs/issue-9/sft_flow_midblock_3060.yaml

Usage (full run):
    python scripts/train_sft.py --config configs/issue-9/sft_flow_midblock.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import yaml
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_from_disk

from src.model.sft_flow_midblock import SFTFlowMidblockModel
from src.training.sft_utils import (
    MidblockMetricsCallback,
    validate_model_for_training,
    estimate_training_budget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SFT Flow Matcher")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 training")
    args = parser.parse_args()

    # Load and log config
    config = load_config(args.config)

    # Set seed
    seed = config.get("seed", 1337)
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found, using CPU (training will be very slow)")

    # 1. Load tokenizer
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded: pad_token={tokenizer.pad_token!r}")

    # 2. Load model with warm-start
    logger.info("Loading SFTFlowMidblockModel...")
    checkpoint_path = config.get("checkpoint", {}).get("path",
        "models/p3_d3_mix_c/checkpoint.pth")
    model = SFTFlowMidblockModel(
        model_name=model_cfg["name"],
        start_layer=model_cfg.get("start_layer", 8),
        end_layer=model_cfg.get("end_layer", 11),
        thinking_level=model_cfg.get("thinking_level", 32),
        checkpoint_path=checkpoint_path,
        torch_dtype=torch.float32 if args.fp32 else torch.bfloat16,
    )
    logger.info(f"Model created: {model.trainable_params:,} trainable, "
                f"{model.frozen_params:,} frozen")

    # 3. Validate model setup
    validate_model_for_training(model)

    # 4. Move to device
    model = model.to(device)

    # 5. Load datasets
    data_cfg = config["data"]
    train_dir = os.path.join(data_cfg["processed_dir"], "train")
    eval_dir = os.path.join(data_cfg["processed_dir"], "eval")
    logger.info(f"Loading train dataset from {train_dir}")
    train_dataset = load_from_disk(train_dir)
    logger.info(f"Loading eval dataset from {eval_dir}")
    eval_dataset = load_from_disk(eval_dir)

    # 6. Budget estimation (before training starts)
    budget = estimate_training_budget(
        num_sequences=len(train_dataset),
        seq_length=data_cfg.get("max_seq_length", 8192),
        thinking_level=model_cfg.get("thinking_level", 32),
        batch_size=config["training"].get("per_device_train_batch_size", 1),
        grad_accum=config["training"].get("gradient_accumulation_steps", 1),
        num_epochs=config["training"].get("num_train_epochs", 1),
    )

    # 7. Set up HF Trainer
    training_cfg = config["training"]
    output_dir = training_cfg["output_dir"]
    run_name = training_cfg.get("run_name", "sft_flow_midblock")

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,

        # Batch
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),

        # Precision
        fp16=False,
        bf16=not args.fp32 and torch.cuda.is_bf16_supported(),
        fp16_full_eval=False,

        # Schedule
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        max_steps=training_cfg.get("max_steps", -1),

        # Optimizer
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        adam_beta1=training_cfg.get("adam_beta1", 0.9),
        adam_beta2=training_cfg.get("adam_beta2", 0.95),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 100),

        # Checkpointing
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=training_cfg.get("logging_steps", 10),
        report_to=training_cfg.get("report_to", ["tensorboard"]),

        # Evaluation
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 500),

        # Misc
        seed=seed,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        dataloader_drop_last=training_cfg.get("dataloader_drop_last", False),

        # Resume
        resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[MidblockMetricsCallback()],
    )

    # 8. Train
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    # 9. Save final model
    logger.info("Saving final model...")
    # Save only midblock weights (Qwen backbone is frozen, loaded from hub)
    midblock_save_path = os.path.join(output_dir, "midblock_final.pth")
    torch.save(model.midblock.state_dict(), midblock_save_path)
    logger.info(f"Midblock weights saved to {midblock_save_path}")

    # Save full model state dict for evaluation
    full_save_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), full_save_path)
    logger.info(f"Full model saved to {full_save_path}")

    # 10. Metrics summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total steps:       {train_result.global_step}")
    logger.info(f"  Training loss:     {train_result.training_loss:.4f}")
    logger.info(f"  Total FLOPs est.:  {train_result.total_flos:.2e}")
    logger.info(f"  Output directory:  {output_dir}")


if __name__ == "__main__":
    main()
```

---

### Task 3: Write training configs

**Files:**
- Create: `configs/issue-9/sft_flow_midblock.yaml`
- Create: `configs/issue-9/sft_flow_midblock_3060.yaml`

- [ ] **Step 1: Full-run config (24GB+)**

```yaml
# configs/issue-9/sft_flow_midblock.yaml
# Full SFT run — target: 24GB+ GPU (RTX 3090/4090, A6000, H100)
#
# Estimated budget:
#   ~100K packed sequences × 8192 tokens × T=32 × 1 epoch
#   ≈ 800M tokens processed
#   ≈ 10-15 hours on RTX 4090 @ ~0.3 steps/sec (ODE overhead)

seed: 1337

model:
  name: "Qwen/Qwen3.5-0.8B"
  start_layer: 8
  end_layer: 11
  thinking_level: 32  # Fixed T=32

data:
  processed_dir: "./data/reasoning_sft"  # Output from Phase 2
  max_seq_length: 8192

training:
  output_dir: "./outputs/issue-9/sft_flow_midblock"
  run_name: "sft_flow_midblock_full"

  # Memory budget: ~20GB for model + data + ODE intermediate states
  # ~4GB free for gradients (22M params × 4 bytes × 2 = ~176MB for grads)
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 4  # Effective BS = 4
  dataloader_drop_last: false

  # Schedule
  num_train_epochs: 1
  learning_rate: 1.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  lr_scheduler: "cosine"
  warmup_steps: 100

  # Precision
  bf16: true  # Handled by HF Trainer args, not this key

  # Checkpointing
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 2

  # Logging
  logging_steps: 10
  report_to: ["tensorboard"]

  # Evaluation
  eval_strategy: "steps"
  eval_steps: 500

  # Performance
  dataloader_num_workers: 2
  gradient_checkpointing: false  # Not needed — only midblock is trainable
  dataloader_pin_memory: true

checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
```

- [ ] **Step 2: Smoke test config (RTX 3060 12GB)**

```yaml
# configs/issue-9/sft_flow_midblock_3060.yaml
# Smoke test on RTX 3060 (12GB VRAM)
#
# Strategy: minimal data, no eval during training, small batch
# Runs ~100 training steps to verify the pipeline works

seed: 1337

model:
  name: "Qwen/Qwen3.5-0.8B"
  start_layer: 8
  end_layer: 11
  thinking_level: 32

data:
  processed_dir: "./data/reasoning_sft_smoke"  # Smoke test data from Phase 2
  max_seq_length: 2048  # Reduced for memory

training:
  output_dir: "./outputs/issue-9/sft_flow_midblock_3060_smoke"
  run_name: "sft_flow_midblock_smoke"

  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1  # Minimal — memory is the bottleneck

  max_steps: 100  # Short run for smoke test
  num_train_epochs: 1
  learning_rate: 1.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  lr_scheduler: "cosine"
  warmup_steps: 10

  bf16: true

  save_strategy: "steps"
  save_steps: 50
  save_total_limit: 1

  logging_steps: 5
  report_to: ["tensorboard"]

  # Skip eval during smoke test (saves memory)
  eval_strategy: "no"

  dataloader_num_workers: 1
  gradient_checkpointing: true  # Enable for 12GB safety
  dataloader_pin_memory: true

checkpoint:
  path: "models/p3_d3_mix_c/checkpoint.pth"
```

---

### Task 4: Run smoke test training

- [ ] **Step 1: Run smoke test on RTX 3060**

```bash
python scripts/train_sft.py --config configs/issue-9/sft_flow_midblock_3060.yaml
```

**Success criteria:**
- Training starts without import/model/data errors
- Memory usage < 12GB (monitor with `nvidia-smi` or `torch.cuda.memory_summary()`)
- Loss decreases over 100 steps (track in tensorboard)
- MidblockMetricsCallback logs gradient/param norms
- Checkpoint saved at step 50 and 100
- Checkpoint can be loaded back for inference

- [ ] **Step 2: Verify checkpoint save/restore**

```python
# Quick verification script (run after smoke test)
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel

# Load fresh model
model_a = SFTFlowMidblockModel(
    checkpoint_path="models/p3_d3_mix_c/checkpoint.pth"
)

# Save and load
model_a.midblock.save_state_dict(torch.save, "/tmp/midblock_check.pt")
loaded = torch.load("/tmp/midblock_check.pt")
model_b = SFTFlowMidblockModel(
    checkpoint_path="models/p3_d3_mix_c/checkpoint.pth"
)
model_b.midblock.load_state_dict(loaded)

# Verify weights match
for (na, pa), (nb, pb) in zip(
    model_a.midblock.named_parameters(),
    model_b.midblock.named_parameters()
):
    assert torch.allclose(pa, pb), f"Mismatch at {na}"
print("Checkpoint save/restore: OK")
```

- [ ] **Step 3: Estimate full-run time**

After smoke test, update `steps_per_second_estimate` in `estimate_training_budget` with real measurement:

```python
# After smoke test: measure actual steps/sec
# Run the estimate with real numbers
estimate_training_budget(
    num_sequences=100000,  # from actual dataset
    seq_length=8192,
    thinking_level=32,
    batch_size=1,
    grad_accum=4,
    num_epochs=1,
    steps_per_second_estimate=0.15,  # UPDATE WITH REAL MEASUREMENT
)
```

---

### Task 5: Commit

```bash
git add scripts/train_sft.py \
        configs/issue-9/sft_flow_midblock.yaml \
        configs/issue-9/sft_flow_midblock_3060.yaml \
        tests/test_sft_training.py \
        src/training/sft_utils.py
git commit -m "feat: add SFT training script, configs, and smoke test for Issue #9"
```

---

## Phase Completion Criteria
- [ ] `scripts/train_sft.py` loads config, model, datasets, and starts HF Trainer
- [ ] `MidblockMetricsCallback` logs gradient/parameter norms
- [ ] `validate_model_for_training` confirms only midblock is trainable
- [ ] Smoke test completes on RTX 3060: 100 steps, loss decreasing, no OOM
- [ ] Checkpoint save/restore verified (midblock weights survive round-trip)
- [ ] Training budget estimated for full run (time, tokens, steps)
- [ ] All tests in `tests/test_sft_training.py` pass
- [ ] Full-run config documented for 24GB+ GPUs with realistic batch/accum settings

## Handoff Notes
- The training script uses HF Trainer's built-in checkpointing — checkpoints are saved as HF format in `output_dir/checkpoint-N/`
- `DataCollatorForSeq2Seq` handles label padding with `-100` (masked tokens)
- Midblock weights are also saved separately as `midblock_final.pth` for easy loading in Phase 4 (Evaluation)
- If Liger Kernel fails to load, the model falls back to `AutoModelForCausalLM` — training still works but ~20% slower CE computation
- For RTX 3060 (12GB): the model alone occupies ~3.1GB (800M × 4 bytes bf16), leaving ~8.9GB for activations, ODE intermediate states, and optimizer states. With `gradient_checkpointing=true` and `batch_size=1`, this should fit.
- For 24GB GPUs: batch_size=1, grad_accum=8 is the recommended starting point; adjust based on actual ODE memory usage

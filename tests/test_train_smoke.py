"""Smoke tests for raw PyTorch training loop.

These tests verify the core training infrastructure:
1. Deterministic dataloaders from cache
2. One train step
3. One val step
4. Checkpoint save/load
5. Variable T sampling from config
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import numpy as np


def test_deterministic_dataloaders_from_cache():
    """Test that dataloaders from cache are deterministic."""
    from src.training.data import create_cache_dataloader

    # Create mock cache data
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        # Create train subdirectory for split-aware loading
        cache_dir = cache_root / "train"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "model_name": "test-model",
            "model_revision": None,
            "start_layer": 8,
            "end_layer": 11,
            "span_depth": 4,
            "seq_len": 32,
            "store_logits": True,
            "num_samples": 16,
        }
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create mock shards with velocity_target for continuous-time training
        for i in range(2):
            h_start = torch.randn(8, 32, 128)
            h_target = torch.randn(8, 32, 128)
            shard_data = {
                "input_ids": torch.randint(0, 1000, (8, 32)),
                "attention_mask": torch.ones(8, 32, dtype=torch.int64),
                "h_start": h_start,
                "h_target": h_target,
                "velocity_target": h_target - h_start,  # v_target = h_end - h_start
                "teacher_logits": torch.randn(8, 32, 1000),
            }
            torch.save(shard_data, cache_dir / f"shard_{i:04d}_of_0002.pt")

        # Create two dataloaders with same seed from cache_root
        loader1 = create_cache_dataloader(
            cache_dir=str(cache_root),
            batch_size=4,
            shuffle=True,
            seed=42,
            split="train",
        )
        loader2 = create_cache_dataloader(
            cache_dir=str(cache_root),
            batch_size=4,
            shuffle=True,
            seed=42,
            split="train",
        )

        # Get batches and compare
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        assert torch.equal(batch1["input_ids"], batch2["input_ids"])
        assert torch.equal(batch1["attention_mask"], batch2["attention_mask"])
        assert torch.equal(batch1["h_start"], batch2["h_start"])
        assert torch.equal(batch1["h_target"], batch2["h_target"])


def test_one_train_step():
    """Test that one training step runs without errors."""
    from src.training.trainer import Trainer

    # Create mock components with proper gradient handling
    mock_model = MagicMock()
    # Create outputs that support backprop
    endpoint_hidden = torch.randn(2, 16, 128, requires_grad=True)
    trajectory_hidden = torch.randn(2, 16, 4, 128, requires_grad=True)
    logits = torch.randn(2, 16, 1000, requires_grad=True)
    mock_model.return_value = {
        "endpoint_hidden": endpoint_hidden,
        "trajectory_hidden": trajectory_hidden,
        "logits": logits,
    }
    # Create a parameter that requires grad
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    mock_loss_fn = MagicMock()

    # Loss needs to be computed from outputs for proper backprop
    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        # Use a simple loss that requires grad
        loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": 0.0,
            "endpoint_loss": loss_val.item() * 0.5,
            "trajectory_loss": loss_val.item() * 0.5,
        }
        return loss_val, metrics

    mock_loss_fn.side_effect = loss_side_effect

    # Create mock batch with velocity_target for continuous-time training
    h_start = torch.randn(2, 16, 128)
    h_target = torch.randn(2, 16, 128)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.int64),
        "h_start": h_start,
        "h_target": h_target,
        "velocity_target": h_target - h_start,  # v_target = h_end - h_start
        "teacher_logits": torch.randn(2, 16, 1000),
    }

    # Create trainer with minimal config
    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,  # Disable continuous time for legacy test
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    # Run one train step
    metrics = trainer.train_step(batch, T=4)

    assert "loss" in metrics
    assert metrics["loss"] != 0  # Loss can be negative, just check it's not zero
    mock_loss_fn.assert_called_once()


def test_one_val_step():
    """Test that one validation step runs without errors."""
    from src.training.trainer import Trainer

    # Create mock components with proper parameters
    mock_model = MagicMock()
    mock_model.return_value = {
        "endpoint_hidden": torch.randn(2, 16, 128),
        "trajectory_hidden": torch.randn(2, 16, 4, 128),
        "logits": torch.randn(2, 16, 1000),
    }
    # Create a parameter that requires grad (needed for optimizer creation)
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    mock_loss_fn = MagicMock()
    mock_loss_fn.return_value = (
        torch.tensor(1.0),
        {
            "total_loss": 1.0,
            "endpoint_loss": 0.5,
            "trajectory_loss": 0.5,
        },
    )

    # Create mock batch with velocity_target for continuous-time training
    h_start = torch.randn(2, 16, 128)
    h_target = torch.randn(2, 16, 128)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.int64),
        "h_start": h_start,
        "h_target": h_target,
        "velocity_target": h_target - h_start,  # v_target = h_end - h_start
        "teacher_logits": torch.randn(2, 16, 1000),
    }

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    # Run one val step
    metrics = trainer.val_step(batch, T=4)

    assert "loss" in metrics
    assert metrics["loss"] > 0
    mock_loss_fn.assert_called_once()
    # Ensure model was in eval mode (no gradients)


def test_checkpoint_save_load():
    """Test that checkpoints can be saved and loaded."""
    from src.training.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create mock components
        mock_model = MagicMock()
        mock_model.parameters = Mock(return_value=[torch.randn(10, requires_grad=True)])
        mock_model.state_dict = Mock(return_value={"param": torch.randn(10)})

        mock_loss_fn = MagicMock()

        config = {
            "optimizer": {
                "name": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
            "train_loop": {
                "precision": "fp32",
                "accumulate_grad_batches": 1,
                "checkpoint_dir": str(checkpoint_dir),
            },
            "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        }

        trainer = Trainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        # Set some state
        trainer.global_step = 100
        trainer.current_epoch = 2

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(checkpoint_dir / "test.ckpt")

        # Verify checkpoint exists
        assert checkpoint_path.exists()

        # Load checkpoint
        mock_model.load_state_dict = Mock()
        trainer.load_checkpoint(checkpoint_path)

        # Verify state restored
        assert trainer.global_step == 100
        assert trainer.current_epoch == 2
        mock_model.load_state_dict.assert_called_once()


def test_variable_t_sampling_from_config():
    """Test that T values are sampled correctly from config distribution."""
    from src.training.trainer import Trainer

    # Create mock model with proper parameters
    mock_model = MagicMock()
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
        },
        "model": {
            "train_T_values": [1, 2, 4, 8],
            "train_T_weights": [0.25, 0.25, 0.25, 0.25],
        },
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    # Sample many T values and check distribution
    t_values = [trainer.sample_T() for _ in range(1000)]

    # Check that all expected values appear
    assert all(t in [1, 2, 4, 8] for t in t_values)

    # Check approximate distribution (with some tolerance)
    from collections import Counter

    counts = Counter(t_values)

    for t in [1, 2, 4, 8]:
        assert counts[t] > 100, f"T={t} should appear frequently, got {counts[t]}"


def test_fixed_t_training():
    """Test training with fixed T=depth."""
    from src.training.trainer import Trainer

    # Create mock model with proper parameters
    mock_model = MagicMock()
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
        },
        # Fixed T (single value, weight 1.0)
        "model": {
            "train_T_values": [4],
            "train_T_weights": [1.0],
        },
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    # All samples should be T=4
    for _ in range(100):
        assert trainer.sample_T() == 4


def test_amp_fp16_available():
    """Test that AMP fp16 is available when configured."""
    from src.training.trainer import Trainer

    mock_model = MagicMock()
    mock_model.parameters = Mock(return_value=[torch.randn(10, requires_grad=True)])
    mock_loss_fn = MagicMock()

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp16-mixed",
            "accumulate_grad_batches": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    # Should not raise error
    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    assert trainer.use_amp is True
    assert trainer.precision == "fp16"


def test_amp_bf16_available():
    """Test that AMP bf16 is available when configured."""
    from src.training.trainer import Trainer

    mock_model = MagicMock()
    mock_model.parameters = Mock(return_value=[torch.randn(10, requires_grad=True)])
    mock_loss_fn = MagicMock()

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "bf16-mixed",
            "accumulate_grad_batches": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    # Should not raise error
    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    assert trainer.use_amp is True
    assert trainer.precision == "bf16"


def test_trainer_samples_continuous_t_between_zero_and_one():
    """Test that trainer samples continuous t values between 0 and 1."""
    from src.training.trainer import Trainer

    # Create mock model with proper parameters
    mock_model = MagicMock()
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    # Sample continuous t values
    t = trainer.sample_continuous_time(batch_size=2, device=torch.device("cpu"))

    assert torch.all(t >= 0.0)
    assert torch.all(t <= 1.0)
    assert t.shape == (2,)


def test_gradient_accumulation():
    """Test that gradient accumulation works correctly."""
    from src.training.trainer import Trainer

    # Create mock model with proper parameters for optimizer
    mock_model = MagicMock()
    mock_param = nn.Parameter(torch.randn(10))
    mock_model.parameters = Mock(return_value=[mock_param])

    # Create outputs that support backprop
    def model_forward(*args, **kwargs):
        endpoint_hidden = torch.randn(2, 16, 128, requires_grad=True)
        return {
            "endpoint_hidden": endpoint_hidden,
            "trajectory_hidden": torch.randn(2, 16, 4128),
            "logits": torch.randn(2, 16, 1000),
        }

    mock_model.side_effect = model_forward

    # Loss function that computes from outputs
    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": 0.0,
            "endpoint_loss": loss_val.item() * 0.5,
            "trajectory_loss": loss_val.item() * 0.5,
        }
        return loss_val, metrics

    mock_loss_fn = MagicMock()
    mock_loss_fn.side_effect = loss_side_effect

    # Create mock batch with velocity_target for continuous-time training
    h_start = torch.randn(2, 16, 128)
    h_target = torch.randn(2, 16, 128)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.int64),
        "h_start": h_start,
        "h_target": h_target,
        "velocity_target": h_target - h_start,  # v_target = h_end - h_start
        "teacher_logits": torch.randn(2, 16, 1000),
    }

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,  # Disable continuous time for legacy test
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
    }

    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    # Run multiple steps (less than accumulation steps)
    for i in range(3):
        metrics = trainer.train_step(batch, T=4)
        # Gradient should not be zeroed yet
        assert trainer.accumulation_step == i + 1

    # Fourth step should trigger optimizer step
    metrics = trainer.train_step(batch, T=4)
    assert trainer.accumulation_step == 0  # Reset after optimizer step

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


def test_format_train_batch_log_includes_kl_loss():
    """Test that batch log formatting includes KL-related metrics."""
    from scripts.train_v0 import format_train_batch_log

    message = format_train_batch_log(
        batch_idx=12,
        metrics={
            "loss": 0.1234,
            "velocity_loss": 0.0123,
            "kl_loss": 0.4567,
            "ce_loss": 0.0,
            "T": 4,
        },
    )

    assert message == (
        "  Batch 12: loss=0.1234, velocity_loss=0.0123, "
        "kl_loss=0.4567, ce_loss=0.0000, T=4"
    )


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


def test_train_step_computes_online_teacher_logits_for_kl():
    """Test that KL training can source teacher logits from a live teacher model."""
    from src.training.trainer import Trainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = self.weight * torch.ones(batch_size, seq_len, 4)
            endpoint_hidden = self.weight * torch.ones(batch_size, seq_len, 2)
            trajectory_hidden = self.weight * torch.ones(batch_size, seq_len, 1, 2)
            return {
                "endpoint_hidden": endpoint_hidden,
                "trajectory_hidden": trajectory_hidden,
                "logits": logits,
            }

    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            self.forward_calls += 1
            batch_size, seq_len = input_ids.shape
            logits = torch.full((batch_size, seq_len, 4), 7.0)
            return {"logits": logits}

    student = DummyStudent()
    teacher = DummyTeacher()
    captured = {}

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        captured["teacher_logits"] = teacher_batch["teacher_logits"].detach().clone()
        captured["teacher_logits_device"] = teacher_batch["teacher_logits"].device.type
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": 0.0,
            "endpoint_loss": loss_val.item(),
            "trajectory_loss": 0.0,
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.int64),
        "h_start": torch.randn(2, 5, 2),
        "velocity_target": torch.randn(2, 5, 2),
        "labels": torch.randint(0, 4, (2, 5)),
    }

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": None},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [1], "train_T_weights": [1.0]},
        "loss": {
            "kl_weight": 0.25,
            "ce_weight": 0.0,
            "teacher_logits_source": "online",
        },
    }

    trainer = Trainer(
        model=student,
        loss_fn=Mock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
        teacher_model=teacher,
    )

    trainer.train_step(batch, T=1)

    assert teacher.forward_calls == 1
    assert torch.allclose(captured["teacher_logits"], torch.full((2, 5, 4), 7.0))
    assert captured["teacher_logits_device"] == "cpu"
    assert trainer._cached_teacher_logits_cpu is None


def test_train_step_prefers_live_teacher_logits_over_cached_batch_logits():
    """Test that live teacher logits replace cached logits when KL is enabled."""
    from src.training.trainer import Trainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = self.weight * torch.ones(batch_size, seq_len, 4)
            endpoint_hidden = self.weight * torch.ones(batch_size, seq_len, 2)
            trajectory_hidden = self.weight * torch.ones(batch_size, seq_len, 1, 2)
            return {
                "endpoint_hidden": endpoint_hidden,
                "trajectory_hidden": trajectory_hidden,
                "logits": logits,
            }

    class DummyTeacher(nn.Module):
        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = torch.full((batch_size, seq_len, 4), 9.0)
            return {"logits": logits}

    student = DummyStudent()
    teacher = DummyTeacher()
    captured = {}

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        captured["teacher_logits"] = teacher_batch["teacher_logits"].detach().clone()
        loss_val = student_outputs["endpoint_hidden"].mean()
        return loss_val, {"total_loss": loss_val.item()}

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.int64),
        "h_start": torch.randn(2, 4, 2),
        "velocity_target": torch.randn(2, 4, 2),
        "labels": torch.randint(0, 4, (2, 4)),
        "teacher_logits": torch.full((2, 4, 4), -3.0),
    }

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": None},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [1], "train_T_weights": [1.0]},
        "loss": {
            "kl_weight": 0.5,
            "ce_weight": 0.0,
            "teacher_logits_source": "online",
        },
    }

    trainer = Trainer(
        model=student,
        loss_fn=Mock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
        teacher_model=teacher,
    )

    trainer.train_step(batch, T=1)

    assert torch.allclose(captured["teacher_logits"], torch.full((2, 4, 4), 9.0))


def test_train_step_skips_live_teacher_when_source_is_cache():
    """Test that teacher forward is skipped unless source is explicitly online."""
    from src.training.trainer import Trainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = self.weight * torch.ones(batch_size, seq_len, 4)
            endpoint_hidden = self.weight * torch.ones(batch_size, seq_len, 2)
            trajectory_hidden = self.weight * torch.ones(batch_size, seq_len, 1, 2)
            return {
                "endpoint_hidden": endpoint_hidden,
                "trajectory_hidden": trajectory_hidden,
                "logits": logits,
            }

    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            self.forward_calls += 1
            batch_size, seq_len = input_ids.shape
            return {"logits": torch.full((batch_size, seq_len, 4), 11.0)}

    student = DummyStudent()
    teacher = DummyTeacher()
    captured = {}

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        captured["teacher_logits"] = teacher_batch["teacher_logits"].detach().clone()
        loss_val = student_outputs["endpoint_hidden"].mean()
        return loss_val, {"total_loss": loss_val.item()}

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.int64),
        "h_start": torch.randn(2, 4, 2),
        "velocity_target": torch.randn(2, 4, 2),
        "labels": torch.randint(0, 4, (2, 4)),
        "teacher_logits": torch.full((2, 4, 4), -5.0),
    }

    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": None},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [1], "train_T_weights": [1.0]},
        "loss": {
            "kl_weight": 0.5,
            "ce_weight": 0.0,
            "teacher_logits_source": "cache",
        },
    }

    trainer = Trainer(
        model=student,
        loss_fn=Mock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
        teacher_model=teacher,
    )

    trainer.train_step(batch, T=1)

    assert teacher.forward_calls == 0
    assert torch.allclose(captured["teacher_logits"], torch.full((2, 4, 4), -5.0))


def test_structured_logger_records_teacher_logits_source_metadata():
    """Test that model metadata explicitly records teacher logits mode."""
    from scripts.train_v0 import StructuredTrainingLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "train.jsonl"
        logger = StructuredTrainingLogger(log_path)

        logger.log_model_info(
            model_summary={
                "name": "Qwen/Qwen3.5-0.8B",
                "teacher_logits_source": "online",
                "uses_online_teacher_logits": True,
            },
            param_summary={"total": 10},
            trainable_params=2,
            frozen_params=8,
        )

        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "model_info"
        assert event["data"]["model_summary"]["teacher_logits_source"] == "online"
        assert event["data"]["model_summary"]["uses_online_teacher_logits"] is True


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


def test_train_script_dataloader_validates_cache_compatibility():
    """Test that train_v0.py create_dataloaders validates cache compatibility before building loaders.

    This test proves that the train-script path (scripts/train_v0.py::create_dataloaders)
    properly validates cache compatibility BEFORE creating dataloaders. Without this validation,
    mismatched seq_len or missing teacher_logits would only be caught at train step time with
    confusing errors like 'teacher_logits not in teacher_batch but kl_weight > 0.0'.
    """
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import src.training.data as data_module

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_v0
    from train_v0 import create_dataloaders

    mock_metadata = MagicMock()
    mock_metadata.model_name = "Qwen/Qwen3.5-0.8B"
    mock_metadata.model_revision = None
    mock_metadata.start_layer = 8
    mock_metadata.end_layer = 11
    mock_metadata.span_depth = 4
    mock_metadata.seq_len = 256
    mock_metadata.store_logits = True

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": None},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "data": {
            "seq_len": 512,
            "batch_size": 2,
            "num_workers": 0,
        },
        "loss": {"kl_weight": 0.25},
    }

    with patch.object(data_module, "_load_metadata", return_value=mock_metadata):
        with pytest.raises(ValueError, match="seq_len mismatch"):
            create_dataloaders(config, cache_dir="./fake_cache_dir")


def test_train_script_dataloader_catches_missing_logits_when_kl_weight_positive():
    """Test that train_v0.py create_dataloaders catches missing teacher_logits when kl_weight > 0.

    When kl_weight > 0, the cache MUST have store_logits=True. If the cache was built without
    logits but config has kl_weight > 0, validation should fail at dataloader creation time
    rather than failing at training time with 'teacher_logits not in teacher_batch'.
    """
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import src.training.data as data_module

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_v0
    from train_v0 import create_dataloaders

    mock_metadata = MagicMock()
    mock_metadata.model_name = "Qwen/Qwen3.5-0.8B"
    mock_metadata.model_revision = None
    mock_metadata.start_layer = 8
    mock_metadata.end_layer = 11
    mock_metadata.span_depth = 4
    mock_metadata.seq_len = 512
    mock_metadata.store_logits = False

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": None},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "data": {"seq_len": 512, "batch_size": 2, "num_workers": 0},
        "loss": {"kl_weight": 0.25},
    }

    with patch.object(data_module, "_load_metadata", return_value=mock_metadata):
        with pytest.raises(ValueError, match="store_logits is False but kl_weight"):
            create_dataloaders(config, cache_dir="./fake_cache_dir")


def test_teacher_state_offline_cache_routes_to_cache_dataloader():
    """Test that offline_cache mode uses create_cache_dataloader path.

    When teacher_state.mode is 'offline_cache', the router should call
    create_cache_dataloader (not get_experiment_dataloaders).
    """
    from unittest.mock import MagicMock, patch, Mock
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from train_v0 import create_dataloaders

    offline_config = {
        "teacher_state": {"mode": "offline_cache"},
        "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": None},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "data": {
            "seq_len": 256,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "loss": {"kl_weight": 0.25},
    }

    with patch("train_v0.create_cache_dataloader") as mock_cache_loader:
        mock_cache_loader.return_value = (MagicMock(), MagicMock())
        with patch("train_v0.validate_cache_compatibility"):
            create_dataloaders(offline_config, cache_dir="./fake_cache")

        assert mock_cache_loader.called, (
            "offline_cache mode should call create_cache_dataloader"
        )


def test_teacher_state_online_no_cache_routes_to_token_dataset():
    """Test that online_no_cache mode uses get_experiment_dataloaders path.

    When teacher_state.mode is 'online_no_cache', the router should call
    get_experiment_dataloaders (not create_cache_dataloader).
    Currently this raises NotImplementedError - this test should FAIL until routing is implemented.
    """
    from unittest.mock import MagicMock, patch, Mock
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_v0
    from src.data.dataset_factory import get_experiment_dataloaders

    online_config = {
        "teacher_state": {"mode": "online_no_cache"},
        "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": None},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "data": {
            "loader": "mixture",
            "seq_len": 128,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "mixture_components": [
                {
                    "name": "fineweb_edu",
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "dataset_config": "sample-10BT",
                    "train_split": "train",
                    "val_split": "train",
                    "format_type": "plain_text",
                    "text_field": "text",
                    "train_samples": 100,
                    "val_samples": 20,
                }
            ],
        },
        "loss": {"kl_weight": 0.25},
    }

    with patch.object(
        train_v0, "get_experiment_dataloaders", wraps=get_experiment_dataloaders
    ) as mock_get_exp_dl:
        mock_get_exp_dl.return_value = {
            "train": MagicMock(),
            "val": MagicMock(),
        }
        train_loader, val_loader = train_v0.create_dataloaders(
            online_config, cache_dir="./fake_cache"
        )

        assert mock_get_exp_dl.called, (
            "online_no_cache mode should call get_experiment_dataloaders (token dataset path), "
            "not an offline cache loader"
        )
        assert train_loader is mock_get_exp_dl.return_value["train"]
        assert val_loader is mock_get_exp_dl.return_value["val"]


def test_teacher_state_online_write_through_routes_to_token_dataset():
    """Test that online_write_through_cache mode uses get_experiment_dataloaders path.

    When teacher_state.mode is 'online_write_through_cache', the router should call
    get_experiment_dataloaders (not create_cache_dataloader).
    Currently this raises NotImplementedError - this test should FAIL until routing is implemented.
    """
    from unittest.mock import MagicMock, patch, Mock
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_v0
    from src.data.dataset_factory import get_experiment_dataloaders

    write_through_config = {
        "teacher_state": {"mode": "online_write_through_cache"},
        "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": None},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "teacher_cache": {"enabled": True},
        "data": {
            "loader": "mixture",
            "seq_len": 128,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "mixture_components": [
                {
                    "name": "fineweb_edu",
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "dataset_config": "sample-10BT",
                    "train_split": "train",
                    "val_split": "train",
                    "format_type": "plain_text",
                    "text_field": "text",
                    "train_samples": 100,
                    "val_samples": 20,
                }
            ],
        },
        "loss": {"kl_weight": 0.25},
    }

    with patch.object(
        train_v0, "get_experiment_dataloaders", wraps=get_experiment_dataloaders
    ) as mock_get_exp_dl:
        mock_get_exp_dl.return_value = {
            "train": MagicMock(),
            "val": MagicMock(),
        }
        train_loader, val_loader = train_v0.create_dataloaders(
            write_through_config, cache_dir="./fake_cache"
        )

        assert mock_get_exp_dl.called, (
            "online_write_through_cache mode should call get_experiment_dataloaders (token dataset path), "
            "not an offline cache loader"
        )
        assert train_loader is mock_get_exp_dl.return_value["train"]
        assert val_loader is mock_get_exp_dl.return_value["val"]


class TestTrainerTeacherStateModes:
    """Tests for trainer behavior with different teacher_state modes.

    Task 6: Teach the trainer to compute or consume teacher states per mode:
    - offline_cache: Consumes cached h_start, velocity_target, optional teacher_logits from batch
    - online_no_cache: Performs live teacher extraction and populates loss batch
    - online_write_through_cache: Performs live extraction and calls cache writer hook
    """

    def test_trainer_offline_mode_consumes_cached_teacher_states(self):
        """Trainer in offline_cache mode should consume cached teacher states from batch.

        When using offline_cache mode, the batch already contains h_start,
        velocity_target, and optional teacher_logits from the pre-built cache.
        The trainer should pass these directly to the loss function.
        """
        from src.training.trainer import Trainer
        import torch.nn as nn
        from unittest.mock import MagicMock, Mock

        mock_model = MagicMock()
        endpoint_hidden = torch.randn(2, 16, 128, requires_grad=True)
        mock_model.return_value = {
            "endpoint_hidden": endpoint_hidden,
            "trajectory_hidden": torch.randn(2, 16, 4, 128),
            "logits": torch.randn(2, 16, 1000),
        }
        mock_param = nn.Parameter(torch.randn(10))
        mock_model.parameters = Mock(return_value=[mock_param])

        mock_loss_fn = MagicMock()

        def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
            h_start = teacher_batch["h_start"]
            velocity_target = teacher_batch["velocity_target"]
            assert h_start is not None, "offline mode batch should contain h_start"
            assert velocity_target is not None, (
                "offline mode batch should contain velocity_target"
            )
            loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
            return loss_val, {"total_loss": loss_val.item()}

        mock_loss_fn.side_effect = loss_side_effect

        config = {
            "teacher_state": {"mode": "offline_cache"},
            "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 0.01},
            "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
            "train_loop": {"precision": "fp32", "accumulate_grad_batches": 1},
            "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        }

        trainer = Trainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        h_start = torch.randn(2, 16, 128)
        h_target = torch.randn(2, 16, 128)
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.int64),
            "h_start": h_start,
            "velocity_target": h_target - h_start,
            "teacher_logits": torch.randn(2, 16, 1000),
        }

        metrics = trainer.train_step(batch, T=4)
        assert "loss" in metrics
        mock_loss_fn.assert_called_once()

    def test_trainer_online_no_cache_mode_extracts_live_teacher_states(self):
        """Trainer in online_no_cache mode should perform live teacher extraction.

        When using online_no_cache mode, the batch only contains input_ids
        and attention_mask. The trainer should extract teacher states using
        QwenInspector and populate h_start, velocity_target for the loss function.
        """
        from src.training.trainer import Trainer
        import torch.nn as nn
        from unittest.mock import MagicMock, Mock, patch

        mock_model = MagicMock()
        endpoint_hidden = torch.randn(2, 16, 128, requires_grad=True)
        mock_model.return_value = {
            "endpoint_hidden": endpoint_hidden,
            "trajectory_hidden": torch.randn(2, 16, 4, 128),
            "logits": torch.randn(2, 16, 1000),
        }
        mock_param = nn.Parameter(torch.randn(10))
        mock_model.parameters = Mock(return_value=[mock_param])

        captured_batch = {}
        mock_loss_fn = MagicMock()

        def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
            captured_batch["teacher_batch"] = teacher_batch
            loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
            return loss_val, {"total_loss": loss_val.item()}

        mock_loss_fn.side_effect = loss_side_effect

        config = {
            "teacher_state": {"mode": "online_no_cache"},
            "model": {
                "name": "Qwen/Qwen3.5-0.8B",
                "train_T_values": [4],
                "train_T_weights": [1.0],
            },
            "replacement_model": {"start_layer": 8, "end_layer": 11},
            "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 0.01},
            "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
            "train_loop": {"precision": "fp32", "accumulate_grad_batches": 1},
        }

        trainer = Trainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        mock_inspector = MagicMock()
        mock_inspector.extract_all.return_value = {
            "h_start": torch.randn(2, 16, 128),
            "h_target": torch.randn(2, 16, 128),
            "logits": torch.randn(2, 16, 1000),
        }

        batch_no_teacher = {
            "input_ids": torch.randint(0, 1000, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.int64),
        }

        with patch("src.training.trainer.QwenInspector", return_value=mock_inspector):
            metrics = trainer.train_step(batch_no_teacher, T=4)

        assert "loss" in metrics
        assert "h_start" in captured_batch["teacher_batch"], (
            "online_no_cache mode should populate h_start via live extraction"
        )
        assert "velocity_target" in captured_batch["teacher_batch"], (
            "online_no_cache mode should populate velocity_target via live extraction"
        )

    def test_trainer_online_write_through_mode_extracts_and_writes_cache(self):
        """Trainer in write_through_cache mode should extract and write to cache.

        When using online_write_through_cache mode, the trainer should:
        1. Perform live teacher extraction (like online_no_cache)
        2. Call the cache writer hook to persist teacher states
        """
        from src.training.trainer import Trainer
        import torch.nn as nn
        from unittest.mock import MagicMock, Mock, patch

        mock_model = MagicMock()
        endpoint_hidden = torch.randn(2, 16, 128, requires_grad=True)
        mock_model.return_value = {
            "endpoint_hidden": endpoint_hidden,
            "trajectory_hidden": torch.randn(2, 16, 4, 128),
            "logits": torch.randn(2, 16, 1000),
        }
        mock_param = nn.Parameter(torch.randn(10))
        mock_model.parameters = Mock(return_value=[mock_param])

        captured_batch = {}
        mock_loss_fn = MagicMock()

        def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
            captured_batch["teacher_batch"] = teacher_batch
            loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
            return loss_val, {"total_loss": loss_val.item()}

        mock_loss_fn.side_effect = loss_side_effect

        mock_cache_writer = MagicMock()

        config = {
            "teacher_state": {"mode": "online_write_through_cache"},
            "model": {
                "name": "Qwen/Qwen3.5-0.8B",
                "train_T_values": [4],
                "train_T_weights": [1.0],
            },
            "replacement_model": {"start_layer": 8, "end_layer": 11},
            "teacher_cache": {"enabled": True, "cache_dir": "./cache/write"},
            "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 0.01},
            "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
            "train_loop": {"precision": "fp32", "accumulate_grad_batches": 1},
        }

        trainer = Trainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )
        trainer._cache_writer = mock_cache_writer

        mock_inspector = MagicMock()
        mock_inspector.extract_all.return_value = {
            "h_start": torch.randn(2, 16, 128),
            "h_target": torch.randn(2, 16, 128),
            "logits": torch.randn(2, 16, 1000),
        }

        batch_no_teacher = {
            "input_ids": torch.randint(0, 1000, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.int64),
        }

        with patch("src.training.trainer.QwenInspector", return_value=mock_inspector):
            metrics = trainer.train_step(batch_no_teacher, T=4)

        assert "loss" in metrics
        assert "h_start" in captured_batch["teacher_batch"], (
            "write_through mode should populate h_start via live extraction"
        )
        (
            mock_cache_writer.write_shard.assert_called_once(),
            ("write_through mode should call cache_writer.write_shard"),
        )

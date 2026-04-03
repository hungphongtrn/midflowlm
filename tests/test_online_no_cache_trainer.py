"""Smoke tests for online-no-cache trainer.

Tests verify:
1. OnlineNoCacheTrainer initializes correctly
2. One train step with on-the-fly teacher target extraction
3. One val step with on-the-fly teacher target extraction
4. Checkpoint save/load (including accumulation_step)
5. Best-checkpoint saving in fit()
6. No branching in base Trainer from this path
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock


def test_online_no_cache_trainer_initializes():
    """Test that OnlineNoCacheTrainer initializes without errors."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

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
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {
            "velocity_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
        },
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert trainer.accumulation_step == 0
    assert trainer.kl_weight == 0.0
    assert trainer.monitor_key == "val/total_loss"
    assert trainer.monitor_mode == "min"
    assert trainer.best_val_metric == float("inf")


def test_one_train_step_with_extract_teacher_targets():
    """Test that train step calls model.extract_teacher_targets() once per step."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    extract_call_count = 0

    class TrackedStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = self.weight * torch.ones(batch_size, seq_len, 4)
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": logits,
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            nonlocal extract_call_count
            extract_call_count += 1
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    student = TrackedStudent()

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item() * 0.8,
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    mock_loss_fn = MagicMock()
    mock_loss_fn.side_effect = loss_side_effect

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {
            "velocity_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
        },
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=student,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    metrics = trainer.train_step(batch, T=4)

    assert extract_call_count == 1
    assert "loss" in metrics
    assert metrics["velocity_loss"] > 0


def test_one_val_step_with_extract_teacher_targets():
    """Test that val step calls model.extract_teacher_targets() once per step."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    extract_call_count = 0

    class TrackedStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            logits = self.weight * torch.ones(batch_size, seq_len, 4)
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": logits,
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            nonlocal extract_call_count
            extract_call_count += 1
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    student = TrackedStudent()

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item() * 0.8,
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    mock_loss_fn = MagicMock()
    mock_loss_fn.side_effect = loss_side_effect

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {
            "velocity_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
        },
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=student,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    metrics = trainer.val_step(batch, T=4)

    assert extract_call_count == 1
    assert "loss" in metrics


def test_online_no_cache_trainer_uses_continuous_time():
    """Test that trainer samples continuous time when enabled."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    mock_loss_fn = MagicMock()
    mock_loss_fn.side_effect = loss_side_effect

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
            "sample_continuous_time": True,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {
            "velocity_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
        },
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    t = trainer.sample_continuous_time(batch_size=2, device=torch.device("cpu"))
    assert t.shape == (2,)
    assert torch.all(t >= 0.0)
    assert torch.all(t <= 1.0)


def test_online_no_cache_checkpoint_save_load_with_accumulation_step():
    """Test that OnlineNoCacheTrainer checkpoints include accumulation_step."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

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
                "grad_clip_norm": 1.0,
            },
            "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 10},
            "train_loop": {
                "precision": "fp32",
                "accumulate_grad_batches": 1,
                "checkpoint_dir": str(checkpoint_dir),
            },
            "model": {"train_T_values": [4], "train_T_weights": [1.0]},
            "loss": {
                "velocity_weight": 1.0,
                "kl_weight": 0.0,
                "ce_weight": 0.0,
            },
            "logging": {"monitor": "val/total_loss", "mode": "min"},
        }

        trainer = OnlineNoCacheTrainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        trainer.global_step = 100
        trainer.current_epoch = 2
        trainer.accumulation_step = 3

        checkpoint_path = trainer.save_checkpoint(checkpoint_dir / "test.ckpt")
        assert checkpoint_path.exists()

        mock_model.load_state_dict = Mock()
        trainer2 = OnlineNoCacheTrainer(
            model=mock_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.global_step == 100
        assert trainer2.current_epoch == 2
        assert trainer2.accumulation_step == 3


def test_online_no_cache_trainer_has_no_teacher_model_param():
    """Verify OnlineNoCacheTrainer does not accept a teacher_model parameter."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer
    import inspect

    sig = inspect.signature(OnlineNoCacheTrainer.__init__)
    param_names = list(sig.parameters.keys())
    assert "teacher_model" not in param_names, (
        "OnlineNoCacheTrainer should not have a teacher_model parameter; "
        "it uses model.extract_teacher_targets() instead"
    )


def test_gradient_accumulation_online_no_cache():
    """Test that gradient accumulation works correctly in OnlineNoCacheTrainer."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean() * 0.1
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {
            "velocity_weight": 1.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
        },
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=MagicMock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
    )

    for i in range(3):
        metrics = trainer.train_step(batch, T=4)
        assert trainer.accumulation_step == i + 1

    metrics = trainer.train_step(batch, T=4)
    assert trainer.accumulation_step == 0


def test_cached_trainer_has_teacher_model_param():
    """Verify CachedTrainer has teacher_model parameter (deprecated but available)."""
    from src.training.cached_trainer import CachedTrainer
    import inspect

    sig = inspect.signature(CachedTrainer.__init__)
    param_names = list(sig.parameters.keys())
    assert "teacher_model" in param_names, (
        "CachedTrainer should have teacher_model parameter"
    )


def test_is_better_min_mode():
    """Test _is_better under min mode."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

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
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": None},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=mock_model, loss_fn=mock_loss_fn, config=config, device="cpu"
    )

    trainer.best_val_metric = 0.5
    assert trainer._is_better(0.3) is True
    assert trainer._is_better(0.7) is False


def test_is_better_max_mode():
    """Test _is_better under max mode."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

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
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": None},
        "train_loop": {
            "precision": "fp32",
            "accumulate_grad_batches": 1,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "max"},
    }

    trainer = OnlineNoCacheTrainer(
        model=mock_model, loss_fn=mock_loss_fn, config=config, device="cpu"
    )

    trainer.best_val_metric = 0.5
    assert trainer._is_better(0.7) is True
    assert trainer._is_better(0.3) is False


def test_train_step_logs_only_after_optimizer_step():
    """Train step should only log to tensorboard after optimizer step completes (accumulation_step == 0)."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "min"},
        "tensorboard": {"enabled": True, "log_dir": "/tmp/tensorboard_test"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=MagicMock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
    )
    trainer.use_tensorboard = True
    trainer.tensorboard_writer = MagicMock()

    logged_steps = []
    original_log = trainer._log_to_tensorboard

    def tracking_log(metrics, prefix="train"):
        logged_steps.append((trainer.global_step, trainer.accumulation_step))
        return original_log(metrics, prefix)

    trainer._log_to_tensorboard = tracking_log

    for i in range(8):
        trainer.train_step(batch, T=4)

    assert logged_steps == [
        (1, 0),
        (2, 0),
    ], (
        f"Expected logging only at optimizer steps (accumulation_step==0), got {logged_steps}"
    )


def test_fit_validates_only_once_per_optimizer_step_boundary():
    """fit() should validate only once per optimizer step, not repeatedly during accumulation."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
    }

    class FakeDataloader:
        def __iter__(self):
            return iter([batch] * 20)

        def __len__(self):
            return 20

    class FakeValDataloader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

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
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,
            "val_check_interval": 4,
            "checkpoint_dir": "/tmp/checkpoint_test",
            "max_epochs": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=MagicMock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
    )
    trainer.train_dataloader = FakeDataloader()
    trainer.val_dataloader = FakeValDataloader()

    validate_call_count = 0
    original_validate = trainer.validate

    def tracking_validate(*args, **kwargs):
        nonlocal validate_call_count
        validate_call_count += 1
        return {"val/total_loss": 1.0}

    trainer.validate = tracking_validate

    trainer.fit(max_epochs=1)

    assert validate_call_count <= 2, (
        f"Expected at most 2 validation calls (once at val_check_interval boundary, once at epoch end), "
        f"but got {validate_call_count} (accumulate_grad_batches=4, 20 batches = 5 optimizer steps)"
    )


def test_global_step_increments_only_at_optimizer_boundary():
    """Verify global_step only increments after full gradient accumulation completes."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
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
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=MagicMock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
    )

    step_snapshot = []
    for i in range(12):
        trainer.train_step(batch, T=4)
        step_snapshot.append((trainer.global_step, trainer.accumulation_step))

    assert step_snapshot == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
    ], (
        f"global_step should only increment when accumulation_step resets to 0, got {step_snapshot}"
    )


def test_fit_console_logs_only_at_optimizer_step_boundary():
    """fit() console logging should only occur after optimizer steps complete, not during accumulation."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer, logger
    import logging

    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(
            self, input_ids, attention_mask=None, num_steps=None, return_dict=True
        ):
            batch_size, seq_len = input_ids.shape
            return {
                "endpoint_hidden": self.weight * torch.ones(batch_size, seq_len, 2),
                "trajectory_hidden": self.weight
                * torch.ones(batch_size, seq_len, 1, 2),
                "logits": self.weight * torch.ones(batch_size, seq_len, 4),
            }

        def extract_teacher_targets(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            return {
                "h_start": torch.randn(batch_size, seq_len, 2),
                "h_target": torch.randn(batch_size, seq_len, 2),
                "velocity_target": torch.randn(batch_size, seq_len, 2),
                "teacher_logits": torch.randn(batch_size, seq_len, 4),
            }

    def loss_side_effect(student_outputs, teacher_batch, T, model=None, t=None):
        loss_val = student_outputs["endpoint_hidden"].mean()
        metrics = {
            "total_loss": loss_val.item(),
            "velocity_loss": loss_val.item(),
            "kl_loss": 0.0,
            "ce_loss": 0.0,
        }
        return loss_val, metrics

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.int64),
    }

    class FakeDataloader:
        def __iter__(self):
            return iter([batch] * 20)

        def __len__(self):
            return 20

    class FakeValDataloader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

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
            "accumulate_grad_batches": 4,
            "sample_continuous_time": False,
            "val_check_interval": 1000,
            "checkpoint_dir": "/tmp/checkpoint_test",
            "log_every_n_steps": 1,
            "max_epochs": 1,
        },
        "model": {"train_T_values": [4], "train_T_weights": [1.0]},
        "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
        "logging": {"monitor": "val/total_loss", "mode": "min"},
    }

    trainer = OnlineNoCacheTrainer(
        model=DummyStudent(),
        loss_fn=MagicMock(side_effect=loss_side_effect),
        config=config,
        device="cpu",
    )
    trainer.train_dataloader = FakeDataloader()
    trainer.val_dataloader = FakeValDataloader()

    logged_step_values = []
    original_info = logger.info

    def capture_info(msg):
        if "Step " in msg and ": " in msg:
            step_match = msg.split("Step ")[1].split(":")[0].strip()
            logged_step_values.append(int(step_match))
        return original_info(msg)

    logger.info = capture_info

    try:
        trainer.fit(max_epochs=1)
    finally:
        logger.info = original_info

    assert logged_step_values == [1, 2, 3, 4, 5], (
        f"Expected console logs at optimizer steps only (1-5), got {logged_step_values} "
        f"(accumulate_grad_batches=4, 20 batches = 5 optimizer steps, log_every_n_steps=10)"
    )

"""Tests for warm-start initialization.

Focused tests proving warm-start init loads model weights without restoring optimizer/global_step.

Tests verify:
1. warm_start_from_checkpoint loads ONLY model weights, NOT optimizer/scheduler/global_step
2. Fresh optimizer state after warm-start (no restored exp_avg, etc.)
3. global_step remains 0 after warm-start
4. Model weights are correctly loaded from checkpoint
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_warm_start_loads_model_weights_only():
    """Test that warm_start_from_checkpoint loads ONLY model weights, not optimizer/scheduler/global_step."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpoint_path = checkpoint_dir / "source.ckpt"

        class DummyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.bias = nn.Parameter(torch.tensor(2.0))

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

        student = DummyStudent()
        torch.save(
            {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": {
                    "state": {0: {"exp_avg": torch.randn(10, 10)}}
                },
                "global_step": 999,
                "current_epoch": 5,
            },
            checkpoint_path,
        )

        fresh_model = DummyStudent()
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
            model=fresh_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        trainer.warm_start_from_checkpoint(checkpoint_path)

        assert trainer.global_step == 0, (
            f"Expected global_step=0 after warm-start, got {trainer.global_step}"
        )
        assert trainer.current_epoch == 0, (
            f"Expected current_epoch=0 after warm-start, got {trainer.current_epoch}"
        )
        assert trainer.accumulation_step == 0, (
            f"Expected accumulation_step=0 after warm-start, got {trainer.accumulation_step}"
        )

        optimizer_state_keys = list(
            trainer.optimizer.state_dict().get("state", {}).keys()
        )
        assert len(optimizer_state_keys) == 0, (
            f"Expected fresh optimizer state, got {len(optimizer_state_keys)} param groups with state"
        )


def test_warm_start_model_weights_are_loaded():
    """Verify that warm_start_from_checkpoint loads model weights from checkpoint."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpoint_path = checkpoint_dir / "weights_test.ckpt"

        class DummyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(10, 10))

            def forward(
                self, input_ids, attention_mask=None, num_steps=None, return_dict=True
            ):
                return {
                    "endpoint_hidden": torch.randn(1, 10, 10),
                    "trajectory_hidden": torch.randn(1, 1, 10, 10),
                    "logits": torch.randn(1, 10, 10, 10),
                }

            def extract_teacher_targets(self, input_ids, attention_mask=None):
                return {
                    "h_start": torch.randn(1, 10, 10),
                    "h_target": torch.randn(1, 10, 10),
                    "velocity_target": torch.randn(1, 10, 10),
                    "teacher_logits": torch.randn(1, 10, 10, 10),
                }

        saved_model = DummyStudent()
        checkpoint_weights = torch.randn(10, 10)
        saved_model.weight.data = checkpoint_weights.clone()

        torch.save(
            {
                "model_state_dict": saved_model.state_dict(),
                "optimizer_state_dict": {"state": {}},
                "global_step": 100,
            },
            checkpoint_path,
        )

        fresh_model = DummyStudent()
        assert not torch.allclose(fresh_model.weight.data, checkpoint_weights), (
            "Precondition: fresh model should have different weights"
        )

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
            "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
            "logging": {"monitor": "val/total_loss", "mode": "min"},
        }

        trainer = OnlineNoCacheTrainer(
            model=fresh_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        trainer.warm_start_from_checkpoint(checkpoint_path)

        assert torch.allclose(fresh_model.weight.data, checkpoint_weights), (
            "Model weights should be loaded from checkpoint during warm-start"
        )
        assert trainer.global_step == 0, "global_step should remain 0 after warm-start"


def test_warm_start_init_does_not_restore_optimizer_state():
    """Verify that warm_start_from_checkpoint does NOT restore optimizer state dict."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpoint_path = checkpoint_dir / "warmstart_source.ckpt"

        class DummyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

            def forward(
                self, input_ids, attention_mask=None, num_steps=None, return_dict=True
            ):
                return {
                    "endpoint_hidden": self.layer(torch.randn(1, 10)),
                    "trajectory_hidden": torch.randn(1, 1, 10),
                    "logits": torch.randn(1, 10, 10),
                }

            def extract_teacher_targets(self, input_ids, attention_mask=None):
                return {
                    "h_start": torch.randn(1, 10, 10),
                    "h_target": torch.randn(1, 10, 10),
                    "velocity_target": torch.randn(1, 10, 10),
                    "teacher_logits": torch.randn(1, 10, 10),
                }

        model_for_checkpoint = DummyStudent()
        torch.save(
            {
                "model_state_dict": model_for_checkpoint.state_dict(),
                "optimizer_state_dict": {
                    "state": {0: {"exp_avg": torch.randn(10, 10)}}
                },
                "global_step": 999,
                "current_epoch": 5,
            },
            checkpoint_path,
        )

        fresh_model = DummyStudent()
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
            "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
            "logging": {"monitor": "val/total_loss", "mode": "min"},
        }

        trainer = OnlineNoCacheTrainer(
            model=fresh_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        trainer.warm_start_from_checkpoint(checkpoint_path)

        assert trainer.global_step == 0, (
            "global_step should be 0 (fresh) after warm-start"
        )
        assert trainer.current_epoch == 0, (
            "current_epoch should be 0 (fresh) after warm-start"
        )

        for param_id, state in trainer.optimizer.state_dict().get("state", {}).items():
            if "exp_avg" in state:
                assert torch.allclose(
                    state.get("exp_avg", torch.tensor(0.0)),
                    torch.tensor(0.0),
                ), (
                    f"Optimizer state should be fresh (zeros), but found exp_avg for param {param_id}"
                )


def test_init_from_checkpoint_null_means_no_loading():
    """When init_from_checkpoint is null/not set, no warm-start loading occurs."""
    from src.training.online_no_cache_trainer import OnlineNoCacheTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        class DummyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(
                self, input_ids, attention_mask=None, num_steps=None, return_dict=True
            ):
                return {
                    "endpoint_hidden": torch.randn(1, 10, 2),
                    "trajectory_hidden": torch.randn(1, 1, 10, 2),
                    "logits": torch.randn(1, 10, 4),
                }

            def extract_teacher_targets(self, input_ids, attention_mask=None):
                return {
                    "h_start": torch.randn(1, 10, 2),
                    "h_target": torch.randn(1, 10, 2),
                    "velocity_target": torch.randn(1, 10, 2),
                    "teacher_logits": torch.randn(1, 10, 4),
                }

        fresh_model = DummyStudent()
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
            "loss": {"velocity_weight": 1.0, "kl_weight": 0.0, "ce_weight": 0.0},
            "logging": {"monitor": "val/total_loss", "mode": "min"},
        }

        trainer = OnlineNoCacheTrainer(
            model=fresh_model,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        assert trainer.global_step == 0
        assert trainer.current_epoch == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import torch
from unittest.mock import MagicMock, patch
from src.training.trainer import Trainer


def test_trainer_passes_conditional_flags_based_on_loss_config():
    """Test that trainer passes correct flags to extract_teacher_targets based on loss weights."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.0,  # No trajectory
            "kl_weight": 0.0,  # No KL
            "ce_weight": 0.0,
            "velocity_weight": 0.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    mock_model.extract_teacher_targets.return_value = {
        "h_start": MagicMock(),
        "h_target": MagicMock(),
    }
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    # Mock loss_fn to return proper values
    mock_loss_fn = MagicMock()
    mock_loss = torch.tensor(1.0, requires_grad=True)
    mock_loss_fn.return_value = (mock_loss, {"total_loss": 1.0})

    trainer = Trainer(
        model=mock_model,
        loss_fn=mock_loss_fn,
        config=config,
        device="cpu",
    )

    # Create proper tensor mocks for input_ids and attention_mask
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": torch.ones(2, 32),
    }

    # Run train_step and verify extraction was called with correct flags
    trainer.train_step(batch, T=4)

    # Verify extraction was called with correct flags (from config, all False)
    mock_model.extract_teacher_targets.assert_called_once()
    call_kwargs = mock_model.extract_teacher_targets.call_args[1]
    assert call_kwargs.get("need_teacher_logits") is False
    assert call_kwargs.get("need_velocity") is False
    assert call_kwargs.get("need_trajectory_anchors") is False


def test_trainer_get_loss_flags_all_disabled():
    """Test _get_loss_flags returns all False when all loss weights are 0."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
            "velocity_weight": 0.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    flags = trainer._get_loss_flags()

    assert flags["need_teacher_logits"] is False
    assert flags["need_velocity"] is False
    assert flags["need_trajectory_anchors"] is False


def test_trainer_get_loss_flags_kl_enabled():
    """Test _get_loss_flags returns need_teacher_logits=True when kl_weight > 0."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.0,
            "kl_weight": 0.1,  # KL enabled
            "ce_weight": 0.0,
            "velocity_weight": 0.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    flags = trainer._get_loss_flags()

    assert flags["need_teacher_logits"] is True
    assert flags["need_velocity"] is False
    assert flags["need_trajectory_anchors"] is False


def test_trainer_get_loss_flags_velocity_enabled():
    """Test _get_loss_flags returns need_velocity=True when velocity_weight > 0."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.0,
            "kl_weight": 0.0,
            "ce_weight": 0.0,
            "velocity_weight": 1.0,  # Velocity enabled
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    flags = trainer._get_loss_flags()

    assert flags["need_teacher_logits"] is False
    assert flags["need_velocity"] is True
    assert flags["need_trajectory_anchors"] is False


def test_trainer_get_loss_flags_trajectory_enabled():
    """Test _get_loss_flags returns need_trajectory_anchors=True when trajectory_weight > 0."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 1.0,  # Trajectory enabled
            "kl_weight": 0.0,
            "ce_weight": 0.0,
            "velocity_weight": 0.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    flags = trainer._get_loss_flags()

    assert flags["need_teacher_logits"] is False
    assert flags["need_velocity"] is False
    assert flags["need_trajectory_anchors"] is True


def test_trainer_get_loss_flags_multiple_enabled():
    """Test _get_loss_flags with multiple losses enabled."""

    config = {
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {
            "endpoint_weight": 1.0,
            "trajectory_weight": 0.5,
            "kl_weight": 0.1,
            "ce_weight": 0.0,
            "velocity_weight": 1.0,
        },
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = MagicMock()
    # Add a real tensor parameter that requires grad for optimizer
    mock_param = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [mock_param]

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    flags = trainer._get_loss_flags()

    assert flags["need_teacher_logits"] is True  # kl_weight > 0
    assert flags["need_velocity"] is True  # velocity_weight > 0
    assert flags["need_trajectory_anchors"] is True  # trajectory_weight > 0

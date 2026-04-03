import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from src.training.trainer import Trainer


def create_mock_model():
    """Create a mock model with parameters that work with optimizer."""
    mock_model = MagicMock()
    # Create a real parameter with requires_grad=True
    mock_param = torch.nn.Parameter(torch.randn(1))
    mock_model.parameters.return_value = [mock_param]
    mock_model.extract_teacher_targets.return_value = {
        "h_start": MagicMock(),
        "h_target": MagicMock(),
    }
    return mock_model


@patch("src.training.trainer.wandb")
def test_wandb_init_with_config(mock_wandb):
    """Test that wandb is initialized with config."""
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    config = {
        "experiment_name": "test_exp",
        "seed": 1337,
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "midflowlm",
                "entity": "myteam",
                "tags": ["v0.1", "test"],
            }
        },
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {},
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = create_mock_model()

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    assert trainer.use_wandb is True
    mock_wandb.init.assert_called_once()

    # Verify init args
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs["project"] == "midflowlm"
    assert call_kwargs["entity"] == "myteam"
    assert call_kwargs["tags"] == ["v0.1", "test"]


@patch("src.training.trainer.wandb")
def test_wandb_log_metrics(mock_wandb):
    """Test that metrics are logged to wandb."""
    mock_wandb.init.return_value = MagicMock()

    config = {
        "experiment_name": "test",
        "logging": {"wandb": {"enabled": True, "project": "test"}},
        "model": {"name": "Qwen/Qwen3.5-0.8B", "max_steps_T": 4},
        "replacement_model": {"start_layer": 8, "end_layer": 11},
        "loss": {},
        "train_loop": {},
        "optimizer": {},
        "scheduler": {},
    }

    mock_model = create_mock_model()

    trainer = Trainer(
        model=mock_model,
        loss_fn=MagicMock(),
        config=config,
        device="cpu",
    )

    # Simulate logging
    trainer._log_to_wandb({"train/loss": 0.5, "train/kl": 0.1}, step=100)

    mock_wandb.log.assert_called_once()
    call_args = mock_wandb.log.call_args[0][0]
    assert call_args["train/loss"] == 0.5
    assert call_args["train/kl"] == 0.1
    assert call_args["step"] == 100

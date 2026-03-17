"""Contract tests for architecture training loss defaults.

These tests verify that architecture training defaults to hidden-state supervision
without requiring offline logits.
"""

import pytest
import torch


class TestArchitectureLossContract:
    """Test default architecture loss contract (no logits required)."""

    def test_default_architecture_config_disables_kl_when_logits_are_absent(self):
        """Test that default config disables KL loss (kl_weight=0.0)."""
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

    def test_kl_path_raises_targeted_error_when_teacher_logits_are_missing(self):
        """Test that KL loss raises targeted error when logits are missing."""
        from src.training.losses import DistillationLoss, LossConfig

        loss_fn = DistillationLoss(
            LossConfig(
                endpoint_weight=1.0,
                trajectory_weight=1.0,
                kl_weight=0.25,
                ce_weight=0.0,
            ),
            span_depth=4,
        )

        # Create dummy student outputs
        student_outputs = {
            "endpoint_hidden": torch.randn(2, 128, 32),
            "trajectory_hidden": [torch.randn(2, 128, 32) for _ in range(4)],
            "logits": torch.randn(2, 128, 1000),
        }

        # Create teacher batch without logits
        teacher_batch_without_logits = {
            "h_target": torch.randn(2, 128, 32),
            "trajectory_targets": [torch.randn(2, 128, 32) for _ in range(4)],
        }

        with pytest.raises(
            ValueError, match="not part of the default architecture-training cache"
        ):
            loss_fn(
                student_outputs=student_outputs,
                teacher_batch=teacher_batch_without_logits,
                T=4,
            )

    def test_hidden_state_losses_work_without_logits(self):
        """Test that endpoint and trajectory losses work without logits."""
        from src.training.losses import DistillationLoss, LossConfig

        loss_fn = DistillationLoss(
            LossConfig(
                endpoint_weight=1.0, trajectory_weight=1.0, kl_weight=0.0, ce_weight=0.0
            ),
            span_depth=4,
        )

        # Create dummy student outputs
        student_outputs = {
            "endpoint_hidden": torch.randn(2, 128, 32),
            "trajectory_hidden": [torch.randn(2, 128, 32) for _ in range(4)],
        }

        # Create teacher batch without logits
        teacher_batch = {
            "h_target": torch.randn(2, 128, 32),
            "trajectory_targets": [torch.randn(2, 128, 32) for _ in range(4)],
        }

        # Should not raise an error
        loss, metrics = loss_fn(
            student_outputs=student_outputs, teacher_batch=teacher_batch, T=4
        )

        assert isinstance(loss, torch.Tensor)
        assert "endpoint_loss" in metrics
        assert "trajectory_loss" in metrics

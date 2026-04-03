import pytest
import torch
from unittest.mock import patch, MagicMock
from src.model.student_qwen import FrozenQwenStudent


def test_extract_teacher_targets_conditional():
    """Test that extract_teacher_targets can skip unneeded targets."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    # When only endpoint needed, teacher_logits should not be returned
    targets = model.extract_teacher_targets(
        input_ids,
        attention_mask,
        need_teacher_logits=False,
        need_velocity=False,
    )
    assert "teacher_logits" not in targets
    assert "velocity_target" not in targets
    assert "h_start" in targets
    assert "h_target" in targets


def test_teacher_logits_not_computed_when_not_needed():
    """Verify that teacher model forward is NOT called with output_logits when not needed."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    # Mock the underlying model forward
    with patch.object(model.model, "forward") as mock_forward:
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(2, 64, 896) for _ in range(13)]
        mock_output.logits = torch.randn(2, 64, 1000)
        mock_forward.return_value = mock_output

        # Call with need_teacher_logits=False
        targets = model.extract_teacher_targets(
            input_ids,
            attention_mask,
            need_teacher_logits=False,
            need_velocity=True,
        )

        # Verify forward was called
        assert mock_forward.called, "Model forward was not called"

        # Get the call arguments
        _, call_kwargs = mock_forward.call_args

        # Verify output_logits is set to False when need_teacher_logits=False
        assert call_kwargs.get("output_logits") is False, (
            f"Expected output_logits=False, got {call_kwargs.get('output_logits')}"
        )

        # Verify teacher_logits is not in targets
        assert "teacher_logits" not in targets, (
            "teacher_logits should not be in targets when need_teacher_logits=False"
        )


def test_velocity_target_computed_when_needed():
    """Test that velocity_target is computed when need_velocity=True."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    targets = model.extract_teacher_targets(
        input_ids,
        attention_mask,
        need_teacher_logits=False,
        need_velocity=True,
    )

    assert "velocity_target" in targets
    assert targets["velocity_target"].shape == targets["h_target"].shape
    # Verify velocity is h_target - h_start
    expected_velocity = targets["h_target"] - targets["h_start"]
    assert torch.allclose(targets["velocity_target"], expected_velocity)


def test_trajectory_anchors_extracted_when_requested():
    """Test that trajectory anchors h8,h9,h10,h11 are extracted when requested."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    targets = model.extract_teacher_targets(
        input_ids,
        attention_mask,
        need_teacher_logits=False,
        need_velocity=False,
        need_trajectory_anchors=True,
    )

    # Should have trajectory_anchors dict with h8, h9, h10, h11
    assert "trajectory_anchors" in targets
    anchors = targets["trajectory_anchors"]
    assert "h8" in anchors
    assert "h9" in anchors
    assert "h10" in anchors
    assert "h11" in anchors


def test_all_targets_returned_by_default():
    """Test that all targets are returned when no flags specified (backward compatibility)."""
    model = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=4,
        device="cpu",
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64)

    # Call without any flags - should return all targets for backward compatibility
    targets = model.extract_teacher_targets(input_ids, attention_mask)

    assert "h_start" in targets
    assert "h_target" in targets
    assert "velocity_target" in targets
    assert "teacher_logits" in targets

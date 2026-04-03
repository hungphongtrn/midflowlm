import pytest
import torch
from src.model.student_qwen import FrozenQwenStudent


def test_create_a1_projector_student():
    """Test creating student with A1 family via config."""
    student = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=8,
        device="cpu",
        family="one_shot_projector",
    )

    assert student.family == "one_shot_projector"
    assert student.family_interface is not None

    # Should be able to forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = student(input_ids, num_steps=1, return_dict=True)
    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape[0] == 2


def test_create_a2_recurrent_student():
    """Test creating student with A2 family."""
    student = FrozenQwenStudent(
        model_name="Qwen/Qwen3.5-0.8B",
        start_layer=8,
        end_layer=11,
        max_steps_T=8,
        device="cpu",
        family="shared_recurrent_residual",
    )

    assert student.family == "shared_recurrent_residual"

    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = student(input_ids, num_steps=4, return_dict=True)
    assert "logits" in outputs

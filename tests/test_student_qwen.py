"""
Tests for frozen student wrapper around Qwen.

These tests verify:
1. Only replacement block parameters are trainable
2. Bypass mode reproduces teacher outputs
3. Wrapper honors configurable start_layer/end_layer
4. Validation can run with different T values without rebuilding model
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml


@pytest.fixture
def config():
    """Load v0 config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "v0_onemotif.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_config(config):
    """Extract model configuration."""
    return {
        "model_name": config["model"]["name"],
        "hidden_size": 896,  # Qwen3.5-0.8B hidden size
        "num_layers": 36,  # Qwen3.5-0.8B has 36 layers
        "max_steps_T": config["model"]["max_steps_T"],
        "start_layer": config["replacement_model"]["start_layer"],
        "end_layer": config["replacement_model"]["end_layer"],
    }


@pytest.fixture
def sample_input(model_config, device):
    """Create sample input for testing."""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    return input_ids, attention_mask


class TestStudentQwenImports:
    """Test that student_qwen module can be imported."""

    def test_import_student_qwen(self):
        """Test that src.model.student_qwen exists and can be imported."""
        from src.model import student_qwen

        assert student_qwen is not None

    def test_import_frozen_qwen_student(self):
        """Test that FrozenQwenStudent class exists."""
        from src.model.student_qwen import FrozenQwenStudent

        assert FrozenQwenStudent is not None


class TestTrainableParameters:
    """Test that only replacement block parameters are trainable."""

    def test_only_midblock_is_trainable(self, model_config, device):
        """Test that only the IterativeMidblock has trainable parameters."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Get all trainable parameters
        trainable_params = []
        frozen_params = []

        for name, param in student.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        # All trainable parameters should belong to the midblock
        for name in trainable_params:
            assert "midblock" in name, f"Trainable parameter outside midblock: {name}"

        # There should be some trainable parameters
        assert len(trainable_params) > 0, "No trainable parameters found"

        # The frozen parameters should include Qwen layers
        qwen_frozen = any("model" in name for name in frozen_params)
        assert qwen_frozen, "No frozen Qwen parameters found"

    def test_frozen_qwen_parameters_no_grad(self, model_config, device):
        """Test that Qwen parameters have requires_grad=False."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Check that base model parameters are frozen
        for name, param in student.named_parameters():
            if "model." in name and "midblock" not in name:
                assert not param.requires_grad, (
                    f"Qwen parameter should be frozen: {name}"
                )

    def test_midblock_parameters_trainable(self, model_config, device):
        """Test that midblock parameters have requires_grad=True."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Check that midblock parameters are trainable
        midblock_params_found = False
        for name, param in student.named_parameters():
            if "midblock" in name:
                midblock_params_found = True
                assert param.requires_grad, (
                    f"Midblock parameter should be trainable: {name}"
                )

        assert midblock_params_found, "No midblock parameters found"


class TestBypassMode:
    """Test that bypass mode reproduces teacher outputs."""

    def test_bypass_mode_outputs_match_teacher(
        self, model_config, device, sample_input
    ):
        """Test that bypass mode produces same outputs as teacher."""
        from src.model.student_qwen import FrozenQwenStudent
        from src.model.qwen_parity import QwenInspector

        input_ids, attention_mask = sample_input

        # Create student in bypass mode
        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
            bypass_mode=True,
        )

        # Get teacher outputs
        inspector = QwenInspector(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            device=device,
        )
        teacher_logits = inspector.extract_final_logits(input_ids, attention_mask)

        # Get student outputs in bypass mode
        student_logits = student(input_ids, attention_mask)

        # Outputs should match
        assert torch.allclose(student_logits, teacher_logits, atol=1e-5)

    def test_bypass_mode_has_no_midblock(self, model_config, device):
        """Test that bypass mode doesn't create a midblock."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
            bypass_mode=True,
        )

        assert student.midblock is None, "Midblock should not exist in bypass mode"


class TestConfigurableLayers:
    """Test that wrapper honors configurable start_layer/end_layer."""

    def test_default_span_8_11(self, model_config, device):
        """Test that default span is layers 8-11."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=8,
            end_layer=11,
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        assert student.start_layer == 8
        assert student.end_layer == 11
        assert student.span_depth == 4

    def test_custom_span_config(self, model_config, device):
        """Test custom start/end layer configuration."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=4,
            end_layer=7,
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        assert student.start_layer == 4
        assert student.end_layer == 7
        assert student.span_depth == 4

    def test_invalid_span_raises_error(self, model_config, device):
        """Test that invalid span raises an error."""
        from src.model.student_qwen import FrozenQwenStudent

        with pytest.raises(ValueError):
            FrozenQwenStudent(
                model_name=model_config["model_name"],
                start_layer=-1,
                end_layer=11,
                max_steps_T=model_config["max_steps_T"],
                device=device,
            )

        with pytest.raises(ValueError):
            FrozenQwenStudent(
                model_name=model_config["model_name"],
                start_layer=8,
                end_layer=100,  # Too large for Qwen3.5-0.8B
                max_steps_T=model_config["max_steps_T"],
                device=device,
            )


class TestVariableT:
    """Test that validation can run with different T values without rebuilding model."""

    def test_variable_t_without_rebuild(self, model_config, device, sample_input):
        """Test running with different T values doesn't require rebuilding."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Run with different T values
        for num_steps in [1, 2, 4, 8]:
            output = student(input_ids, attention_mask, num_steps=num_steps)
            assert output.shape[0] == input_ids.shape[0]
            assert output.shape[1] == input_ids.shape[1]
            # Shape should be [batch, seq, vocab_size]
            assert output.dim() == 3

    def test_num_steps_defaults_to_max(self, model_config, device, sample_input):
        """Test that num_steps defaults to max_steps_T."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=8,
            device=device,
        )

        # Call without specifying num_steps
        output = student(input_ids, attention_mask)
        assert output.shape[0] == input_ids.shape[0]
        assert output.shape[1] == input_ids.shape[1]

    def test_num_steps_can_exceed_max_for_inference(
        self, model_config, device, sample_input
    ):
        """Test that num_steps can exceed max_steps_T for inference flexibility."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=4,
            device=device,
        )

        # Run with larger T - should work for inference
        output = student(input_ids, attention_mask, num_steps=8)
        assert output.shape[0] == input_ids.shape[0]
        assert output.shape[1] == input_ids.shape[1]


class TestODESolverIntegration:
    """Test ODE solver integration in student model."""

    def test_solver_method_euler(self, model_config, device, sample_input):
        """Test that solver_method='euler' works."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Run with explicit solver method
        output = student(input_ids, attention_mask, num_steps=8, solver_method="euler")
        assert output.shape[:2] == input_ids.shape
        # Shape should be [batch, seq, vocab_size]
        assert output.dim() == 3

    def test_solver_method_rk4(self, model_config, device, sample_input):
        """Test that solver_method='rk4' works."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Run with RK4 solver
        output = student(input_ids, attention_mask, num_steps=8, solver_method="rk4")
        assert output.shape[:2] == input_ids.shape
        assert output.dim() == 3

    def test_num_steps_100_with_euler(self, model_config, device, sample_input):
        """Test running with many steps and euler method."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=8,
            device=device,
        )

        # Run with 100 steps - should work with proper step_size
        output = student(
            input_ids, attention_mask, num_steps=100, solver_method="euler"
        )
        assert output.shape[:2] == input_ids.shape
        assert output.dim() == 3

    def test_ode_solver_with_return_dict(self, model_config, device, sample_input):
        """Test ODE solver with return_dict=True returns trajectory."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=4,
            device=device,
        )

        # Run with return_dict to get trajectory
        output = student(
            input_ids,
            attention_mask,
            num_steps=4,
            solver_method="euler",
            return_dict=True,
        )

        assert "logits" in output
        assert "endpoint_hidden" in output
        assert "trajectory_hidden" in output
        assert output["logits"].shape[:2] == input_ids.shape
        # Trajectory should have shape [batch, seq, num_steps, hidden]
        assert output["trajectory_hidden"].shape[2] == 4  # num_steps


class TestHFOutputCompatibility:
    """Test HuggingFace-style output compatibility."""

    def test_output_has_logits_attribute(self, model_config, device, sample_input):
        """Test that output has logits attribute for HF compatibility."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        # Use return_dict=True
        output = student(input_ids, attention_mask, return_dict=True)

        assert hasattr(output, "logits")
        assert output.logits.shape[0] == input_ids.shape[0]
        assert output.logits.shape[1] == input_ids.shape[1]

    def test_output_can_be_dict(self, model_config, device, sample_input):
        """Test that output can be returned as dict."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        output = student(input_ids, attention_mask, return_dict=True)
        assert isinstance(output, dict)
        assert "logits" in output


class TestParameterCounts:
    """Test parameter count reporting."""

    def test_get_trainable_parameter_count(self, model_config, device):
        """Test that trainable parameter count is reported correctly."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        trainable = student.get_trainable_parameter_count()
        total = student.get_total_parameter_count()

        assert trainable > 0
        assert total > trainable
        # Trainable should be much smaller than total (only midblock is trainable)
        assert trainable < total * 0.1, "Trainable params should be <10% of total"

    def test_bypass_mode_has_zero_trainable(self, model_config, device):
        """Test that bypass mode has zero trainable parameters."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
            bypass_mode=True,
        )

        trainable = student.get_trainable_parameter_count()
        assert trainable == 0, "Bypass mode should have zero trainable parameters"


class TestExtractTeacherTargets:
    """Test the extract_teacher_targets method for online-no-cache training."""

    def test_extract_teacher_targets_exists(self, model_config, device):
        """Test that extract_teacher_targets method exists."""
        from src.model.student_qwen import FrozenQwenStudent

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        assert hasattr(student, "extract_teacher_targets")
        assert callable(student.extract_teacher_targets)

    def test_extract_teacher_targets_returns_correct_keys(
        self, model_config, device, sample_input
    ):
        """Test that extract_teacher_targets returns all required keys."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        result = student.extract_teacher_targets(input_ids, attention_mask)

        assert "h_start" in result
        assert "h_target" in result
        assert "velocity_target" in result
        assert "teacher_logits" in result

    def test_extract_teacher_targets_single_forward_pass(
        self, model_config, device, sample_input
    ):
        """Test that extract_teacher_targets uses a single forward pass."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        result = student.extract_teacher_targets(input_ids, attention_mask)

        batch_size, seq_len = input_ids.shape
        hidden_size = result["h_start"].shape[-1]

        assert result["h_start"].shape == (batch_size, seq_len, hidden_size)
        assert result["h_target"].shape == (batch_size, seq_len, hidden_size)
        assert result["velocity_target"].shape == (batch_size, seq_len, hidden_size)
        assert result["teacher_logits"].shape[0] == batch_size
        assert result["teacher_logits"].shape[1] == seq_len

    def test_extract_teacher_targets_velocity_is_difference(
        self, model_config, device, sample_input
    ):
        """Test that velocity_target = h_target - h_start."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        result = student.extract_teacher_targets(input_ids, attention_mask)

        expected_velocity = result["h_target"] - result["h_start"]
        assert torch.allclose(result["velocity_target"], expected_velocity, atol=1e-6)

    def test_extract_teacher_targets_no_grad(self, model_config, device, sample_input):
        """Test that extract_teacher_targets runs under no_grad."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        result = student.extract_teacher_targets(input_ids, attention_mask)

        for key, tensor in result.items():
            assert not tensor.requires_grad, f"{key} should not require grad"

    def test_extract_teacher_targets_parity_with_qwen_inspector(
        self, model_config, device, sample_input
    ):
        """Test that extract_teacher_targets matches QwenInspector outputs."""
        from src.model.student_qwen import FrozenQwenStudent
        from src.model.qwen_parity import QwenInspector

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        teacher_result = student.extract_teacher_targets(input_ids, attention_mask)

        inspector = QwenInspector(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            device=device,
        )

        h_start_inspector = inspector.extract_h_start(input_ids, attention_mask)
        h_target_inspector = inspector.extract_h_target(input_ids, attention_mask)
        logits_inspector = inspector.extract_final_logits(input_ids, attention_mask)

        assert torch.allclose(teacher_result["h_start"], h_start_inspector, atol=1e-5)
        assert torch.allclose(teacher_result["h_target"], h_target_inspector, atol=1e-5)
        assert torch.allclose(
            teacher_result["teacher_logits"], logits_inspector, atol=1e-5
        )

    def test_extract_teacher_targets_bypass_mode(
        self, model_config, device, sample_input
    ):
        """Test extract_teacher_targets in bypass mode still works."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
            bypass_mode=True,
        )

        result = student.extract_teacher_targets(input_ids, attention_mask)

        assert "h_start" in result
        assert "h_target" in result
        assert "velocity_target" in result
        assert "teacher_logits" in result

        expected_velocity = result["h_target"] - result["h_start"]
        assert torch.allclose(result["velocity_target"], expected_velocity, atol=1e-6)

    def test_extract_teacher_targets_reuses_student_qwen_weights(
        self, model_config, device, sample_input
    ):
        """Test that extract_teacher_targets uses the same model as the student."""
        from src.model.student_qwen import FrozenQwenStudent

        input_ids, attention_mask = sample_input

        student = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
        )

        teacher_result = student.extract_teacher_targets(input_ids, attention_mask)

        student_bypass = FrozenQwenStudent(
            model_name=model_config["model_name"],
            start_layer=model_config["start_layer"],
            end_layer=model_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            device=device,
            bypass_mode=True,
        )
        student_logits_bypass = student_bypass(input_ids, attention_mask)

        assert torch.allclose(
            teacher_result["teacher_logits"], student_logits_bypass, atol=1e-5
        )

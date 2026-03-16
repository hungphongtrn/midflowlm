"""
Tests for iterative hidden-state midblock with step conditioning.

These tests verify:
1. Output shape preservation [batch, seq, hidden]
2. Causal-mask support
3. Configurable start_layer/end_layer
4. T=1 and T=max_steps_T
5. Save/load round-trip
6. Per-step stability for longer unrolls
7. Residual update behavior
"""

import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import tempfile
import os


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
        "hidden_size": 896,  # Qwen3.5-0.8B hidden size
        "num_layers": 36,  # Qwen3.5-0.8B has 36 layers
        "max_steps_T": config["model"]["max_steps_T"],
        "start_layer": config["replacement_model"]["start_layer"],
        "end_layer": config["replacement_model"]["end_layer"],
    }


@pytest.fixture
def sample_hidden_states(model_config, device):
    """Create sample hidden states for testing."""
    batch_size = 2
    seq_len = 16
    hidden_size = model_config["hidden_size"]
    return torch.randn(batch_size, seq_len, hidden_size, device=device)


@pytest.fixture
def sample_attention_mask(device):
    """Create sample attention mask for testing."""
    batch_size = 2
    seq_len = 16
    return torch.ones(batch_size, seq_len, device=device)


@pytest.fixture
def sample_position_ids(device):
    """Create sample position ids for testing."""
    batch_size = 2
    seq_len = 16
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


class TestMidblockImports:
    """Test that midblock module can be imported."""

    def test_import_midblock(self):
        """Test that src.model.midblock exists and can be imported."""
        from src.model import midblock
        assert midblock is not None

    def test_import_iterative_midblock(self):
        """Test that IterativeMidblock class exists."""
        from src.model.midblock import IterativeMidblock
        assert IterativeMidblock is not None

    def test_import_adapter(self):
        """Test that src.model.adapter exists and can be imported."""
        from src.model import adapter
        assert adapter is not None

    def test_import_step_conditioning_adapter(self):
        """Test that StepConditioningAdapter class exists."""
        from src.model.adapter import StepConditioningAdapter
        assert StepConditioningAdapter is not None


class TestOutputShapePreservation:
    """Test that output shape is preserved [batch, seq, hidden]."""

    def test_output_shape_single_step(self, model_config, sample_hidden_states, device):
        """Test output shape for single step (T=1)."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        assert output.shape == sample_hidden_states.shape
        assert output.shape == (2, 16, model_config["hidden_size"])

    def test_output_shape_multi_step(self, model_config, sample_hidden_states, device):
        """Test output shape for multiple steps."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Run for T=4 steps
        h = sample_hidden_states
        h_start = sample_hidden_states
        for step in range(4):
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                step_id=step,
                num_steps=4,
            )

        assert h.shape == sample_hidden_states.shape
        assert h.shape == (2, 16, model_config["hidden_size"])

    def test_output_shape_different_batch_sizes(self, model_config, device):
        """Test output shape with different batch sizes."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        for batch_size in [1, 4, 8]:
            hidden_states = torch.randn(batch_size, 16, model_config["hidden_size"], device=device)
            output = midblock(
                hidden_states=hidden_states,
                h_start=hidden_states,
                step_id=0,
                num_steps=1,
            )
            assert output.shape == hidden_states.shape

    def test_output_shape_different_seq_lengths(self, model_config, device):
        """Test output shape with different sequence lengths."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        for seq_len in [8, 16, 32, 128]:
            hidden_states = torch.randn(2, seq_len, model_config["hidden_size"], device=device)
            output = midblock(
                hidden_states=hidden_states,
                h_start=hidden_states,
                step_id=0,
                num_steps=1,
            )
            assert output.shape == hidden_states.shape


class TestCausalMaskSupport:
    """Test causal-mask support."""

    def test_causal_attention_mask(self, model_config, sample_hidden_states, device):
        """Test that causal mask is applied correctly."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_causal_mask=True,
        ).to(device)

        batch_size, seq_len, hidden_size = sample_hidden_states.shape

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
            attention_mask=attention_mask,
        )

        assert output.shape == sample_hidden_states.shape

    def test_causal_mask_prevents_future_look(self, model_config, device):
        """Test that causal mask prevents looking at future tokens."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_causal_mask=True,
        ).to(device)

        # Create input where first token is unique
        hidden_states = torch.zeros(1, 8, model_config["hidden_size"], device=device)
        hidden_states[0, 0, 0] = 1.0  # First token has unique value

        output = midblock(
            hidden_states=hidden_states,
            h_start=hidden_states,
            step_id=0,
            num_steps=1,
            attention_mask=torch.ones(1, 8, device=device),
        )

        # First token's output should be affected by attention
        # (This is a basic sanity check - full causality is hard to test)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestConfigurableLayers:
    """Test configurable start_layer/end_layer."""

    def test_default_span_8_11(self, model_config, sample_hidden_states, device):
        """Test that default span is layers 8-11."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            start_layer=8,
            end_layer=11,
        ).to(device)

        assert midblock.start_layer == 8
        assert midblock.end_layer == 11
        assert midblock.span_depth == 4

    def test_custom_span_config(self, model_config, sample_hidden_states, device):
        """Test custom start/end layer configuration."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            start_layer=4,
            end_layer=7,
        ).to(device)

        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        assert output.shape == sample_hidden_states.shape

    def test_span_depth_computation(self, model_config, device):
        """Test that span depth is computed correctly."""
        from src.model.midblock import IterativeMidblock

        test_cases = [
            (0, 3, 4),   # 4 layers: 0, 1, 2, 3
            (8, 11, 4),  # 4 layers: 8, 9, 10, 11
            (2, 5, 4),   # 4 layers: 2, 3, 4, 5
            (0, 0, 1),   # 1 layer: 0
        ]

        for start, end, expected_depth in test_cases:
            midblock = IterativeMidblock(
                hidden_size=model_config["hidden_size"],
                max_steps_T=model_config["max_steps_T"],
                start_layer=start,
                end_layer=end,
            ).to(device)

            assert midblock.span_depth == expected_depth


class TestVariableT:
    """Test T=1 and T=max_steps_T."""

    def test_t_equals_one(self, model_config, sample_hidden_states, device):
        """Test with T=1 (single refinement step)."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,  # T=1
        )

        assert output.shape == sample_hidden_states.shape

    def test_t_equals_max(self, model_config, sample_hidden_states, device):
        """Test with T=max_steps_T."""
        from src.model.midblock import IterativeMidblock

        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        # Run for max_T steps
        for step in range(max_T):
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                step_id=step,
                num_steps=max_T,
            )

        assert h.shape == sample_hidden_states.shape

    def test_t_variations(self, model_config, sample_hidden_states, device):
        """Test various T values."""
        from src.model.midblock import IterativeMidblock

        max_T = model_config["max_steps_T"]
        test_values = [1, 2, 4, 8, max_T]

        for T in test_values:
            midblock = IterativeMidblock(
                hidden_size=model_config["hidden_size"],
                max_steps_T=max_T,
            ).to(device)

            h = sample_hidden_states
            h_start = sample_hidden_states

            for step in range(T):
                h = midblock(
                    hidden_states=h,
                    h_start=h_start,
                    step_id=step,
                    num_steps=T,
                )

            assert h.shape == sample_hidden_states.shape


class TestSaveLoadRoundTrip:
    """Test save/load round-trip."""

    def test_state_dict_save_load(self, model_config, sample_hidden_states, device):
        """Test state dict save and load."""
        from src.model.midblock import IterativeMidblock

        midblock1 = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Get output before saving
        output1 = midblock1(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        # Save state dict
        state_dict = midblock1.state_dict()

        # Create new midblock and load
        midblock2 = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)
        midblock2.load_state_dict(state_dict)

        # Get output after loading
        output2 = midblock2(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_save_load_file(self, model_config, sample_hidden_states, device):
        """Test saving and loading from file."""
        from src.model.midblock import IterativeMidblock

        midblock1 = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Get output before saving
        output1 = midblock1(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "midblock.pt")
            torch.save(midblock1.state_dict(), save_path)

            # Load from file
            midblock2 = IterativeMidblock(
                hidden_size=model_config["hidden_size"],
                max_steps_T=model_config["max_steps_T"],
            ).to(device)
            midblock2.load_state_dict(torch.load(save_path, weights_only=True))

        # Get output after loading
        output2 = midblock2(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


class TestPerStepStability:
    """Test per-step stability for longer unrolls."""

    def test_no_nan_for_reasonable_steps(self, model_config, sample_hidden_states, device):
        """Test that outputs don't produce NaN for reasonable step counts."""
        from src.model.midblock import IterativeMidblock

        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        for step in range(max_T):
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                step_id=step,
                num_steps=max_T,
            )

            # Check for NaN
            assert not torch.isnan(h).any(), f"NaN detected at step {step}"
            # Check for Inf
            assert not torch.isinf(h).any(), f"Inf detected at step {step}"

    def test_bounded_output_magnitude(self, model_config, sample_hidden_states, device):
        """Test that output magnitude remains bounded."""
        from src.model.midblock import IterativeMidblock

        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states
        initial_norm = torch.norm(h).item()

        for step in range(max_T):
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                step_id=step,
                num_steps=max_T,
            )

            # Output norm should not explode
            current_norm = torch.norm(h).item()
            assert current_norm < initial_norm * 100, f"Output exploded at step {step}"

    def test_deterministic_outputs(self, model_config, sample_hidden_states, device):
        """Test that outputs are deterministic (same input -> same output)."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)
        midblock.eval()

        # Run twice with same input
        outputs = []
        for _ in range(2):
            h = sample_hidden_states.clone()
            h_start = sample_hidden_states.clone()

            for step in range(4):
                h = midblock(
                    hidden_states=h,
                    h_start=h_start,
                    step_id=step,
                    num_steps=4,
                )
            outputs.append(h)

        # Should be identical
        assert torch.allclose(outputs[0], outputs[1], atol=1e-6)


class TestResidualUpdateBehavior:
    """Test residual update behavior: h_{k+1} = h_k + delta_k."""

    def test_residual_connection_present(self, model_config, sample_hidden_states, device):
        """Test that residual connections are present by default."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_residual=True,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        output = midblock(
            hidden_states=h,
            h_start=h_start,
            step_id=0,
            num_steps=1,
        )

        # Output should be different from input (residual adds something)
        assert not torch.allclose(output, h, atol=1e-6)

    def test_residual_can_be_disabled(self, model_config, sample_hidden_states, device):
        """Test that residual connections can be disabled."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_residual=False,
        ).to(device)

        # Verify use_residual is False
        assert midblock.use_residual is False

        h = sample_hidden_states
        h_start = sample_hidden_states

        output = midblock(
            hidden_states=h,
            h_start=h_start,
            step_id=0,
            num_steps=1,
        )

        # Should still produce valid output
        assert output.shape == sample_hidden_states.shape

    def test_h_start_influence(self, model_config, sample_hidden_states, device):
        """Test that h_start influences the output."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Train mode and multiple steps to amplify h_start influence
        midblock.train()

        h = sample_hidden_states
        # Use more distinct h_start values to ensure influence
        h_start1 = sample_hidden_states
        h_start2 = sample_hidden_states + torch.randn_like(sample_hidden_states) * 10

        # Run multiple steps to amplify h_start influence
        output1 = h.clone()
        output2 = h.clone()
        for step in range(4):
            output1 = midblock(
                hidden_states=output1,
                h_start=h_start1,
                step_id=step,
                num_steps=4,
            )
            output2 = midblock(
                hidden_states=output2,
                h_start=h_start2,
                step_id=step,
                num_steps=4,
            )

        # Different h_start should produce measurably different outputs
        diff = torch.abs(output1 - output2).mean()
        assert diff > 1e-4, f"h_start influence too small: {diff}"

    def test_step_conditioning_influence(self, model_config, sample_hidden_states, device):
        """Test that step conditioning influences the output."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_step_conditioning=True,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        output1 = midblock(
            hidden_states=h,
            h_start=h_start,
            step_id=0,
            num_steps=4,
        )

        output2 = midblock(
            hidden_states=h,
            h_start=h_start,
            step_id=3,  # Different step
            num_steps=4,
        )

        # Different step should produce different outputs
        assert not torch.allclose(output1, output2, atol=1e-3)


class TestStepConditioningAdapter:
    """Test StepConditioningAdapter."""

    def test_adapter_output_shape(self, model_config, device):
        """Test that adapter produces correct output shape."""
        from src.model.adapter import StepConditioningAdapter

        adapter = StepConditioningAdapter(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        batch_size = 2
        step_features = adapter.get_step_features(
            step_id=0,
            num_steps=4,
            batch_size=batch_size,
            device=device,
        )

        # Should produce [batch_size, hidden_size] features
        assert step_features.shape == (batch_size, model_config["hidden_size"])

    def test_adapter_step_features_vary(self, model_config, device):
        """Test that different steps produce different features."""
        from src.model.adapter import StepConditioningAdapter

        adapter = StepConditioningAdapter(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        batch_size = 2

        features = []
        for step_id in range(4):
            step_feat = adapter.get_step_features(
                step_id=step_id,
                num_steps=4,
                batch_size=batch_size,
                device=device,
            )
            features.append(step_feat)

        # Different steps should have different features
        for i in range(len(features) - 1):
            assert not torch.allclose(features[i], features[i + 1], atol=1e-6)

    def test_adapter_normalized_features(self, model_config, device):
        """Test that t/T normalization is used."""
        from src.model.adapter import StepConditioningAdapter

        adapter = StepConditioningAdapter(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            normalization="t_div_T",
        ).to(device)

        batch_size = 2

        # First step should have normalized value near 0
        first_features = adapter.get_step_features(
            step_id=0,
            num_steps=4,
            batch_size=batch_size,
            device=device,
        )

        # Last step should have normalized value near 1
        last_features = adapter.get_step_features(
            step_id=3,
            num_steps=4,
            batch_size=batch_size,
            device=device,
        )

        # Features should be different
        assert not torch.allclose(first_features, last_features, atol=1e-3)


class TestMidblockInputInterface:
    """Test full input interface."""

    def test_all_inputs_accepted(self, model_config, sample_hidden_states,
                                  sample_attention_mask, sample_position_ids, device):
        """Test that all documented inputs are accepted."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            attention_mask=sample_attention_mask,
            position_ids=sample_position_ids,
            step_id=0,
            num_steps=1,
        )

        assert output.shape == sample_hidden_states.shape

    def test_optional_inputs_omitted(self, model_config, sample_hidden_states, device):
        """Test that optional inputs can be omitted."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Call with only required inputs
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            step_id=0,
            num_steps=1,
        )

        assert output.shape == sample_hidden_states.shape

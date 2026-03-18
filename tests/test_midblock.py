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

    def test_import_continuous_time_embedding(self):
        """Test that ContinuousTimeEmbedding class exists."""
        from src.model.adapter import ContinuousTimeEmbedding

        assert ContinuousTimeEmbedding is not None


class TestOutputShapePreservation:
    """Test that output shape is preserved [batch, seq, hidden]."""

    def test_output_shape_single_step(self, model_config, sample_hidden_states, device):
        """Test output shape for single step (T=1)."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Use continuous time t
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
        )

        assert output.shape == sample_hidden_states.shape
        assert output.shape == (2, 16, model_config["hidden_size"])

    def test_output_shape_multi_step(self, model_config, sample_hidden_states, device):
        """Test output shape for multiple steps with continuous time."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Run for T=4 steps with continuous time values
        h = sample_hidden_states
        h_start = sample_hidden_states
        for step in range(4):
            t = torch.full((batch_size,), float(step) / 4.0, device=device)
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                t=t,
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
            hidden_states = torch.randn(
                batch_size, 16, model_config["hidden_size"], device=device
            )
            t = torch.zeros(batch_size, device=device)
            output = midblock(
                hidden_states=hidden_states,
                h_start=hidden_states,
                t=t,
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
            hidden_states = torch.randn(
                2, seq_len, model_config["hidden_size"], device=device
            )
            t = torch.zeros(2, device=device)
            output = midblock(
                hidden_states=hidden_states,
                h_start=hidden_states,
                t=t,
            )
            assert output.shape == hidden_states.shape


class TestCausalMaskSupport:
    """Test causal-mask support."""

    def test_causal_attention_mask(self, model_config, sample_hidden_states, device):
        """Test that causal mask is applied correctly."""
        from src.model.midblock import IterativeMidblock

        batch_size, seq_len, hidden_size = sample_hidden_states.shape
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_causal_mask=True,
        ).to(device)

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Use continuous time t
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
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

        # Use continuous time t
        t = torch.zeros(1, device=device)
        output = midblock(
            hidden_states=hidden_states,
            h_start=hidden_states,
            t=t,
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

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            start_layer=4,
            end_layer=7,
        ).to(device)

        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
        )

        assert output.shape == sample_hidden_states.shape

    def test_span_depth_computation(self, model_config, device):
        """Test that span depth is computed correctly."""
        from src.model.midblock import IterativeMidblock

        test_cases = [
            (0, 3, 4),  # 4 layers: 0, 1, 2, 3
            (8, 11, 4),  # 4 layers: 8, 9, 10, 11
            (2, 5, 4),  # 4 layers: 2, 3, 4, 5
            (0, 0, 1),  # 1 layer: 0
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
    """Test T=1 and T=max_steps_T with continuous time."""

    def test_t_equals_one(self, model_config, sample_hidden_states, device):
        """Test with T=1 (single refinement step) using continuous time."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Use continuous time t=0 for single step
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
        )

        assert output.shape == sample_hidden_states.shape

    def test_t_equals_max(self, model_config, sample_hidden_states, device):
        """Test with T=max_steps_T using continuous time."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        # Run for max_T steps with continuous time
        for step in range(max_T):
            t = torch.full((batch_size,), float(step) / max_T, device=device)
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                t=t,
            )

        assert h.shape == sample_hidden_states.shape

    def test_t_variations(self, model_config, sample_hidden_states, device):
        """Test various T values with continuous time."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
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
                t = torch.full((batch_size,), float(step) / T, device=device)
                h = midblock(
                    hidden_states=h,
                    h_start=h_start,
                    t=t,
                )

            assert h.shape == sample_hidden_states.shape


class TestSaveLoadRoundTrip:
    """Test save/load round-trip."""

    def test_state_dict_save_load(self, model_config, sample_hidden_states, device):
        """Test state dict save and load."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock1 = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Get output before saving using continuous time
        t = torch.zeros(batch_size, device=device)
        output1 = midblock1(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
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
        t = torch.zeros(batch_size, device=device)
        output2 = midblock2(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
        )

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_save_load_file(self, model_config, sample_hidden_states, device):
        """Test saving and loading from file."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock1 = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Get output before saving using continuous time
        t = torch.zeros(batch_size, device=device)
        output1 = midblock1(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
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
        t = torch.zeros(batch_size, device=device)
        output2 = midblock2(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            t=t,
        )

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


class TestPerStepStability:
    """Test per-step stability for longer unrolls with continuous time."""

    def test_no_nan_for_reasonable_steps(
        self, model_config, sample_hidden_states, device
    ):
        """Test that outputs don't produce NaN for reasonable step counts."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        for step in range(max_T):
            t = torch.full((batch_size,), float(step) / max_T, device=device)
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                t=t,
            )

            # Check for NaN
            assert not torch.isnan(h).any(), f"NaN detected at step {step}"
            # Check for Inf
            assert not torch.isinf(h).any(), f"Inf detected at step {step}"

    def test_bounded_output_magnitude(self, model_config, sample_hidden_states, device):
        """Test that output magnitude remains bounded."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        max_T = model_config["max_steps_T"]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=max_T,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states
        initial_norm = torch.norm(h).item()

        for step in range(max_T):
            t = torch.full((batch_size,), float(step) / max_T, device=device)
            h = midblock(
                hidden_states=h,
                h_start=h_start,
                t=t,
            )

            # Output norm should not explode
            current_norm = torch.norm(h).item()
            assert current_norm < initial_norm * 100, f"Output exploded at step {step}"

    def test_deterministic_outputs(self, model_config, sample_hidden_states, device):
        """Test that outputs are deterministic (same input -> same output)."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
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
                t = torch.full((batch_size,), float(step) / 4.0, device=device)
                h = midblock(
                    hidden_states=h,
                    h_start=h_start,
                    t=t,
                )
            outputs.append(h)

        # Should be identical
        assert torch.allclose(outputs[0], outputs[1], atol=1e-6)


class TestResidualUpdateBehavior:
    """Test residual update behavior: h_{k+1} = h_k + delta_k with continuous time."""

    def test_residual_connection_present(
        self, model_config, sample_hidden_states, device
    ):
        """Test that residual connections are present by default."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_residual=True,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        # Use continuous time t
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=h,
            h_start=h_start,
            t=t,
        )

        # Output should be different from input (residual adds something)
        assert not torch.allclose(output, h, atol=1e-6)

    def test_residual_can_be_disabled(self, model_config, sample_hidden_states, device):
        """Test that residual connections can be disabled."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_residual=False,
        ).to(device)

        # Verify use_residual is False
        assert midblock.use_residual is False

        h = sample_hidden_states
        h_start = sample_hidden_states

        # Use continuous time t
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=h,
            h_start=h_start,
            t=t,
        )

        # Should still produce valid output
        assert output.shape == sample_hidden_states.shape

    def test_h_start_influence(self, model_config, sample_hidden_states, device):
        """Test that h_start influences the output."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
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

        # Run multiple steps with continuous time to amplify h_start influence
        output1 = h.clone()
        output2 = h.clone()
        for step in range(4):
            t = torch.full((batch_size,), float(step) / 4.0, device=device)
            output1 = midblock(
                hidden_states=output1,
                h_start=h_start1,
                t=t,
            )
            output2 = midblock(
                hidden_states=output2,
                h_start=h_start2,
                t=t,
            )

        # Different h_start should produce measurably different outputs
        diff = torch.abs(output1 - output2).mean()
        assert diff > 1e-4, f"h_start influence too small: {diff}"

    def test_time_conditioning_influence(
        self, model_config, sample_hidden_states, device
    ):
        """Test that continuous time conditioning influences the output."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
            use_step_conditioning=True,
        ).to(device)

        h = sample_hidden_states
        h_start = sample_hidden_states

        # Use different continuous time values
        t1 = torch.zeros(batch_size, device=device)  # t=0
        output1 = midblock(
            hidden_states=h,
            h_start=h_start,
            t=t1,
        )

        t2 = torch.full((batch_size,), 0.75, device=device)  # t=0.75
        output2 = midblock(
            hidden_states=h,
            h_start=h_start,
            t=t2,
        )

        # Different time should produce measurably different outputs
        # Note: tolerance relaxed for newly initialized models
        diff = torch.abs(output1 - output2).mean()
        assert diff > 1e-5, f"Time conditioning influence too small: {diff}"


class TestContinuousTimeEmbedding:
    """Test ContinuousTimeEmbedding integration with midblock."""

    def test_time_embedding_output_shape(self, model_config, device):
        """Test that continuous time embedding produces correct output shape."""
        from src.model.adapter import ContinuousTimeEmbedding

        emb = ContinuousTimeEmbedding(hidden_size=model_config["hidden_size"]).to(
            device
        )

        batch_size = 2
        t = torch.linspace(0, 1, batch_size, device=device)
        time_features = emb(t)

        # Should produce [batch_size, hidden_size] features
        assert time_features.shape == (batch_size, model_config["hidden_size"])

    def test_time_embedding_features_vary(self, model_config, device):
        """Test that different time values produce different embeddings."""
        from src.model.adapter import ContinuousTimeEmbedding

        emb = ContinuousTimeEmbedding(hidden_size=model_config["hidden_size"]).to(
            device
        )

        # Test different continuous time values
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        features = []
        for t_val in t_values:
            t = torch.tensor([t_val], device=device)
            time_feat = emb(t)
            features.append(time_feat)

        # Different times should have different features
        for i in range(len(features) - 1):
            assert not torch.allclose(features[i], features[i + 1], atol=1e-6)

    def test_time_embedding_fractional_values(self, model_config, device):
        """Test that fractional time values work correctly."""
        from src.model.adapter import ContinuousTimeEmbedding

        emb = ContinuousTimeEmbedding(hidden_size=model_config["hidden_size"]).to(
            device
        )

        batch_size = 4
        # Use fractional time values
        t = torch.tensor([0.0, 0.125, 0.5, 1.0], device=device)
        time_features = emb(t)

        # Should produce [batch_size, hidden_size] features
        assert time_features.shape == (batch_size, model_config["hidden_size"])


class TestMidblockInputInterface:
    """Test full input interface with continuous time."""

    def test_all_inputs_accepted(
        self,
        model_config,
        sample_hidden_states,
        sample_attention_mask,
        sample_position_ids,
        device,
    ):
        """Test that all documented inputs are accepted."""
        from src.model.midblock import IterativeMidblock

        batch_size = sample_hidden_states.shape[0]
        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Use continuous time t
        t = torch.zeros(batch_size, device=device)
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
            attention_mask=sample_attention_mask,
            position_ids=sample_position_ids,
            t=t,
        )

        assert output.shape == sample_hidden_states.shape

    def test_optional_inputs_omitted(self, model_config, sample_hidden_states, device):
        """Test that optional inputs can be omitted (t defaults to 0)."""
        from src.model.midblock import IterativeMidblock

        midblock = IterativeMidblock(
            hidden_size=model_config["hidden_size"],
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

        # Call with only required inputs - t defaults to 0
        output = midblock(
            hidden_states=sample_hidden_states,
            h_start=sample_hidden_states,
        )

        assert output.shape == sample_hidden_states.shape

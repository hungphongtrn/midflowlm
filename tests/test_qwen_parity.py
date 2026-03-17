"""
Tests for Qwen parity inspection module.

These tests verify that we can correctly extract:
- h_start (hidden state before replacement span)
- span states (hidden states within replacement span)
- final logits from the teacher model

And that a bypass wrapper can reproduce teacher outputs within tolerance.
"""

import pytest
import torch
import yaml
from pathlib import Path


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
def sample_input(config, device):
    """Create sample input for testing."""
    batch_size = 2
    seq_len = config["data"]["seq_len"]
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
    }


class TestQwenParityImports:
    """Test that qwen_parity module can be imported."""

    def test_import_qwen_parity(self):
        """Test that src.model.qwen_parity exists and can be imported."""
        from src.model import qwen_parity
        assert qwen_parity is not None

    def test_import_qwen_inspector(self):
        """Test that QwenInspector class exists."""
        from src.model.qwen_parity import QwenInspector
        assert QwenInspector is not None

    def test_import_bypass_wrapper(self):
        """Test that BypassWrapper class exists."""
        from src.model.qwen_parity import BypassWrapper
        assert BypassWrapper is not None


class TestQwenInspector:
    """Test QwenInspector functionality."""

    def test_inspector_initialization(self, config, device):
        """Test that QwenInspector initializes with correct span config."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        start_layer = config["replacement_model"]["start_layer"]
        end_layer = config["replacement_model"]["end_layer"]

        inspector = QwenInspector(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

        assert inspector.start_layer == start_layer
        assert inspector.end_layer == end_layer
        assert inspector.span_depth == end_layer - start_layer + 1

    def test_inspector_default_span_8_11(self, config, device):
        """Test that default span is layers 8-11 as per config."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]

        inspector = QwenInspector(
            model_name=model_name,
            device=device,
        )

        # Default should match config
        assert inspector.start_layer == 8
        assert inspector.end_layer == 11
        assert inspector.span_depth == 4

    def test_extract_embeddings(self, config, sample_input, device):
        """Test extracting embeddings output."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        inspector = QwenInspector(model_name=model_name, device=device)

        embeddings = inspector.extract_embeddings(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Embeddings should have shape [batch, seq, hidden_dim]
        assert embeddings.ndim == 3
        assert embeddings.shape[0] == sample_input["input_ids"].shape[0]
        assert embeddings.shape[1] == sample_input["input_ids"].shape[1]
        assert embeddings.shape[2] > 0  # hidden dimension

    def test_extract_h_start(self, config, sample_input, device):
        """Test extracting h_start (hidden state before replacement span)."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        start_layer = config["replacement_model"]["start_layer"]
        inspector = QwenInspector(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=config["replacement_model"]["end_layer"],
            device=device,
        )

        h_start = inspector.extract_h_start(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # h_start should have shape [batch, seq, hidden_dim]
        assert h_start.ndim == 3
        assert h_start.shape[0] == sample_input["input_ids"].shape[0]
        assert h_start.shape[1] == sample_input["input_ids"].shape[1]

    def test_extract_span_states(self, config, sample_input, device):
        """Test extracting hidden states for each layer inside replacement span."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        start_layer = config["replacement_model"]["start_layer"]
        end_layer = config["replacement_model"]["end_layer"]
        inspector = QwenInspector(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

        span_states = inspector.extract_span_states(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Should return a list of hidden states for each layer in span
        expected_depth = end_layer - start_layer + 1
        assert len(span_states) == expected_depth

        # Each state should have shape [batch, seq, hidden_dim]
        for state in span_states:
            assert state.ndim == 3
            assert state.shape[0] == sample_input["input_ids"].shape[0]
            assert state.shape[1] == sample_input["input_ids"].shape[1]

    def test_extract_final_logits(self, config, sample_input, device):
        """Test extracting final logits from teacher model."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        inspector = QwenInspector(model_name=model_name, device=device)

        logits = inspector.extract_final_logits(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Logits should have shape [batch, seq, vocab_size]
        assert logits.ndim == 3
        assert logits.shape[0] == sample_input["input_ids"].shape[0]
        assert logits.shape[1] == sample_input["input_ids"].shape[1]
        assert logits.shape[2] > 0  # vocab size

    def test_extract_all(self, config, sample_input, device):
        """Test extracting all teacher outputs at once."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        start_layer = config["replacement_model"]["start_layer"]
        end_layer = config["replacement_model"]["end_layer"]
        inspector = QwenInspector(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

        outputs = inspector.extract_all(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Should have all expected keys
        assert "embeddings" in outputs
        assert "h_start" in outputs
        assert "span_states" in outputs
        assert "h_target" in outputs  # h_end, state after last span layer
        assert "logits" in outputs

        # Verify span_states count
        expected_depth = end_layer - start_layer + 1
        assert len(outputs["span_states"]) == expected_depth


class TestConfigurableSpan:
    """Test that span extraction is configurable."""

    def test_custom_start_end_layer(self, config, sample_input, device):
        """Test that different start/end layers work correctly."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]

        # Test with different span
        inspector = QwenInspector(
            model_name=model_name,
            start_layer=4,
            end_layer=7,
            device=device,
        )

        assert inspector.start_layer == 4
        assert inspector.end_layer == 7
        assert inspector.span_depth == 4

        outputs = inspector.extract_all(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        assert len(outputs["span_states"]) == 4


class TestBypassWrapper:
    """Test bypass/no-op wrapper for exact teacher comparison."""

    def test_bypass_wrapper_initialization(self, config, device):
        """Test that BypassWrapper initializes correctly."""
        from src.model.qwen_parity import BypassWrapper

        model_name = config["model"]["name"]
        wrapper = BypassWrapper(
            model_name=model_name,
            device=device,
        )

        assert wrapper is not None
        assert wrapper.model is not None

    def test_bypass_reproduces_teacher_logits(self, config, sample_input, device):
        """Test that bypass wrapper reproduces teacher logits within tolerance."""
        from src.model.qwen_parity import QwenInspector, BypassWrapper

        model_name = config["model"]["name"]

        # Get teacher logits directly
        inspector = QwenInspector(model_name=model_name, device=device)
        teacher_logits = inspector.extract_final_logits(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Get logits through bypass wrapper
        wrapper = BypassWrapper(model_name=model_name, device=device)
        wrapper_logits = wrapper.forward(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        # Should be very close (numerical tolerance)
        tolerance = 1e-4
        max_diff = torch.max(torch.abs(teacher_logits - wrapper_logits)).item()
        assert max_diff < tolerance, f"Max diff {max_diff} exceeds tolerance {tolerance}"

    def test_bypass_with_span_config(self, config, sample_input, device):
        """Test bypass wrapper respects span configuration."""
        from src.model.qwen_parity import BypassWrapper

        model_name = config["model"]["name"]
        start_layer = config["replacement_model"]["start_layer"]
        end_layer = config["replacement_model"]["end_layer"]

        wrapper = BypassWrapper(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

        outputs = wrapper.forward_with_hidden_states(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )

        assert "h_start" in outputs
        assert "span_states" in outputs
        assert "h_target" in outputs
        assert "logits" in outputs


class TestParameterFreezing:
    """Test frozen/trainable parameter counts."""

    def test_all_parameters_frozen_by_default(self, config, device):
        """Test that all Qwen parameters are frozen by default."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        inspector = QwenInspector(model_name=model_name, device=device)

        trainable = sum(p.numel() for p in inspector.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in inspector.model.parameters())

        # All parameters should be frozen by default
        assert trainable == 0
        assert total > 0

    def test_parameter_counts_match_expectations(self, config, device):
        """Test that parameter counts match Qwen3.5-0.8B expectations."""
        from src.model.qwen_parity import QwenInspector

        model_name = config["model"]["name"]
        inspector = QwenInspector(model_name=model_name, device=device)

        total = sum(p.numel() for p in inspector.model.parameters())

        # Qwen3.5-0.8B should have approximately 0.8B parameters
        # Allow some tolerance for embedding/vocab size variations
        assert 700_000_000 < total < 900_000_000, f"Total params {total} not in expected range"


class TestFlowBlockPlaceholder:
    """Test that flow_block.py is now a placeholder."""

    def test_flow_block_is_placeholder(self):
        """Test that flow_block.py exists but is deprecated/placeholder."""
        from src.model import flow_block

        # Should exist but should indicate it's deprecated
        assert hasattr(flow_block, "__doc__")
        assert "deprecated" in flow_block.__doc__.lower() or "placeholder" in flow_block.__doc__.lower()

    def test_flow_block_no_custom_rmsnorm(self):
        """Test that flow_block doesn't contain custom RMSNorm."""
        from src.model import flow_block

        # Should not have custom RMSNorm implementation
        assert not hasattr(flow_block, "RMSNorm")

    def test_flow_block_no_custom_swiglu(self):
        """Test that flow_block doesn't contain custom SwiGLU."""
        from src.model import flow_block

        # Should not have custom SwiGLU implementation
        assert not hasattr(flow_block, "SwiGLU")

    def test_flow_block_no_custom_gqa(self):
        """Test that flow_block doesn't contain custom GQA."""
        from src.model import flow_block

        # Should not have custom GQA implementation
        assert not hasattr(flow_block, "GroupedQueryAttention")

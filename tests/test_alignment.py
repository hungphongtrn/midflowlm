"""Tests for trajectory alignment policy module.

These tests verify that we can:
1. Align trajectories exactly when T = depth(span)
2. Compress trajectories when T < depth(span)
3. Expand/interpolate trajectories when T > depth(span)
4. Fail fast when required trajectory targets are missing
"""

import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import MagicMock


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
def sample_trajectory_targets(device):
    """Create sample trajectory targets (teacher hidden states for layers 8-11).
    
    Shape: [batch_size, seq_len, span_depth, hidden_dim]
    span_depth = 4 (layers 8, 9, 10, 11)
    """
    batch_size, seq_len, span_depth, hidden_dim = 2, 128, 4, 1024
    return torch.randn(batch_size, seq_len, span_depth, hidden_dim, device=device)


@pytest.fixture
def alignment_config():
    """Sample alignment configuration."""
    return {
        "T_less_than_depth": {
            "policy": "compression",
            "description": "Compress teacher trajectory to match student steps",
            "method": "uniform_sampling",
        },
        "T_equals_depth": {
            "policy": "exact",
            "description": "Exact layerwise trajectory matching",
            "method": "one_to_one_mapping",
        },
        "T_greater_than_depth": {
            "policy": "expansion",
            "description": "Expand/interpolate teacher trajectory",
            "method": "linear_interpolation",
        },
    }


class TestAlignmentImports:
    """Test that alignment module can be imported."""

    def test_import_alignment(self):
        """Test that src.training.alignment exists and can be imported."""
        from src.training import alignment
        assert alignment is not None

    def test_import_trajectory_aligner(self):
        """Test that TrajectoryAligner class exists."""
        from src.training.alignment import TrajectoryAligner
        assert TrajectoryAligner is not None

    def test_import_alignment_policy(self):
        """Test that AlignmentPolicy enum exists."""
        from src.training.alignment import AlignmentPolicy
        assert AlignmentPolicy is not None


class TestExactAlignment:
    """Test exact alignment when T = depth(span)."""

    def test_exact_alignment_mapping(self, device, sample_trajectory_targets):
        """Test one-to-one mapping when T equals span depth."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        span_depth = 4
        T = 4  # T equals depth

        aligner = TrajectoryAligner(span_depth=span_depth)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=T,
            policy=AlignmentPolicy.EXACT,
        )

        # Shape should be [batch_size, seq_len, T, hidden_dim]
        assert aligned.shape == (2, 128, T, 1024)

        # For exact policy, output should equal input
        assert torch.allclose(aligned, sample_trajectory_targets)

    def test_exact_alignment_layer_indices(self, device, sample_trajectory_targets):
        """Test that layer indices are correctly mapped for exact alignment."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)
        mapping = aligner.get_layer_mapping(T=4, policy=AlignmentPolicy.EXACT)

        # Should be identity mapping: [0, 1, 2, 3] -> [0, 1, 2, 3]
        assert mapping == [(0, 0), (1, 1), (2, 2), (3, 3)]


class TestCompressionAlignment:
    """Test compression policy when T < depth(span)."""

    def test_compression_uniform_sampling(self, device, sample_trajectory_targets):
        """Test uniform sampling compression when T < depth."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        span_depth = 4
        T = 2  # T less than depth

        aligner = TrajectoryAligner(span_depth=span_depth)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=T,
            policy=AlignmentPolicy.COMPRESSION,
            compression_method="uniform_sampling",
        )

        # Shape should be [batch_size, seq_len, T, hidden_dim]
        assert aligned.shape == (2, 128, T, 1024)

        # For uniform sampling with T=2 and depth=4, should sample indices [0, 3]
        # (first and last layers for maximum coverage)
        expected_0 = sample_trajectory_targets[:, :, 0, :]
        expected_1 = sample_trajectory_targets[:, :, 3, :]

        assert torch.allclose(aligned[:, :, 0, :], expected_0)
        assert torch.allclose(aligned[:, :, 1, :], expected_1)

    def test_compression_layer_indices(self):
        """Test layer index mapping for compression."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)

        # T=2, depth=4: uniform sampling should pick [0, 3] (first and last)
        mapping = aligner.get_layer_mapping(
            T=2, policy=AlignmentPolicy.COMPRESSION, compression_method="uniform_sampling"
        )
        assert mapping == [(0, 0), (1, 3)]

        # T=1, depth=4: should pick middle or first
        mapping = aligner.get_layer_mapping(
            T=1, policy=AlignmentPolicy.COMPRESSION, compression_method="uniform_sampling"
        )
        assert len(mapping) == 1

    def test_compression_weighted_sampling(self, device, sample_trajectory_targets):
        """Test weighted sampling compression method."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=2,
            policy=AlignmentPolicy.COMPRESSION,
            compression_method="weighted_sampling",
            weights=[0.4, 0.3, 0.2, 0.1],
        )

        assert aligned.shape == (2, 128, 2, 1024)


class TestExpansionAlignment:
    """Test expansion/interpolation policy when T > depth(span)."""

    def test_expansion_linear_interpolation(self, device, sample_trajectory_targets):
        """Test linear interpolation when T > depth."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        span_depth = 4
        T = 8  # T greater than depth

        aligner = TrajectoryAligner(span_depth=span_depth)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=T,
            policy=AlignmentPolicy.EXPANSION,
            expansion_method="linear_interpolation",
        )

        # Shape should be [batch_size, seq_len, T, hidden_dim]
        assert aligned.shape == (2, 128, T, 1024)

        # First and last should match original trajectory
        assert torch.allclose(aligned[:, :, 0, :], sample_trajectory_targets[:, :, 0, :])
        assert torch.allclose(aligned[:, :, -1, :], sample_trajectory_targets[:, :, -1, :])

    def test_expansion_layer_mapping(self):
        """Test layer mapping documentation for expansion."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)
        mapping = aligner.get_layer_mapping(
            T=8, policy=AlignmentPolicy.EXPANSION, expansion_method="linear_interpolation"
        )

        # Should have 8 entries (one per T step)
        assert len(mapping) == 8

        # Check interpolation ratios
        # Index 0 -> layer 0, ratio 0.0
        # Index 7 -> layer 3, ratio 1.0
        assert mapping[0] == (0, 0, 0.0)
        assert mapping[-1] == (7, 3, 1.0)


class TestFailFastBehavior:
    """Test fail-fast behavior when trajectory targets are missing."""

    def test_missing_trajectory_targets(self):
        """Test that alignment fails fast when trajectory targets are None."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)

        with pytest.raises(ValueError, match="trajectory_targets.*required"):
            aligner.align_targets(
                trajectory_targets=None,
                T=4,
                policy=AlignmentPolicy.EXACT,
            )

    def test_empty_trajectory_targets(self, device):
        """Test that alignment fails fast when trajectory targets are empty."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)

        with pytest.raises(ValueError, match="trajectory_targets.*empty"):
            aligner.align_targets(
                trajectory_targets=torch.tensor([]),
                T=4,
                policy=AlignmentPolicy.EXACT,
            )

    def test_invalid_T_value(self, sample_trajectory_targets):
        """Test that alignment fails fast for invalid T values."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)

        with pytest.raises(ValueError, match="T.*positive"):
            aligner.align_targets(
                trajectory_targets=sample_trajectory_targets,
                T=0,
                policy=AlignmentPolicy.EXACT,
            )

        with pytest.raises(ValueError, match="T.*positive"):
            aligner.align_targets(
                trajectory_targets=sample_trajectory_targets,
                T=-1,
                policy=AlignmentPolicy.EXACT,
            )

    def test_mismatched_depth(self, device):
        """Test that alignment fails fast when trajectory depth doesn't match span_depth."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)

        # Create trajectory with wrong depth (5 instead of 4)
        wrong_trajectory = torch.randn(2, 128, 5, 1024, device=device)

        with pytest.raises(ValueError, match="depth.*mismatch"):
            aligner.align_targets(
                trajectory_targets=wrong_trajectory,
                T=4,
                policy=AlignmentPolicy.EXACT,
            )


class TestAlignmentFromConfig:
    """Test loading alignment configuration from config file."""

    def test_load_alignment_from_config(self, config):
        """Test that alignment settings are loaded from config."""
        from src.training.alignment import TrajectoryAligner

        replacement_config = config["replacement_model"]
        trajectory_config = replacement_config["trajectory_alignment"]

        aligner = TrajectoryAligner.from_config(
            span_depth=replacement_config["depth"],
            config=trajectory_config,
        )

        assert aligner.span_depth == 4
        assert aligner.config == trajectory_config

    def test_auto_select_policy_from_config(self, config, sample_trajectory_targets):
        """Test automatic policy selection based on T and config."""
        from src.training.alignment import TrajectoryAligner

        replacement_config = config["replacement_model"]
        trajectory_config = replacement_config["trajectory_alignment"]

        aligner = TrajectoryAligner.from_config(
            span_depth=replacement_config["depth"],
            config=trajectory_config,
        )

        # T=4 (equals depth) should use exact policy
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=4,
        )
        assert aligned.shape == (2, 128, 4, 1024)

        # T=2 (less than depth) should use compression
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=2,
        )
        assert aligned.shape == (2, 128, 2, 1024)

        # T=8 (greater than depth) should use expansion
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=8,
        )
        assert aligned.shape == (2, 128, 8, 1024)


class TestAlignmentOutputFormat:
    """Test that aligned targets are in format consumable by loss module."""

    def test_output_tensor_format(self, sample_trajectory_targets):
        """Test that output is a contiguous tensor suitable for loss computation."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=4,
            policy=AlignmentPolicy.EXACT,
        )

        assert isinstance(aligned, torch.Tensor)
        assert aligned.is_contiguous()
        assert aligned.dtype == sample_trajectory_targets.dtype
        assert aligned.device == sample_trajectory_targets.device

    def test_output_for_loss_consumption(self, sample_trajectory_targets):
        """Test output dimensions work with typical loss functions."""
        from src.training.alignment import TrajectoryAligner, AlignmentPolicy

        aligner = TrajectoryAligner(span_depth=4)
        aligned = aligner.align_targets(
            trajectory_targets=sample_trajectory_targets,
            T=2,
            policy=AlignmentPolicy.COMPRESSION,
        )

        batch_size, seq_len, T, hidden_dim = aligned.shape

        # Should be able to reshape for trajectory loss: [batch*seq, T, hidden]
        loss_input = aligned.reshape(batch_size * seq_len, T, hidden_dim)
        assert loss_input.shape == (batch_size * seq_len, T, hidden_dim)

        # Or for per-step loss: [batch*seq*T, hidden]
        loss_input_flat = aligned.reshape(-1, hidden_dim)
        assert loss_input_flat.shape == (batch_size * seq_len * T, hidden_dim)

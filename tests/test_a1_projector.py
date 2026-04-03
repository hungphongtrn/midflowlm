import pytest
import torch
from src.model.student_families import OneShotProjector


def test_one_shot_projector_forward():
    """Test A1 one-shot projector produces valid hidden states."""
    projector = OneShotProjector(
        hidden_size=896,
        mlp_ratio=4.0,
    )

    h_start = torch.randn(2, 64, 896)
    h_end_hat = projector(h_start)

    assert h_end_hat.shape == h_start.shape
    # A1 is residual MLP: h_start + g_theta(h_start)
    # Output should be different from input
    assert not torch.allclose(h_end_hat, h_start)


def test_one_shot_projector_is_residual_mlp():
    """Verify A1 implements residual projection per v0.1 spec."""
    from torch import nn

    projector = OneShotProjector(hidden_size=896)

    # Should have simple MLP layers, not transformer blocks
    has_transformer = any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in projector.modules()
    )
    assert not has_transformer, "A1 should be simple MLP, not transformer"

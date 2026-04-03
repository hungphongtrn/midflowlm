import pytest
import torch
from src.model.student_families import SharedRecurrentResidual


def test_shared_recurrent_variable_t():
    """Test A2 shared recurrent block works with different T values."""
    model = SharedRecurrentResidual(
        hidden_size=896,
        max_steps_T=8,
        mlp_ratio=4.0,
    )

    h_start = torch.randn(2, 64, 896)

    # Test T=1
    h_t1 = model(h_start, num_steps=1)
    assert h_t1.shape == h_start.shape

    # Test T=4
    h_t4 = model(h_start, num_steps=4)
    assert h_t4.shape == h_start.shape

    # With more steps, output should be different
    assert not torch.allclose(h_t1, h_t4)


def test_shared_recurrent_uses_same_block():
    """Verify A2 reuses same parameters across steps."""
    from torch import nn

    model = SharedRecurrentResidual(hidden_size=896)

    # Should have single transformer block, not T separate blocks
    transformer_blocks = [
        m
        for m in model.modules()
        if isinstance(m, (nn.TransformerEncoderLayer, nn.MultiheadAttention))
    ]
    # One or few blocks, not T separate blocks
    assert len(transformer_blocks) <= 2

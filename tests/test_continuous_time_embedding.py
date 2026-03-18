"""Tests for ContinuousTimeEmbedding.

These tests verify:
1. Continuous time embedding accepts fractional t values
2. Rejection of integer step API
3. Proper shape handling for various input formats
4. Differentiability
"""

import pytest
import torch
import torch.nn as nn


def test_continuous_time_embedding_accepts_fractional_t():
    """Test that fractional time values are accepted."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor([0.0, 0.125, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (4, 64)


def test_continuous_time_embedding_rejects_integer_step_api():
    """Test that integer step_id API is rejected."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    with pytest.raises(TypeError):
        emb(step_id=3)


def test_continuous_time_embedding_scalar_input():
    """Test that scalar t values work."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor(0.5)
    out = emb(t)
    assert out.shape == (1, 64)


def test_continuous_time_embedding_single_batch():
    """Test that single batch item works."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor([0.5])
    out = emb(t)
    assert out.shape == (1, 64)


def test_continuous_time_embedding_differentiable():
    """Test that embeddings are differentiable."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    out = emb(t)
    loss = out.sum()
    loss.backward()
    assert t.grad is not None
    assert not torch.all(t.grad == 0)


def test_continuous_time_embedding_output_values_in_range():
    """Test that output values are bounded."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.linspace(0, 1, 100)
    out = emb(t)
    # Sinusoidal embeddings should be bounded
    assert torch.all(torch.abs(out) <= 2.0)  # Allow some margin


def test_continuous_time_embedding_different_times_produce_different_outputs():
    """Test that different time values produce different embeddings."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    out1 = emb(t1)
    out2 = emb(t2)
    assert not torch.allclose(out1, out2)


def test_continuous_time_embedding_device_handling():
    """Test that embeddings work on different devices."""
    from src.model.adapter import ContinuousTimeEmbedding

    emb = ContinuousTimeEmbedding(hidden_size=64)
    t = torch.tensor([0.0, 0.5, 1.0])
    out = emb(t)
    assert out.device == t.device


def test_continuous_time_embedding_hidden_sizes():
    """Test that various hidden sizes work."""
    from src.model.adapter import ContinuousTimeEmbedding

    for hidden_size in [32, 64, 128, 256, 512]:
        emb = ContinuousTimeEmbedding(hidden_size=hidden_size)
        t = torch.tensor([0.5])
        out = emb(t)
        assert out.shape == (1, hidden_size)

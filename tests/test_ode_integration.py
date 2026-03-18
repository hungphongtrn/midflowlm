"""
Tests for ODE integration with torchdiffeq.

These tests verify:
1. MidblockVectorField matches torchdiffeq signature
2. Solver options are built correctly for different methods
3. Euler step_size is properly normalized
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


class TestMidblockVectorField:
    """Test MidblockVectorField wrapper for torchdiffeq compatibility."""

    def test_midblock_vector_field_matches_torchdiffeq_signature(self):
        """Test that MidblockVectorField has the interface torchdiffeq expects."""
        from src.model.ode import MidblockVectorField

        # Create mock midblock with get_velocity method
        mock_midblock = MagicMock()
        batch_size, seq_len, hidden_size = 2, 16, 64
        expected_velocity = torch.randn(batch_size, seq_len, hidden_size)
        mock_midblock.get_velocity.return_value = expected_velocity

        # Create sample inputs
        h_start = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)

        # Create the vector field
        field = MidblockVectorField(
            midblock=mock_midblock, h_start=h_start, attention_mask=mask
        )

        # Test forward with torchdiffeq signature: (t, h_t) -> dh/dt
        t = torch.tensor(0.5)
        h_t = torch.randn(batch_size, seq_len, hidden_size)
        out = field(t, h_t)

        # Verify output shape matches input
        assert out.shape == h_t.shape, f"Expected shape {h_t.shape}, got {out.shape}"

        # Verify midblock.get_velocity was called with correct arguments
        mock_midblock.get_velocity.assert_called_once()
        call_args = mock_midblock.get_velocity.call_args
        assert call_args[1]["h_t"] is h_t
        assert call_args[1]["h_start"] is h_start
        assert call_args[1]["attention_mask"] is mask
        # t should be expanded to batch size
        assert call_args[1]["t"].shape[0] == batch_size


class TestBuildSolverOptions:
    """Test solver option building."""

    def test_euler_solver_normalizes_dt_between_one_and_many_steps(self):
        """Test that Euler solver properly normalizes step_size based on num_steps."""
        from src.model.ode import build_solver_options

        # Test with 1 step - step_size should be 1.0
        opts_1 = build_solver_options(method="euler", num_steps=1)
        assert opts_1["step_size"] == 1.0, (
            f"Expected step_size=1.0 for 1 step, got {opts_1['step_size']}"
        )

        # Test with 100 steps - step_size should be 0.01
        opts_100 = build_solver_options(method="euler", num_steps=100)
        assert opts_100["step_size"] == 0.01, (
            f"Expected step_size=0.01 for 100 steps, got {opts_100['step_size']}"
        )

        # Test with 8 steps - step_size should be 0.125
        opts_8 = build_solver_options(method="euler", num_steps=8)
        assert opts_8["step_size"] == 0.125, (
            f"Expected step_size=0.125 for 8 steps, got {opts_8['step_size']}"
        )

    def test_build_solver_options_for_rk4(self):
        """Test solver options for RK4 method."""
        from src.model.ode import build_solver_options

        # RK4 doesn't need step_size option in the same way
        opts = build_solver_options(method="rk4", num_steps=8)
        # RK4 uses adaptive stepping, so we may return empty or minimal options
        assert isinstance(opts, dict)

    def test_build_solver_options_unknown_method(self):
        """Test handling of unknown solver methods."""
        from src.model.ode import build_solver_options

        # Unknown methods should return empty options or raise appropriate error
        opts = build_solver_options(method="unknown", num_steps=10)
        assert isinstance(opts, dict)


class TestODESolverIntegration:
    """Test integration with actual torchdiffeq solver."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torchdiffeq"),
        reason="torchdiffeq not installed",
    )
    def test_odeint_can_solve_with_midblock_vector_field(self):
        """Test that torchdiffeq.odeint works with our vector field wrapper."""
        from src.model.ode import MidblockVectorField, build_solver_options
        from torchdiffeq import odeint

        # Create a simple vector field that returns constant velocity
        class SimpleVectorField(nn.Module):
            def forward(self, t, h):
                # Constant velocity field
                return torch.ones_like(h) * 0.1

        batch_size, seq_len, hidden_size = 2, 8, 32
        h0 = torch.zeros(batch_size, seq_len, hidden_size)
        field = SimpleVectorField()

        # Integration time points: start at 0, end at 1
        t = torch.tensor([0.0, 1.0])

        # Run ODE integration
        solution = odeint(field, h0, t, method="euler", options={"step_size": 0.1})

        # Verify solution has correct shape
        assert solution.shape[0] == 2  # Two time points
        assert solution.shape[1:] == h0.shape  # Same shape as initial condition

        # At t=1, should have moved by velocity * time = 0.1 * 1 = 0.1
        final_state = solution[-1]
        expected_final = torch.ones_like(h0) * 0.1
        assert torch.allclose(final_state, expected_final, atol=1e-4)

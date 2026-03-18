"""
ODE integration module for flow-based student model.

This module provides wrappers and utilities for integrating torchdiffeq
with the FlowMidblock for continuous-time inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class MidblockVectorField(nn.Module):
    """Vector field wrapper for FlowMidblock compatible with torchdiffeq.odeint.

    This class wraps the FlowMidblock's get_velocity method into the interface
    that torchdiffeq expects: forward(t, h_t) -> dh/dt

    Args:
        midblock: The FlowMidblock instance that provides get_velocity
        h_start: Starting hidden states [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
    """

    def __init__(
        self,
        midblock: nn.Module,
        h_start: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.midblock = midblock
        self.h_start = h_start
        self.attention_mask = attention_mask

    def forward(self, t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """Compute velocity field at time t and state h_t.

        This method matches the signature expected by torchdiffeq.odeint:
        - t: scalar time value or tensor (torchdiffeq will pass a scalar tensor)
        - h_t: current hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Velocity tensor [batch_size, seq_len, hidden_size]
        """
        batch_size = h_t.shape[0]

        # Expand t to batch size - midblock expects [batch] shaped tensor
        # t is typically a scalar tensor from torchdiffeq
        if isinstance(t, torch.Tensor):
            t_batch = torch.full(
                (batch_size,), float(t.item()), device=h_t.device, dtype=h_t.dtype
            )
        else:
            t_batch = torch.full(
                (batch_size,), float(t), device=h_t.device, dtype=h_t.dtype
            )

        # Call the midblock's get_velocity method
        return self.midblock.get_velocity(
            h_t=h_t,
            h_start=self.h_start,
            attention_mask=self.attention_mask,
            t=t_batch,
        )


def build_solver_options(method: str, num_steps: int) -> Dict[str, Any]:
    """Build solver options dictionary for torchdiffeq.odeint.

    Constructs appropriate options based on the solver method and desired
    number of steps. For Euler method, calculates step_size as 1.0/num_steps
    to normalize integration over [0, 1] time range.

    Args:
        method: Solver method name ("euler", "rk4", "dopri5", etc.)
        num_steps: Number of integration steps

    Returns:
        Dictionary of solver options for torchdiffeq.odeint
    """
    if method == "euler":
        # For Euler, we need explicit step_size
        # Integration over [0, 1] with num_steps -> step_size = 1.0 / num_steps
        return {"step_size": 1.0 / num_steps}
    elif method == "midpoint":
        # Midpoint method also benefits from explicit step_size
        return {"step_size": 1.0 / num_steps}
    elif method in ["rk4", "dopri5", "adaptive_heun"]:
        # Adaptive methods don't need explicit step_size
        # They adapt based on error tolerance
        return {}
    else:
        # Unknown method - return empty options
        # torchdiffeq will use default behavior
        return {}

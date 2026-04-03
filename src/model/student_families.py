"""Student model families for midflowlm v0.1 experiments.

This module implements the trainable architecture families:
- A1: One-shot residual MLP projector (baseline)
- A2: Multi-step with step conditioning (future)
- A3: Flow-matching inspired (future)

Per v0.1 spec, each family targets a different computation budget and
assumption about iterative refinement necessity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["OneShotProjector"]


class OneShotProjector(nn.Module):
    """A1: One-shot residual MLP projector.

    The simplest architecture family - a one-shot residual MLP that transforms
    h_start to h_end in a single forward pass. This serves as a baseline to verify
    if iterative refinement is actually beneficial.

    Per v0.1 spec:
        h_end_hat = h_start + g_theta(h_start)

    where g_theta is a simple two-layer MLP with GELU activation.

    Args:
        hidden_size: Dimension of hidden states
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int = 896,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio

        intermediate_size = int(hidden_size * mlp_ratio)

        # Simple two-layer MLP: hidden -> intermediate -> hidden
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

    def forward(
        self, h_start: torch.Tensor, num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass through residual MLP.

        Args:
            h_start: Starting hidden states [batch, seq, hidden]
            num_steps: Ignored (A1 is one-shot only, for API compatibility)

        Returns:
            Transformed hidden states [batch, seq, hidden]
        """
        # MLP transformation: g_theta(h_start)
        hidden = self.fc1(h_start)
        hidden = F.gelu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)  # Single dropout after transformation

        # Residual connection: h_start + g_theta(h_start)
        h_end_hat = h_start + hidden

        return h_end_hat

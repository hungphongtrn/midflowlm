"""Step conditioning adapter for iterative hidden-state refinement.

This module provides step conditioning mechanisms for the iterative midblock,
allowing the model to be aware of the current timestep and total steps.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class StepConditioningAdapter(nn.Module):
    """Adapter for step conditioning in iterative refinement.

    This adapter generates step features that encode the current step id
    and total number of steps, allowing the midblock to adjust its behavior
    based on the refinement progress.

    Supports multiple encoding modes:
    - discrete: Learned embedding for each step
    - sinusoidal: Sinusoidal position encoding (like transformer)
    - t_div_T: Normalized timestep t/T as a scalar feature
    - combined: Combination of discrete embedding and t/T

    Args:
        hidden_size: Dimension of hidden states
        max_steps_T: Maximum number of refinement steps
        encoding_mode: How to encode step information
        normalization: How to normalize step features (default: "t_div_T")
    """

    def __init__(
        self,
        hidden_size: int,
        max_steps_T: int = 8,
        encoding_mode: Literal["discrete", "sinusoidal", "t_div_T", "combined"] = "combined",
        normalization: Optional[Literal["t_div_T"]] = "t_div_T",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps_T = max_steps_T
        self.encoding_mode = encoding_mode
        self.normalization = normalization

        if encoding_mode == "discrete":
            # Learned embedding for each possible step
            self.step_embedding = nn.Embedding(max_steps_T, hidden_size)
        elif encoding_mode == "sinusoidal":
            # Precompute sinusoidal encodings
            self.register_buffer(
                "sinusoidal_encodings",
                self._create_sinusoidal_encodings(max_steps_T, hidden_size),
            )
        elif encoding_mode == "t_div_T":
            # Simple MLP to project normalized timestep
            self.timestep_proj = nn.Sequential(
                nn.Linear(1, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, hidden_size),
            )
        elif encoding_mode == "combined":
            # Combine discrete embedding with t/T projection
            self.step_embedding = nn.Embedding(max_steps_T, hidden_size // 2)
            self.timestep_proj = nn.Sequential(
                nn.Linear(1, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, hidden_size // 2),
            )
        else:
            raise ValueError(f"Unknown encoding_mode: {encoding_mode}")

    def _create_sinusoidal_encodings(self, num_steps: int, dim: int) -> torch.Tensor:
        """Create sinusoidal position encodings.

        Args:
            num_steps: Number of steps to encode
            dim: Dimension of encoding

        Returns:
            Tensor of shape [num_steps, dim]
        """
        position = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / dim)
        )

        encodings = torch.zeros(num_steps, dim)
        encodings[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            encodings[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            encodings[:, 1::2] = torch.cos(position * div_term)

        return encodings

    def get_step_features(
        self,
        step_id: int,
        num_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get step conditioning features.

        Args:
            step_id: Current step index (0-indexed)
            num_steps: Total number of steps
            batch_size: Batch size for output
            device: Target device

        Returns:
            Step features of shape [batch_size, hidden_size]
        """
        # Clamp step_id to valid range
        step_id = max(0, min(step_id, self.max_steps_T - 1))

        if self.encoding_mode == "discrete":
            # Use learned embedding
            step_tensor = torch.tensor([step_id], dtype=torch.long, device=device)
            features = self.step_embedding(step_tensor)
            features = features.expand(batch_size, -1)

        elif self.encoding_mode == "sinusoidal":
            # Use precomputed sinusoidal encodings
            features = self.sinusoidal_encodings[step_id:step_id+1].to(device)
            features = features.expand(batch_size, -1)

        elif self.encoding_mode == "t_div_T":
            # Normalize step_id to [0, 1] range
            normalized_t = float(step_id) / max(1, num_steps - 1) if num_steps > 1 else 0.0
            t_tensor = torch.tensor([[normalized_t]], dtype=torch.float32, device=device)
            t_tensor = t_tensor.expand(batch_size, 1)
            features = self.timestep_proj(t_tensor)

        elif self.encoding_mode == "combined":
            # Combine both discrete embedding and normalized timestep
            step_tensor = torch.tensor([step_id], dtype=torch.long, device=device)
            discrete_feat = self.step_embedding(step_tensor)
            discrete_feat = discrete_feat.expand(batch_size, -1)

            normalized_t = float(step_id) / max(1, num_steps - 1) if num_steps > 1 else 0.0
            t_tensor = torch.tensor([[normalized_t]], dtype=torch.float32, device=device)
            t_tensor = t_tensor.expand(batch_size, 1)
            continuous_feat = self.timestep_proj(t_tensor)

            features = torch.cat([discrete_feat, continuous_feat], dim=-1)

        return features

    def forward(
        self,
        step_id: int,
        num_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass - alias for get_step_features."""
        return self.get_step_features(step_id, num_steps, batch_size, device)


class BoundaryConditioningAdapter(nn.Module):
    """Adapter for boundary (start/end) conditioning.

    This adapter helps the midblock maintain awareness of the starting
    point h_start during iterative refinement.

    Args:
        hidden_size: Dimension of hidden states
        conditioning_mode: How to incorporate boundary information
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_mode: Literal["concat", "add", "gate"] = "concat",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_mode = conditioning_mode

        if conditioning_mode == "concat":
            # Project concatenated [h; h_start] back to hidden_size
            self.boundary_proj = nn.Linear(hidden_size * 2, hidden_size)
        elif conditioning_mode == "gate":
            # Gating mechanism
            self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.value_proj = nn.Linear(hidden_size * 2, hidden_size)
        # "add" mode requires no parameters

    def apply_boundary_conditioning(
        self,
        hidden_states: torch.Tensor,
        h_start: torch.Tensor,
    ) -> torch.Tensor:
        """Apply boundary conditioning to hidden states.

        Args:
            hidden_states: Current hidden states [batch, seq, hidden]
            h_start: Starting hidden states [batch, seq, hidden]

        Returns:
            Conditioned hidden states [batch, seq, hidden]
        """
        if self.conditioning_mode == "concat":
            # Concatenate and project
            combined = torch.cat([hidden_states, h_start], dim=-1)
            return self.boundary_proj(combined)

        elif self.conditioning_mode == "add":
            # Simple addition
            return hidden_states + h_start

        elif self.conditioning_mode == "gate":
            # Gated combination
            combined = torch.cat([hidden_states, h_start], dim=-1)
            gate = torch.sigmoid(self.gate_proj(combined))
            value = self.value_proj(combined)
            return gate * value + (1 - gate) * hidden_states

        else:
            raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_start: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass - alias for apply_boundary_conditioning."""
        return self.apply_boundary_conditioning(hidden_states, h_start)

"""Continuous-time velocity predictor for ODE-based flow matching.

This module implements the core flow midblock that replaces a span of Qwen layers.
It predicts velocity (change rate) instead of absolute deltas:
    v_theta(h_t, t) = neural_network(h_t, h_start, t)

where v_theta is the velocity field used for ODE integration.

The FlowMidblock supports:
- get_velocity(): Predict velocity field v_theta(h_t, t)
- forward(): Simple Euler integration step: h_{t+dt} = h_t + v * dt

IterativeMidblock is kept as an alias for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Literal

from src.model.adapter import ContinuousTimeEmbedding, BoundaryConditioningAdapter
from src.model.components import RMSNorm, SwiGLUMLP, CausalSelfAttention, RefinerBlock


class IterativeMidblock(nn.Module):
    """Iterative hidden-state midblock with step conditioning.

    This module replaces a span of Qwen layers (default: layers 8-11) with
    an iterative refinement process. It performs residual updates:
        h_{k+1} = h_k + delta_k

    where delta_k is computed by a refiner block that uses:
    - Causal self-attention
    - Step conditioning (t/T normalized timestep)
    - Optional boundary conditioning from h_start

    Args:
        hidden_size: Dimension of hidden states (896 for Qwen3.5-0.8B)
        max_steps_T: Maximum number of refinement steps
        start_layer: First layer of the replaced span (for metadata)
        end_layer: Last layer of the replaced span (for metadata)
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
        qkv_bias: Whether to use bias in Q/K/V projections
        use_causal_mask: Whether to use causal attention
        use_step_conditioning: Whether to use step conditioning
        use_residual: Whether to use residual updates (default: True)
        step_encoding_mode: How to encode step information
    """

    def __init__(
        self,
        hidden_size: int = 896,
        max_steps_T: int = 8,
        start_layer: int = 8,
        end_layer: int = 11,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        use_causal_mask: bool = True,
        use_step_conditioning: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps_T = max_steps_T
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.span_depth = end_layer - start_layer + 1
        self.use_causal_mask = use_causal_mask
        self.use_step_conditioning = use_step_conditioning
        self.use_residual = use_residual

        # Continuous time embedding for ODE-based flow matching
        if use_step_conditioning:
            self.time_embedding = ContinuousTimeEmbedding(hidden_size=hidden_size)
            # Project time embeddings to combine with hidden states
            self.time_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Boundary conditioning adapter
        self.boundary_adapter = BoundaryConditioningAdapter(
            hidden_size=hidden_size,
            conditioning_mode="concat",
        )

        # Core refiner block
        self.refiner = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        # Final projection for delta
        self.delta_proj = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Special initialization for residual path
        # Initialize delta_proj to near-zero for stable residual start
        delta_linear = self.delta_proj[1]
        if isinstance(delta_linear, nn.Linear):
            nn.init.normal_(delta_linear.weight, std=1e-5)
            if delta_linear.bias is not None:
                nn.init.zeros_(delta_linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_start: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for one iterative refinement step.

        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_size]
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] (1 for valid)
            position_ids: Position IDs [batch_size, seq_len]
            t: Continuous time value(s) in [0, 1], shape [batch] or scalar

        Returns:
            Refined hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Apply boundary conditioning
        conditioned = self.boundary_adapter(hidden_states, h_start)

        # Apply time conditioning if enabled
        if self.use_step_conditioning:
            if t is None:
                # Default to t=0 if not provided
                t = torch.zeros(batch_size, device=device)
            # Ensure t is on the correct device
            if isinstance(t, torch.Tensor) and t.device != device:
                t = t.to(device)
            time_features = self.time_embedding(t)
            # Expand time features to sequence length
            time_features = time_features.unsqueeze(1).expand(-1, seq_len, -1)
            # Combine with conditioned hidden states
            combined = torch.cat([conditioned, time_features], dim=-1)
            conditioned = self.time_proj(combined)

        # Apply refiner block (computes delta)
        delta = self.refiner(
            conditioned,
            attention_mask=attention_mask if self.use_causal_mask else None,
            position_ids=position_ids,
        )

        # Project to final delta
        delta = self.delta_proj(delta)

        # Apply residual update
        if self.use_residual:
            output = hidden_states + delta
        else:
            output = delta

        return output

    def iterative_refinement(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run iterative refinement for multiple steps.

        Convenience method that runs the forward pass for num_steps iterations.
        Uses uniform time steps from 0 to 1.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]

        Returns:
            Final refined hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size = h_start.shape[0]
        device = h_start.device
        h = h_start
        # Create uniform time steps from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        for step_idx in range(num_steps):
            t = torch.full((batch_size,), float(timesteps[step_idx]), device=device)
            h = self.forward(
                hidden_states=h,
                h_start=h_start,
                attention_mask=attention_mask,
                position_ids=position_ids,
                t=t,
            )
        return h

    def get_config(self) -> dict:
        """Get configuration dictionary for saving."""
        return {
            "hidden_size": self.hidden_size,
            "max_steps_T": self.max_steps_T,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "span_depth": self.span_depth,
            "use_causal_mask": self.use_causal_mask,
            "use_step_conditioning": self.use_step_conditioning,
            "use_residual": self.use_residual,
        }

    @classmethod
    def from_config(cls, config: dict) -> "IterativeMidblock":
        """Create instance from configuration dictionary."""
        return cls(
            hidden_size=config["hidden_size"],
            max_steps_T=config["max_steps_T"],
            start_layer=config["start_layer"],
            end_layer=config["end_layer"],
            use_causal_mask=config.get("use_causal_mask", True),
            use_step_conditioning=config.get("use_step_conditioning", True),
            use_residual=config.get("use_residual", True),
        )


class FlowMidblock(nn.Module):
    """Continuous-time velocity predictor for ODE-based flow matching.

    This module replaces a span of Qwen layers with a velocity field that
    predicts how fast the state should change at any given time t:
        v_theta(h_t, t) = neural_network(h_t, h_start, t)

    The velocity field is used for ODE integration to solve:
        dh/dt = v_theta(h_t, t)

    Key differences from IterativeMidblock:
    - Predicts velocity (change rate) instead of absolute deltas
    - Supports continuous time values t in [0, 1]
    - Provides get_velocity() for ODE solver compatibility
    - Provides forward() with dt parameter for manual Euler stepping

    Args:
        hidden_size: Dimension of hidden states (896 for Qwen3.5-0.8B)
        max_steps_T: Maximum number of refinement steps (for config compatibility)
        start_layer: First layer of the replaced span (for metadata)
        end_layer: Last layer of the replaced span (for metadata)
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
        qkv_bias: Whether to use bias in Q/K/V projections
        use_causal_mask: Whether to use causal attention
        use_step_conditioning: Whether to use time conditioning
    """

    def __init__(
        self,
        hidden_size: int = 896,
        max_steps_T: int = 8,
        start_layer: int = 8,
        end_layer: int = 11,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        use_causal_mask: bool = True,
        use_step_conditioning: bool = True,
        use_residual: bool = True,  # Backward compatibility, ignored
        **kwargs,  # Accept additional kwargs for backward compatibility
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps_T = max_steps_T
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.span_depth = end_layer - start_layer + 1
        self.use_causal_mask = use_causal_mask
        self.use_step_conditioning = use_step_conditioning
        self.use_residual = use_residual  # Kept for backward compatibility

        # Continuous time embedding for ODE-based flow matching
        if use_step_conditioning:
            self.time_embedding = ContinuousTimeEmbedding(hidden_size=hidden_size)
            # Project time embeddings to combine with hidden states
            self.time_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Boundary conditioning adapter
        self.boundary_adapter = BoundaryConditioningAdapter(
            hidden_size=hidden_size,
            conditioning_mode="concat",
        )

        # Core refiner block (predicts velocity)
        self.refiner = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        # Final projection for velocity prediction
        self.velocity_proj = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Special initialization for velocity path
        # Initialize velocity_proj to near-zero for stable initial velocity
        velocity_linear = self.velocity_proj[1]
        if isinstance(velocity_linear, nn.Linear):
            nn.init.normal_(velocity_linear.weight, std=1e-5)
            if velocity_linear.bias is not None:
                nn.init.zeros_(velocity_linear.bias)

    def _euler_step(
        self,
        h_t: torch.Tensor,
        h_start: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        velocity = self.get_velocity(h_t, h_start, attention_mask, t)
        return h_t + velocity * dt

    def get_velocity(
        self,
        h_t: torch.Tensor,
        h_start: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v_theta(h_t, t).

        This is the core API for ODE solvers. Given the current state h_t at
        time t, predict the velocity (rate of change) of the hidden state.

        Args:
            h_t: Current hidden states [batch_size, seq_len, hidden_size]
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] (1 for valid)
            t: Continuous time value(s) in [0, 1], shape [batch]

        Returns:
            Velocity tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = h_t.shape
        device = h_t.device

        # Apply boundary conditioning
        conditioned = self.boundary_adapter(h_t, h_start)

        # Apply time conditioning if enabled
        if self.use_step_conditioning:
            # Ensure t is on the correct device
            if isinstance(t, torch.Tensor) and t.device != device:
                t = t.to(device)
            time_features = self.time_embedding(t)
            # Expand time features to sequence length
            time_features = time_features.unsqueeze(1).expand(-1, seq_len, -1)
            # Combine with conditioned hidden states
            combined = torch.cat([conditioned, time_features], dim=-1)
            # Ensure dtype matches projection layer weights for mixed precision
            combined = combined.to(self.time_proj.weight.dtype)
            conditioned = self.time_proj(combined)

        # Apply refiner block (computes features for velocity)
        features = self.refiner(
            conditioned,
            attention_mask=attention_mask if self.use_causal_mask else None,
        )

        # Ensure dtype matches projection layer weights for mixed precision
        features = features.to(self.velocity_proj[1].weight.dtype)
        # Project to velocity prediction
        velocity = self.velocity_proj(features)

        return velocity

    def forward(
        self,
        h_t: torch.Tensor = None,
        h_start: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        t: torch.Tensor = None,
        dt: float = 1.0,
        hidden_states: torch.Tensor = None,  # Backward compatibility alias
        position_ids: Optional[torch.Tensor] = None,  # Backward compatibility, ignored
    ) -> torch.Tensor:
        """Simple Euler integration step.

        Performs one Euler step: h_{t+dt} = h_t + v_theta(h_t, t) * dt

        Args:
            h_t: Current hidden states [batch_size, seq_len, hidden_size]
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            t: Continuous time value(s) in [0, 1], shape [batch]
            dt: Time step size for Euler integration (default: 1.0)
            hidden_states: Deprecated alias for h_t (backward compatibility)
            position_ids: Deprecated, kept for backward compatibility

        Returns:
            Next hidden states [batch_size, seq_len, hidden_size]
        """
        # Handle backward compatibility
        if h_t is None and hidden_states is not None:
            h_t = hidden_states

        # Default t to zeros if not provided
        if t is None:
            batch_size = h_t.shape[0]
            device = h_t.device
            t = torch.zeros(batch_size, device=device)
        # Get velocity at current time
        velocity = self.get_velocity(h_t, h_start, attention_mask, t)

        # Euler step: h_next = h_t + v * dt
        h_next = h_t + velocity * dt

        return h_next

    def iterative_refinement(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run iterative refinement for multiple steps.

        Convenience method that runs Euler integration for num_steps iterations.
        Uses uniform time steps from 0 to 1.

        When training, each Euler step is wrapped in gradient checkpointing
        to trade compute for memory — activations from only one step are kept
        in GPU memory at a time instead of all T steps.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Final refined hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size = h_start.shape[0]
        device = h_start.device
        h = h_start

        # Create uniform time steps from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        dt = 1.0 / num_steps

        for step_idx in range(num_steps):
            t = torch.full((batch_size,), float(timesteps[step_idx]), device=device)
            if self.training:
                h = checkpoint(
                    self._euler_step,
                    h,
                    h_start,
                    attention_mask,
                    t,
                    dt,
                    use_reentrant=False,
                )
            else:
                h = self._euler_step(h, h_start, attention_mask, t, dt)

        return h

    def get_config(self) -> dict:
        """Get configuration dictionary for saving."""
        return {
            "hidden_size": self.hidden_size,
            "max_steps_T": self.max_steps_T,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "span_depth": self.span_depth,
            "use_causal_mask": self.use_causal_mask,
            "use_step_conditioning": self.use_step_conditioning,
        }

    @classmethod
    def from_config(cls, config: dict) -> "FlowMidblock":
        """Create instance from configuration dictionary."""
        return cls(
            hidden_size=config["hidden_size"],
            max_steps_T=config["max_steps_T"],
            start_layer=config["start_layer"],
            end_layer=config["end_layer"],
            use_causal_mask=config.get("use_causal_mask", True),
            use_step_conditioning=config.get("use_step_conditioning", True),
        )


# Backward compatibility alias
IterativeMidblock = FlowMidblock

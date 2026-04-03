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
import math
from typing import Optional, List

__all__ = ["OneShotProjector", "SharedRecurrentResidual"]


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


# =============================================================================
# Supporting Components for A2
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with Grouped Query Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Handle GQA
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.o_proj(attn_output)
        return self.resid_dropout(output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden = gate * up
        hidden = self.down_proj(hidden)
        return self.dropout(hidden)


class RefinerBlock(nn.Module):
    """Single refiner block with attention and MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=False,
        )
        self.norm2 = RMSNorm(hidden_size)
        intermediate_size = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SharedRecurrentResidual(nn.Module):
    """A2: Multi-step shared recurrent residual refinement.

    This architecture implements iterative latent refinement using a single
    shared transformer block applied multiple times. Unlike A1 which does
    one-shot transformation, A2 allows the model to refine its representation
    through multiple steps using the same parameters.

    Per v0.1 spec:
        h_{t+1} = h_t + f_theta(h_t)

    where f_theta is a shared transformer block (attention + SwiGLU MLP)
    applied for T steps. The block is shared across all steps, enabling
    parameter-efficient iterative refinement.

    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        max_steps_T: Maximum number of refinement steps
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 8,
        max_steps_T: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps_T = max_steps_T

        # Single shared refiner block
        self.block = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Output projection with normalization
        self.output_norm = RMSNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard normal initialization."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

    def forward(
        self,
        h_start: torch.Tensor,
        num_steps: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with iterative shared-block refinement.

        Args:
            h_start: Starting hidden states [batch, seq, hidden]
            num_steps: Number of refinement steps (default: max_steps_T)
            attention_mask: Optional attention mask [batch, seq]

        Returns:
            Refined hidden states [batch, seq, hidden]
        """
        h = h_start

        # Run shared block for num_steps iterations
        for _ in range(num_steps):
            h = self.block(h, attention_mask)

        h = self.output_norm(h)
        return h

    def forward_with_trajectory(
        self,
        h_start: torch.Tensor,
        num_steps: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Forward pass returning full trajectory for analysis.

        Args:
            h_start: Starting hidden states [batch, seq, hidden]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch, seq]

        Returns:
            List of hidden states at each step (including h_start)
        """
        trajectory = [h_start]
        h = h_start

        for _ in range(num_steps):
            h = self.block(h, attention_mask)
            trajectory.append(h)

        # Apply output norm to final state only
        trajectory[-1] = self.output_norm(trajectory[-1])

        return trajectory

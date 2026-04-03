"""Trainable student model families for distillation.

This module implements the trainable student families for Phase 1:
- A1: Single Shared Block (T=1)
- A2: Shared Recurrent Residual (iterative, no step conditioning)
- A3: FlowMidblock (with step conditioning)

All families follow the same API for easy comparison and integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


__all__ = [
    "SharedRecurrentResidual",
]


# =============================================================================
# Shared Components (mirroring baselines.py patterns)
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


# =============================================================================
# Student Family A2: Shared Recurrent Residual
# =============================================================================


class SharedRecurrentResidual(nn.Module):
    """A2: Shared recurrent residual trainable family.

    This model applies a single shared refiner block (attention + MLP) iteratively
    for T steps, without step conditioning. This tests whether simple repeated
    application of the same block is sufficient, or if minFM-inspired step
    conditioning (A3) is needed.

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

        # Single shared block (no step conditioning)
        self.block = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Output projection
        self.output_norm = RMSNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(
        self,
        h_start: torch.Tensor,
        num_steps: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with iterative refinement.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Refined hidden states [batch_size, seq_len, hidden_size]
        """
        h = h_start

        # Run the same block for num_steps iterations (no step conditioning)
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
        """Forward pass returning full trajectory.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            List of hidden states at each step (including h_start)
        """
        trajectory = [h_start]
        h = h_start

        for _ in range(num_steps):
            h = self.block(h, attention_mask)
            trajectory.append(h)

        return trajectory

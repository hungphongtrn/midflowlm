"""Iterative hidden-state midblock with step conditioning.

This module implements the core iterative refinement block that replaces
a span of Qwen layers. It performs residual updates:
    h_{k+1} = h_k + delta_k

where delta_k is computed by a transformer-style block with causal attention
and step conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import math

from src.model.adapter import StepConditioningAdapter, BoundaryConditioningAdapter


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer.

    Implements causal (left-to-right) attention suitable for language modeling.
    Uses Grouped Query Attention (GQA) pattern similar to Qwen.

    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        qkv_bias: Whether to use bias in Q/K/V projections
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with causal attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (1 for valid tokens)
            position_ids: [batch_size, seq_len] (optional position ids)

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Q: [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: [batch, seq, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle GQA: repeat K/V heads if needed
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask (for padding)
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer (as used in Qwen and other modern LMs).

    Args:
        hidden_size: Dimension of hidden states
        intermediate_size: Size of intermediate layer
        dropout: Dropout probability
    """

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
        """Forward pass with SwiGLU activation.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden = gate * up
        hidden = self.down_proj(hidden)
        return self.dropout(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used in Qwen and other modern transformers.

    Args:
        hidden_size: Dimension of hidden states
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [..., hidden_size]

        Returns:
            Normalized tensor [..., hidden_size]
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


class RefinerBlock(nn.Module):
    """Single refiner block with attention and MLP.

    Similar to a transformer decoder layer but with step conditioning.

    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
        qkv_bias: Whether to use bias in Q/K/V projections
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout,
        )

        # MLP
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
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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
        step_encoding_mode: Literal["discrete", "sinusoidal", "t_div_T", "combined"] = "combined",
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

        # Step conditioning adapter
        if use_step_conditioning:
            self.step_adapter = StepConditioningAdapter(
                hidden_size=hidden_size,
                max_steps_T=max_steps_T,
                encoding_mode=step_encoding_mode,
            )
            # Project step features to combine with hidden states
            self.step_proj = nn.Linear(hidden_size * 2, hidden_size)

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
        if hasattr(self, 'delta_proj'):
            nn.init.normal_(self.delta_proj[1].weight, std=1e-5)
            if self.delta_proj[1].bias is not None:
                nn.init.zeros_(self.delta_proj[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_start: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        step_id: int = 0,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Forward pass for one iterative refinement step.

        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_size]
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] (1 for valid)
            position_ids: Position IDs [batch_size, seq_len]
            step_id: Current step index (0-indexed)
            num_steps: Total number of steps in this refinement

        Returns:
            Refined hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Apply boundary conditioning
        conditioned = self.boundary_adapter(hidden_states, h_start)

        # Apply step conditioning if enabled
        if self.use_step_conditioning:
            step_features = self.step_adapter(
                step_id=step_id,
                num_steps=num_steps,
                batch_size=batch_size,
                device=device,
            )
            # Expand step features to sequence length
            step_features = step_features.unsqueeze(1).expand(-1, seq_len, -1)
            # Combine with conditioned hidden states
            combined = torch.cat([conditioned, step_features], dim=-1)
            conditioned = self.step_proj(combined)

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

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]

        Returns:
            Final refined hidden states [batch_size, seq_len, hidden_size]
        """
        h = h_start
        for step_id in range(num_steps):
            h = self.forward(
                hidden_states=h,
                h_start=h_start,
                attention_mask=attention_mask,
                position_ids=position_ids,
                step_id=step_id,
                num_steps=num_steps,
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

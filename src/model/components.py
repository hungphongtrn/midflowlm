import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


class SwiGLUMLP(nn.Module):
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


class CausalSelfAttention(nn.Module):
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
        position_ids: Optional[torch.Tensor] = None,
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

        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask[:, None, None, :].to(torch.bool)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class RefinerBlock(nn.Module):
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

        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
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
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

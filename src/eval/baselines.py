"""Baseline models and evaluation metrics for falsifiable experiments.

This module implements baseline models for comparison:
1. Identity baseline: h_end = h_start (no transformation)
2. T=1 shared-block baseline: Single transformation with shared parameters
3. Simple recurrent baseline: Multi-step without minFM-inspired step conditioning

Metrics include:
- Endpoint hidden-state error
- Trajectory error
- KL divergence
- Perplexity
- Latency/throughput
- Stability metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import statistics
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from src.model.components import RMSNorm, SwiGLUMLP, CausalSelfAttention, RefinerBlock


# =============================================================================
# Baseline Models
# =============================================================================

class IdentityBaseline(nn.Module):
    """Identity baseline: h_end = h_start.

    This baseline performs no transformation, returning the input unchanged.
    Useful for measuring whether iterative refinement actually helps.
    """

    def __init__(self):
        super().__init__()

    def forward(self, h_start: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """Return input unchanged.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Ignored (for API compatibility)

        Returns:
            h_start unchanged
        """
        return h_start


class T1SharedBlockBaseline(nn.Module):
    """T=1 shared-block baseline.

    Uses a single transformer-style block to transform h_start to h_end.
    This is similar to running one layer of the original model.

    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP intermediate size to hidden size
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Single refiner block
        self.block = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Final projection
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
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through single block.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Must be 1 (ignored otherwise for API compatibility)
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Transformed hidden states [batch_size, seq_len, hidden_size]
        """
        h = self.block(h_start, attention_mask)
        h = self.output_norm(h)
        return h


class SimpleRecurrentBaseline(nn.Module):
    """Simple recurrent baseline without minFM-inspired step conditioning.

    This baseline performs multiple refinement steps using the same block
    repeatedly, WITHOUT using step conditioning (t/T normalization).
    This tests whether step conditioning is actually necessary.

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

        # Flag to indicate no step conditioning
        self.use_step_conditioning = False

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


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_endpoint_error(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """Compute endpoint hidden-state error (MSE).

    Args:
        h_pred: Predicted endpoint hidden states
        h_target: Target endpoint hidden states
        reduction: "mean" or "sum"

    Returns:
        MSE error as float
    """
    with torch.no_grad():
        error = F.mse_loss(h_pred, h_target, reduction=reduction)
    return error.item()


def compute_trajectory_error(
    trajectory_pred: List[torch.Tensor],
    trajectory_target: List[torch.Tensor],
    reduction: str = "mean",
) -> float:
    """Compute trajectory error (MSE over all steps).

    Args:
        trajectory_pred: List of predicted hidden states at each step
        trajectory_target: List of target hidden states at each step
        reduction: "mean" or "sum"

    Returns:
        Average MSE error over trajectory
    """
    if len(trajectory_pred) != len(trajectory_target):
        raise ValueError("Trajectories must have same length")

    errors = []
    with torch.no_grad():
        for h_pred, h_target in zip(trajectory_pred, trajectory_target):
            error = F.mse_loss(h_pred, h_target, reduction=reduction)
            errors.append(error.item())

    return statistics.mean(errors)


def compute_kl_divergence(
    logits_pred: torch.Tensor,
    logits_target: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Compute KL divergence between predicted and target distributions.

    Args:
        logits_pred: Predicted logits [batch_size, seq_len, vocab_size]
        logits_target: Target logits [batch_size, seq_len, vocab_size]
        temperature: Temperature for softmax

    Returns:
        KL divergence (forward: pred || target)
    """
    with torch.no_grad():
        # Apply temperature
        logits_pred = logits_pred / temperature
        logits_target = logits_target / temperature

        # Compute log probabilities
        log_probs_pred = F.log_softmax(logits_pred, dim=-1)
        log_probs_target = F.log_softmax(logits_target, dim=-1)
        probs_target = torch.exp(log_probs_target)

        # KL(pred || target) = sum(pred * log(pred/target))
        kl = F.kl_div(
            log_probs_target,
            log_probs_pred,
            reduction='batchmean',
            log_target=True,
        )
    return kl.item()


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity from logits and labels.

    Args:
        logits: Logits tensor [batch_size, seq_len, vocab_size]
        labels: Label token IDs [batch_size, seq_len]
        ignore_index: Index to ignore in labels

    Returns:
        Perplexity value
    """
    with torch.no_grad():
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        perplexity = torch.exp(loss)
    return perplexity.item()


def compute_latency_metrics(
    latencies: List[float],
    batch_size: int,
    seq_len: int,
) -> Dict[str, float]:
    """Compute latency and throughput metrics.

    Args:
        latencies: List of latency values in seconds
        batch_size: Batch size used
        seq_len: Sequence length used

    Returns:
        Dictionary with metrics
    """
    if not latencies:
        return {
            "mean_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "tokens_per_second": 0.0,
        }

    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    min_latency = min(latencies)
    max_latency = max(latencies)

    # Tokens per second = (batch_size * seq_len) / mean_latency
    tokens_per_second = (batch_size * seq_len) / mean_latency if mean_latency > 0 else 0.0

    return {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "tokens_per_second": tokens_per_second,
    }


def compute_stability_metrics(
    trajectory: List[torch.Tensor],
) -> Dict[str, float]:
    """Compute hidden-state stability metrics.

    Args:
        trajectory: List of hidden states at each step

    Returns:
        Dictionary with stability metrics
    """
    if len(trajectory) < 2:
        return {
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "mean_delta_norm": 0.0,
            "max_delta_norm": 0.0,
            "mean_relative_delta": 0.0,
        }

    norms = []
    delta_norms = []
    relative_deltas = []

    with torch.no_grad():
        for i, h in enumerate(trajectory):
            # Compute norm of current state
            norm = torch.norm(h, dim=-1).mean().item()
            norms.append(norm)

            # Compute delta from previous step
            if i > 0:
                delta = h - trajectory[i - 1]
                delta_norm = torch.norm(delta, dim=-1).mean().item()
                delta_norms.append(delta_norm)

                # Relative change
                relative = delta_norm / (norm + 1e-8)
                relative_deltas.append(relative)

    return {
        "mean_norm": statistics.mean(norms),
        "std_norm": statistics.stdev(norms) if len(norms) > 1 else 0.0,
        "mean_delta_norm": statistics.mean(delta_norms),
        "max_delta_norm": max(delta_norms),
        "mean_relative_delta": statistics.mean(relative_deltas),
    }


# =============================================================================
# Metrics Report
# =============================================================================

@dataclass
class MetricsReport:
    """Comprehensive metrics report for a model evaluation.

    Attributes:
        endpoint_error: MSE between predicted and target endpoint states
        trajectory_error: Average MSE over trajectory
        kl_divergence: KL divergence between predicted and target distributions
        perplexity: Perplexity on evaluation set
        latency_ms: Mean latency in milliseconds
        tokens_per_second: Throughput metric
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        stability_metrics: Dictionary of stability metrics
        model_name: Name of the model evaluated
        num_steps_T: Number of steps used for evaluation
    """

    endpoint_error: float
    trajectory_error: Optional[float] = None
    kl_divergence: Optional[float] = None
    perplexity: Optional[float] = None
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_params: int = 0
    trainable_params: int = 0
    stability_metrics: Optional[Dict[str, float]] = None
    model_name: str = "unknown"
    num_steps_T: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Model: {self.model_name} (T={self.num_steps_T})",
            f"Parameters: {self.total_params:,} total, {self.trainable_params:,} trainable",
            f"Endpoint Error: {self.endpoint_error:.6f}",
        ]

        if self.trajectory_error is not None:
            lines.append(f"Trajectory Error: {self.trajectory_error:.6f}")

        if self.kl_divergence is not None:
            lines.append(f"KL Divergence: {self.kl_divergence:.6f}")

        if self.perplexity is not None:
            lines.append(f"Perplexity: {self.perplexity:.2f}")

        lines.extend([
            f"Latency: {self.latency_ms:.2f} ms",
            f"Throughput: {self.tokens_per_second:.2f} tokens/sec",
        ])

        if self.stability_metrics:
            lines.append("Stability Metrics:")
            for key, value in self.stability_metrics.items():
                lines.append(f"  {key}: {value:.6f}")

        return "\n".join(lines)


def get_parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Get parameter counts for a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total,
        "trainable_params": trainable,
    }

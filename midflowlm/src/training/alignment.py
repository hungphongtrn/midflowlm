"""Trajectory alignment policy module for variable T regimes.

This module defines explicit policies for aligning teacher trajectory targets
with student model steps when T (number of refinement steps) differs from
the teacher span depth.

Three regimes are supported:
1. T = depth(span): Exact layerwise matching
2. T < depth(span): Compression via uniform or weighted sampling
3. T > depth(span): Expansion via linear interpolation

The aligned targets are returned in a format directly consumable by loss modules.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
import torch.nn.functional as F


class AlignmentPolicy(Enum):
    """Enumeration of trajectory alignment policies."""

    EXACT = auto()  # One-to-one mapping when T == depth
    COMPRESSION = auto()  # Sample when T < depth
    EXPANSION = auto()  # Interpolate when T > depth


class TrajectoryAligner:
    """Aligns teacher trajectory targets with student steps.

    The aligner handles three regimes:
    - Exact: T equals span depth, returns targets as-is
    - Compression: T less than span depth, samples teacher layers
    - Expansion: T greater than span depth, interpolates between teacher layers

    Args:
        span_depth: Number of layers in the teacher span (e.g., 4 for layers 8-11)
        config: Optional configuration dict from YAML config file
    """

    def __init__(
        self,
        span_depth: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        if span_depth <= 0:
            raise ValueError(f"span_depth must be positive, got {span_depth}")

        self.span_depth = span_depth
        self.config = config or {}

    @classmethod
    def from_config(cls, span_depth: int, config: Dict[str, Any]) -> "TrajectoryAligner":
        """Create a TrajectoryAligner from configuration dict.

        Args:
            span_depth: Number of layers in the teacher span
            config: Configuration dict with trajectory_alignment settings

        Returns:
            TrajectoryAligner instance configured from config
        """
        return cls(span_depth=span_depth, config=config)

    def align_targets(
        self,
        trajectory_targets: Optional[torch.Tensor],
        T: int,
        policy: Optional[AlignmentPolicy] = None,
        compression_method: Optional[str] = None,
        expansion_method: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Align trajectory targets for the given T.

        Args:
            trajectory_targets: Teacher hidden states tensor of shape
                [batch_size, seq_len, span_depth, hidden_dim]
            T: Number of student refinement steps
            policy: Alignment policy to use. If None, auto-selects based on T vs depth.
            compression_method: Method for compression ("uniform_sampling" or "weighted_sampling")
            expansion_method: Method for expansion ("linear_interpolation")
            weights: Optional weights for weighted sampling

        Returns:
            Aligned targets of shape [batch_size, seq_len, T, hidden_dim]

        Raises:
            ValueError: If trajectory_targets is None/empty, T is invalid,
                       or depth mismatch occurs
        """
        # Fail-fast validation
        self._validate_inputs(trajectory_targets, T)

        # Auto-select policy if not specified
        if policy is None:
            policy = self._auto_select_policy(T)

        # Apply appropriate alignment policy
        if policy == AlignmentPolicy.EXACT:
            return self._apply_exact(trajectory_targets, T)
        elif policy == AlignmentPolicy.COMPRESSION:
            method = compression_method or self._get_config_method("T_less_than_depth", "uniform_sampling")
            return self._apply_compression(trajectory_targets, T, method, weights)
        elif policy == AlignmentPolicy.EXPANSION:
            method = expansion_method or self._get_config_method("T_greater_than_depth", "linear_interpolation")
            return self._apply_expansion(trajectory_targets, T, method)
        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _validate_inputs(
        self,
        trajectory_targets: Optional[torch.Tensor],
        T: int,
    ) -> None:
        """Validate inputs with fail-fast behavior."""
        if trajectory_targets is None:
            raise ValueError("trajectory_targets is required but got None")

        if trajectory_targets.numel() == 0:
            raise ValueError("trajectory_targets is empty")

        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        # Check depth dimension matches
        if trajectory_targets.dim() < 3:
            raise ValueError(
                f"trajectory_targets must have at least 3 dimensions "
                f"[batch, seq, depth, ...], got shape {trajectory_targets.shape}"
            )

        actual_depth = trajectory_targets.shape[2]
        if actual_depth != self.span_depth:
            raise ValueError(
                f"trajectory_targets depth ({actual_depth}) does not match "
                f"span_depth ({self.span_depth}) - depth mismatch"
            )

    def _auto_select_policy(self, T: int) -> AlignmentPolicy:
        """Automatically select policy based on T vs span_depth."""
        if T == self.span_depth:
            return AlignmentPolicy.EXACT
        elif T < self.span_depth:
            return AlignmentPolicy.COMPRESSION
        else:
            return AlignmentPolicy.EXPANSION

    def _get_config_method(self, regime_key: str, default: str) -> str:
        """Get method from config or return default."""
        if self.config and regime_key in self.config:
            return self.config[regime_key].get("method", default)
        return default

    def _apply_exact(
        self,
        trajectory_targets: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Apply exact one-to-one mapping.

        When T == span_depth, returns the trajectory targets unchanged.
        """
        # For exact policy, T must equal span_depth
        if T != self.span_depth:
            raise ValueError(
                f"EXACT policy requires T ({T}) to equal span_depth ({self.span_depth})"
            )

        # Return contiguous copy to ensure proper memory layout
        return trajectory_targets.contiguous()

    def _apply_compression(
        self,
        trajectory_targets: torch.Tensor,
        T: int,
        method: str,
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Apply compression to select T layers from span_depth layers.

        Methods:
        - uniform_sampling: Evenly select T layers from span_depth layers
        - weighted_sampling: Select T layers based on provided weights
        """
        batch_size, seq_len, depth, hidden_dim = trajectory_targets.shape

        if method == "uniform_sampling":
            indices = self._compute_uniform_indices(T, depth)
        elif method == "weighted_sampling":
            indices = self._compute_weighted_indices(T, depth, weights)
        else:
            raise ValueError(f"Unknown compression method: {method}")

        # Gather selected layers
        # trajectory_targets: [batch, seq, depth, hidden]
        # indices: list of layer indices to select
        aligned = trajectory_targets[:, :, indices, :]

        return aligned.contiguous()

    def _apply_expansion(
        self,
        trajectory_targets: torch.Tensor,
        T: int,
        method: str,
    ) -> torch.Tensor:
        """Apply expansion to interpolate T layers from span_depth layers.

        Methods:
        - linear_interpolation: Linearly interpolate between teacher layers
        """
        if method != "linear_interpolation":
            raise ValueError(f"Unknown expansion method: {method}")

        batch_size, seq_len, depth, hidden_dim = trajectory_targets.shape

        # Compute interpolation positions
        # Map T student steps to depth-1 intervals between teacher layers
        # Position 0 -> layer 0, position T-1 -> layer depth-1
        positions = torch.linspace(0, depth - 1, T, device=trajectory_targets.device)

        # Perform linear interpolation
        aligned = self._linear_interpolate_1d(trajectory_targets, positions)

        return aligned.contiguous()

    def _compute_uniform_indices(self, T: int, depth: int) -> List[int]:
        """Compute uniformly sampled indices for compression.

        Selects T indices from depth layers with uniform spacing.
        """
        if T == 1:
            # Single step: select middle layer
            return [depth // 2]

        # Compute uniform spacing
        # Use round to get integer indices
        indices = [round(i * (depth - 1) / (T - 1)) for i in range(T)]

        # Ensure uniqueness and bounds
        indices = sorted(set(indices))

        # If we lost some due to rounding, fill in gaps
        while len(indices) < T:
            # Add missing indices by checking gaps
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] > 1:
                    # Insert middle index
                    mid = (indices[i] + indices[i + 1]) // 2
                    indices.insert(i + 1, mid)
                    break
            else:
                # No gaps, append or prepend
                if indices[0] > 0:
                    indices.insert(0, indices[0] - 1)
                elif indices[-1] < depth - 1:
                    indices.append(indices[-1] + 1)

        return indices[:T]

    def _compute_weighted_indices(
        self,
        T: int,
        depth: int,
        weights: Optional[List[float]],
    ) -> List[int]:
        """Compute weighted sampled indices for compression.

        Selects T indices based on cumulative weight distribution.
        """
        if weights is None:
            # Fall back to uniform if no weights provided
            return self._compute_uniform_indices(T, depth)

        if len(weights) != depth:
            raise ValueError(f"weights length ({len(weights)}) must match depth ({depth})")

        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()

        # Compute cumulative distribution
        cumsum = torch.cumsum(weights, dim=0)

        # Select T points uniformly in [0, 1] and map to indices
        positions = torch.linspace(0, 1 - 1e-6, T)

        indices = []
        for pos in positions:
            # Find first index where cumsum > pos
            idx = torch.searchsorted(cumsum, pos).item()
            idx = min(idx, depth - 1)  # Bounds check
            indices.append(idx)

        # Ensure unique indices (greedy fill)
        indices = sorted(set(indices))
        while len(indices) < T:
            # Add unused indices with highest weights
            used = set(indices)
            available = [(i, weights[i].item()) for i in range(depth) if i not in used]
            available.sort(key=lambda x: -x[1])  # Sort by weight descending
            if available:
                indices.append(available[0][0])
                indices.sort()

        return indices[:T]

    def _linear_interpolate_1d(
        self,
        trajectory_targets: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly interpolate trajectory at given positions.

        Args:
            trajectory_targets: [batch, seq, depth, hidden]
            positions: [T] positions in [0, depth-1]

        Returns:
            Interpolated tensor of shape [batch, seq, T, hidden]
        """
        batch_size, seq_len, depth, hidden_dim = trajectory_targets.shape
        T = positions.shape[0]

        # Get lower and upper indices for each position
        lower_idx = torch.floor(positions).long()
        upper_idx = torch.ceil(positions).long()

        # Clamp to valid range
        lower_idx = torch.clamp(lower_idx, 0, depth - 1)
        upper_idx = torch.clamp(upper_idx, 0, depth - 1)

        # Compute interpolation weights
        # weight for upper, (1 - weight) for lower
        weights = positions - lower_idx.float()

        # Gather lower and upper values
        # trajectory_targets: [batch, seq, depth, hidden]
        lower_vals = trajectory_targets[:, :, lower_idx, :]  # [batch, seq, T, hidden]
        upper_vals = trajectory_targets[:, :, upper_idx, :]  # [batch, seq, T, hidden]

        # Interpolate
        # Expand weights for broadcasting: [T] -> [1, 1, T, 1]
        w = weights.view(1, 1, T, 1)
        interpolated = (1 - w) * lower_vals + w * upper_vals

        return interpolated

    def get_layer_mapping(
        self,
        T: int,
        policy: Optional[AlignmentPolicy] = None,
        compression_method: Optional[str] = None,
        expansion_method: Optional[str] = None,
    ) -> List[Union[Tuple[int, int], Tuple[int, int, float]]]:
        """Get the layer mapping for documentation/debugging.

        Returns a list of tuples describing how each student step maps to
        teacher layers:
        - Exact/Compression: [(student_step, teacher_layer), ...]
        - Expansion: [(student_step, teacher_layer, interpolation_ratio), ...]

        Args:
            T: Number of student steps
            policy: Alignment policy (auto-selected if None)
            compression_method: Method for compression
            expansion_method: Method for expansion

        Returns:
            List of mapping tuples
        """
        if policy is None:
            policy = self._auto_select_policy(T)

        if policy == AlignmentPolicy.EXACT:
            # One-to-one mapping
            return [(i, i) for i in range(T)]

        elif policy == AlignmentPolicy.COMPRESSION:
            method = compression_method or "uniform_sampling"
            if method == "uniform_sampling":
                indices = self._compute_uniform_indices(T, self.span_depth)
            else:
                indices = list(range(min(T, self.span_depth)))
            return [(i, indices[i]) for i in range(len(indices))]

        elif policy == AlignmentPolicy.EXPANSION:
            # Expansion with interpolation ratios
            positions = torch.linspace(0, self.span_depth - 1, T)
            mapping = []
            for i, pos in enumerate(positions.tolist()):
                lower = int(pos)
                upper = min(lower + 1, self.span_depth - 1)
                # Handle edge case where pos is exactly the last layer
                if pos >= self.span_depth - 1:
                    ratio = 1.0
                else:
                    ratio = pos - lower
                mapping.append((i, lower, ratio))
            return mapping

        else:
            raise ValueError(f"Unknown policy: {policy}")

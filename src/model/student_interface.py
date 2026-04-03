"""Unified interface for all student architecture families.

This module provides a consistent API across A1 (one-shot), A2 (shared recurrent),
and A3 (flow-matching) families, decoupling FrozenQwenStudent from family-specific
logic.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from src.model.student_families import OneShotProjector, SharedRecurrentResidual
from src.model.midblock import FlowMidblock, IterativeMidblock


class StudentFamilyInterface:
    """Unified interface for all student architecture families.

    Provides a consistent API across A1 (one-shot), A2 (shared recurrent),
    and A3 (flow-matching) families.

    Args:
        family_model: The underlying family model instance
        family_type: Type of family - "one_shot_projector", "shared_recurrent",
                     or "flow_midblock"
    """

    def __init__(self, family_model: nn.Module, family_type: str):
        self.family = family_model
        self.family_type = family_type

        # Validate family type
        valid_types = ["one_shot_projector", "shared_recurrent", "flow_midblock"]
        if family_type not in valid_types:
            raise ValueError(
                f"Invalid family_type: {family_type}. Must be one of {valid_types}"
            )

    def forward_refinement(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward refinement across all family types.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_trajectory: If True, also return trajectory of hidden states

        Returns:
            Dict with:
                - "endpoint_hidden": Final hidden states [batch_size, seq_len, hidden_size]
                - "trajectory_hidden": Optional trajectory [batch_size, seq_len, num_steps, hidden_size]
        """
        if self.family_type == "one_shot_projector":
            return self._forward_a1(
                h_start, num_steps, attention_mask, return_trajectory
            )
        elif self.family_type == "shared_recurrent":
            return self._forward_a2(
                h_start, num_steps, attention_mask, return_trajectory
            )
        elif self.family_type == "flow_midblock":
            return self._forward_a3(
                h_start, num_steps, attention_mask, return_trajectory
            )
        else:
            raise ValueError(f"Unknown family_type: {self.family_type}")

    def _forward_a1(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor],
        return_trajectory: bool,
    ) -> Dict[str, torch.Tensor]:
        """A1: One-shot projector forward.

        A1 always applies exactly one step regardless of num_steps parameter.
        """
        # A1 ignores num_steps - always one shot
        h_end = self.family(h_start, num_steps=1, attention_mask=attention_mask)

        result = {"endpoint_hidden": h_end}

        if return_trajectory:
            # For A1, trajectory is just [h_start, h_end] stacked
            # Shape: [batch, seq, 1, hidden] (only 1 actual step)
            trajectory = torch.stack([h_end], dim=2)  # [batch, seq, 1, hidden]
            result["trajectory_hidden"] = trajectory

        return result

    def _forward_a2(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor],
        return_trajectory: bool,
    ) -> Dict[str, torch.Tensor]:
        """A2: Shared recurrent residual forward."""
        if return_trajectory:
            # Use forward_with_trajectory if available
            if hasattr(self.family, "forward_with_trajectory"):
                trajectory_list = self.family.forward_with_trajectory(
                    h_start, num_steps=num_steps, attention_mask=attention_mask
                )
                # trajectory_list includes h_start at index 0
                h_end = trajectory_list[-1]
                # Stack trajectory excluding h_start: [num_steps, batch, seq, hidden]
                trajectory_stacked = torch.stack(trajectory_list[1:], dim=0)
                # Transpose to [batch, seq, num_steps, hidden]
                trajectory_stacked = trajectory_stacked.transpose(0, 1).transpose(1, 2)
                return {
                    "endpoint_hidden": h_end,
                    "trajectory_hidden": trajectory_stacked,
                }
            else:
                # Fallback: manually collect trajectory
                trajectory = []
                h = h_start
                for _ in range(num_steps):
                    h = self.family.block(h, attention_mask)
                    trajectory.append(h)
                h_end = self.family.output_norm(h)
                trajectory_stacked = torch.stack(
                    trajectory, dim=2
                )  # [batch, seq, num_steps, hidden]
                return {
                    "endpoint_hidden": h_end,
                    "trajectory_hidden": trajectory_stacked,
                }
        else:
            # Simple forward without trajectory
            h_end = self.family(
                h_start, num_steps=num_steps, attention_mask=attention_mask
            )
            return {"endpoint_hidden": h_end}

    def _forward_a3(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor],
        return_trajectory: bool,
    ) -> Dict[str, torch.Tensor]:
        """A3: Flow midblock forward with ODE integration."""
        # For A3, we use iterative_refinement which handles the flow matching
        if return_trajectory:
            # Manually collect trajectory since iterative_refinement only returns endpoint
            trajectory = []
            h = h_start
            batch_size = h_start.shape[0]
            device = h_start.device

            # Create uniform time steps
            timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]

            for step_idx in range(num_steps):
                t = torch.full((batch_size,), float(timesteps[step_idx]), device=device)
                dt = 1.0 / num_steps

                # Use forward (Euler step) or get_velocity
                if hasattr(self.family, "forward"):
                    h = self.family(
                        h_t=h,
                        h_start=h_start,
                        attention_mask=attention_mask,
                        t=t,
                        dt=dt,
                    )
                else:
                    # Fallback using get_velocity
                    velocity = self.family.get_velocity(h, h_start, attention_mask, t)
                    h = h + velocity * dt

                trajectory.append(h)

            h_end = h
            trajectory_stacked = torch.stack(
                trajectory, dim=2
            )  # [batch, seq, num_steps, hidden]
            return {
                "endpoint_hidden": h_end,
                "trajectory_hidden": trajectory_stacked,
            }
        else:
            # Use iterative_refinement for endpoint only
            if hasattr(self.family, "iterative_refinement"):
                h_end = self.family.iterative_refinement(
                    h_start, num_steps=num_steps, attention_mask=attention_mask
                )
            else:
                # Manual Euler integration
                h = h_start
                batch_size = h_start.shape[0]
                device = h_start.device
                timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]

                for step_idx in range(num_steps):
                    t = torch.full(
                        (batch_size,), float(timesteps[step_idx]), device=device
                    )
                    dt = 1.0 / num_steps
                    velocity = self.family.get_velocity(h, h_start, attention_mask, t)
                    h = h + velocity * dt

                h_end = h

            return {"endpoint_hidden": h_end}

    def forward_refinement_with_velocity(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward refinement with velocity output (for flow-based families like A3).

        This is specifically for flow-matching families that work with velocity fields.
        For non-flow families (A1, A2), this falls back to standard forward_refinement.

        Args:
            h_start: Starting hidden states [batch_size, seq_len, hidden_size]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_trajectory: If True, also return trajectory

        Returns:
            Dict with:
                - "endpoint_hidden": Final hidden states
                - "trajectory_hidden": Optional trajectory
                - "velocity": Final velocity prediction (flow families only)
        """
        if self.family_type == "flow_midblock":
            return self._forward_a3_with_velocity(
                h_start, num_steps, attention_mask, return_trajectory
            )
        else:
            # For non-flow families, just return standard result without velocity
            result = self.forward_refinement(
                h_start, num_steps, attention_mask, return_trajectory
            )
            return result

    def _forward_a3_with_velocity(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor],
        return_trajectory: bool,
    ) -> Dict[str, torch.Tensor]:
        """A3 forward with velocity output."""
        trajectory = []
        h = h_start
        batch_size = h_start.shape[0]
        device = h_start.device

        # Create uniform time steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        final_velocity = None

        for step_idx in range(num_steps):
            t = torch.full((batch_size,), float(timesteps[step_idx]), device=device)

            # Get velocity at current state
            velocity = self.family.get_velocity(h, h_start, attention_mask, t)

            if step_idx == num_steps - 1:
                final_velocity = velocity

            dt = 1.0 / num_steps
            h = h + velocity * dt

            if return_trajectory:
                trajectory.append(h)

        h_end = h
        result = {
            "endpoint_hidden": h_end,
            "velocity": final_velocity,
        }

        if return_trajectory:
            trajectory_stacked = torch.stack(trajectory, dim=2)
            result["trajectory_hidden"] = trajectory_stacked

        return result

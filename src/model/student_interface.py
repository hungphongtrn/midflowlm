"""Student family interface for unified refinement across different architectures.

This module provides a unified interface for different student family models,
allowing FrozenQwenStudent to work with A1, A2, or A3 families transparently.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union


class StudentFamilyInterface:
    """Unified interface for student family models.

    Wraps different family models (A1: OneShotProjector, A2: SharedRecurrentResidual,
    A3: FlowMidblock) and provides a consistent API for the refinement process.

    Args:
        family_model: The underlying family model (OneShotProjector, SharedRecurrentResidual, etc.)
        family_type: Type of family model ("one_shot_projector", "shared_recurrent_residual", "flow_midblock")
    """

    def __init__(
        self,
        family_model: nn.Module,
        family_type: str = "flow_midblock",
    ):
        self.family_model = family_model
        self.family_type = family_type

    def forward_refinement(
        self,
        h_start: torch.Tensor,
        num_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """Run refinement using the family model.

        Args:
            h_start: Starting hidden states [batch, seq, hidden]
            num_steps: Number of refinement steps
            attention_mask: Optional attention mask [batch, seq]
            return_trajectory: If True, return full trajectory of hidden states

        Returns:
            Dictionary with:
                - endpoint_hidden: Final hidden state after refinement
                - trajectory_hidden: Optional trajectory of hidden states [batch, seq, steps, hidden]
        """
        if self.family_type == "one_shot_projector":
            # A1: One-shot MLP projector
            h_end = self.family_model(h_start, num_steps=num_steps)
            result = {"endpoint_hidden": h_end}
            if return_trajectory:
                # For A1, trajectory is just start and end
                trajectory = torch.stack(
                    [h_start, h_end], dim=2
                )  # [batch, seq, 2, hidden]
                result["trajectory_hidden"] = trajectory[:, :, 1:]  # Skip first state
            return result

        elif self.family_type == "shared_recurrent_residual":
            # A2: Shared recurrent residual
            if return_trajectory and hasattr(
                self.family_model, "forward_with_trajectory"
            ):
                trajectory_list = self.family_model.forward_with_trajectory(
                    h_start, num_steps=num_steps, attention_mask=attention_mask
                )
                # Stack trajectory: [num_steps+1, batch, seq, hidden]
                trajectory = torch.stack(trajectory_list, dim=0)
                # Transpose to [batch, seq, num_steps+1, hidden] and skip initial state
                trajectory_stacked = trajectory.transpose(0, 1).transpose(1, 2)
                h_end = trajectory_list[-1]
                return {
                    "endpoint_hidden": h_end,
                    "trajectory_hidden": trajectory_stacked[:, :, 1:],
                }
            else:
                h_end = self.family_model(
                    h_start, num_steps=num_steps, attention_mask=attention_mask
                )
                return {"endpoint_hidden": h_end}

        elif self.family_type == "flow_midblock":
            # A3: Flow-matching with ODE integration
            # This requires special handling via ODE solver
            # The actual ODE integration happens in the caller (FrozenQwenStudent.forward)
            # This interface method is not used directly for flow_midblock
            raise NotImplementedError(
                "flow_midblock family uses direct ODE integration in FrozenQwenStudent."
                "Use the midblock directly, not through this interface."
            )

        else:
            raise ValueError(f"Unknown family type: {self.family_type}")

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters in the family model."""
        return sum(p.numel() for p in self.family_model.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Get total number of parameters in the family model."""
        return sum(p.numel() for p in self.family_model.parameters())

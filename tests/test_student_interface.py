import pytest
import torch
from src.model.student_interface import StudentFamilyInterface
from src.model.student_families import OneShotProjector, SharedRecurrentResidual


def test_a1_interface_compliance():
    """Test A1 conforms to student family interface."""
    family = OneShotProjector(hidden_size=896)
    interface = StudentFamilyInterface(family, family_type="one_shot_projector")

    h_start = torch.randn(2, 64, 896)

    # Should support unified forward
    result = interface.forward_refinement(h_start, num_steps=1)
    h_end = result["endpoint_hidden"]
    assert h_end.shape == h_start.shape


def test_a2_interface_compliance():
    """Test A2 conforms to student family interface."""
    family = SharedRecurrentResidual(hidden_size=896, max_steps_T=8)
    interface = StudentFamilyInterface(family, family_type="shared_recurrent_residual")

    h_start = torch.randn(2, 64, 896)

    # Should support unified forward with variable T
    result = interface.forward_refinement(h_start, num_steps=4)
    h_end = result["endpoint_hidden"]
    assert h_end.shape == h_start.shape


def test_interface_returns_dict_for_trajectory():
    """Test interface returns dict with endpoint_hidden and optionally trajectory_hidden."""
    family = SharedRecurrentResidual(hidden_size=896, max_steps_T=8)
    interface = StudentFamilyInterface(family, family_type="shared_recurrent_residual")

    h_start = torch.randn(2, 64, 896)

    # Without trajectory
    result = interface.forward_refinement(h_start, num_steps=4, return_trajectory=False)
    assert isinstance(result, dict)
    assert "endpoint_hidden" in result
    assert "trajectory_hidden" not in result

    # With trajectory
    result = interface.forward_refinement(h_start, num_steps=4, return_trajectory=True)
    assert "endpoint_hidden" in result
    assert "trajectory_hidden" in result

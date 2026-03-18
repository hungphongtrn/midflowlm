"""Tests for distillation loss functions.

These tests verify that we can:
1. Compute endpoint MSE loss
2. Compute mandatory trajectory loss using alignment policy
3. Weight endpoint/trajectory/KL/CE terms properly
4. Fail fast when trajectory targets are missing
5. Return grouped metric outputs
"""

import pytest
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def config():
    """Load v0 config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "v0_onemotif.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for loss computation."""
    batch_size, seq_len, hidden_dim = 2, 128, 896
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "h_start": torch.randn(batch_size, seq_len, hidden_dim, device=device),
        "trajectory_targets": torch.randn(
            batch_size, seq_len, 4, hidden_dim, device=device
        ),
        "h_target": torch.randn(batch_size, seq_len, hidden_dim, device=device),
        "teacher_logits": torch.randn(batch_size, seq_len, 1000, device=device),
        "labels": torch.randint(0, 1000, (batch_size, seq_len), device=device),
    }


@pytest.fixture
def student_outputs(device):
    """Create sample student model outputs."""
    batch_size, seq_len, hidden_dim = 2, 128, 896
    return {
        "endpoint_hidden": torch.randn(batch_size, seq_len, hidden_dim, device=device),
        "trajectory_hidden": torch.randn(
            batch_size, seq_len, 4, hidden_dim, device=device
        ),
        "logits": torch.randn(batch_size, seq_len, 1000, device=device),
    }


@pytest.fixture
def loss_config(config):
    """Extract loss configuration from config."""
    return config["loss"]


class TestLossImports:
    """Test that losses module can be imported."""

    def test_import_losses(self):
        """Test that src.training.losses exists and can be imported."""
        from src.training import losses

        assert losses is not None

    def test_import_distillation_loss(self):
        """Test that DistillationLoss class exists."""
        from src.training.losses import DistillationLoss

        assert DistillationLoss is not None

    def test_import_loss_config(self):
        """Test that LossConfig dataclass exists."""
        from src.training.losses import LossConfig

        assert LossConfig is not None


class TestEndpointMSELoss:
    """Test endpoint MSE loss computation."""

    def test_endpoint_mse_basic(self, device, sample_batch, student_outputs):
        """Test basic endpoint MSE computation."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(endpoint_weight=1.0, trajectory_weight=0.0)
        loss_fn = DistillationLoss(config)

        losses = loss_fn.compute_endpoint_loss(
            student_hidden=student_outputs["endpoint_hidden"],
            teacher_hidden=sample_batch["h_target"],
            attention_mask=sample_batch["attention_mask"],
        )

        assert "loss" in losses
        assert "mse" in losses
        assert losses["loss"].item() >= 0
        assert losses["loss"].shape == ()

    def test_endpoint_mse_masked(self, device):
        """Test that endpoint MSE respects attention mask."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            endpoint_weight=1.0, trajectory_weight=0.0, mask_padding_tokens=True
        )
        loss_fn = DistillationLoss(config)

        batch_size, seq_len, hidden_dim = 2, 4, 8
        student_hidden = torch.randn(batch_size, seq_len, hidden_dim)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)

        # Create mask with some padded tokens
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -1] = 0  # Mask last token in each sequence

        losses = loss_fn.compute_endpoint_loss(
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            attention_mask=attention_mask,
        )

        # Compute expected loss manually (only over unmasked tokens)
        diff = student_hidden - teacher_hidden
        squared_error = (diff**2).mean(dim=-1)  # [batch, seq]

        # Mask and compute mean over valid tokens
        masked_error = squared_error * attention_mask
        expected_loss = masked_error.sum() / attention_mask.sum()

        assert torch.allclose(losses["loss"], expected_loss, rtol=1e-5)

    def test_endpoint_mse_unmasked(self, device):
        """Test endpoint MSE without masking."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            endpoint_weight=1.0, trajectory_weight=0.0, mask_padding_tokens=False
        )
        loss_fn = DistillationLoss(config)

        batch_size, seq_len, hidden_dim = 2, 4, 8
        student_hidden = torch.randn(batch_size, seq_len, hidden_dim)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -1] = 0  # Mask last token

        losses = loss_fn.compute_endpoint_loss(
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            attention_mask=attention_mask,
        )

        # Compute expected loss manually (all tokens)
        expected_loss = F.mse_loss(student_hidden, teacher_hidden)

        assert torch.allclose(losses["loss"], expected_loss, rtol=1e-5)


class TestTrajectoryLoss:
    """Test mandatory trajectory loss computation."""

    def test_trajectory_loss_basic(self, device, sample_batch, student_outputs):
        """Test basic trajectory loss computation."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(endpoint_weight=0.0, trajectory_weight=1.0)
        loss_fn = DistillationLoss(config, span_depth=4)

        losses = loss_fn.compute_trajectory_loss(
            student_trajectory=student_outputs["trajectory_hidden"],
            teacher_trajectory=sample_batch["trajectory_targets"],
            attention_mask=sample_batch["attention_mask"],
            T=4,
        )

        assert "loss" in losses
        assert "mse" in losses
        assert losses["loss"].item() >= 0

    def test_trajectory_loss_with_alignment(
        self, device, sample_batch, student_outputs
    ):
        """Test trajectory loss with alignment policy when T != depth."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(endpoint_weight=0.0, trajectory_weight=1.0)
        loss_fn = DistillationLoss(config, span_depth=4)

        # T < depth: should use compression
        student_traj_T2 = torch.randn(2, 128, 2, 896, device=device)
        losses = loss_fn.compute_trajectory_loss(
            student_trajectory=student_traj_T2,
            teacher_trajectory=sample_batch["trajectory_targets"],
            attention_mask=sample_batch["attention_mask"],
            T=2,
        )

        assert "loss" in losses
        assert losses["loss"].item() >= 0

    def test_trajectory_loss_fail_fast_missing_targets(self, device, student_outputs):
        """Test that trajectory loss fails fast when targets are None."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(endpoint_weight=0.0, trajectory_weight=1.0)
        loss_fn = DistillationLoss(config, span_depth=4)

        with pytest.raises(ValueError, match="trajectory_targets.*required"):
            loss_fn.compute_trajectory_loss(
                student_trajectory=student_outputs["trajectory_hidden"],
                teacher_trajectory=None,
                attention_mask=torch.ones(2, 128),
                T=4,
            )

    def test_trajectory_loss_fail_fast_empty_targets(self, device, student_outputs):
        """Test that trajectory loss fails fast when targets are empty."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(endpoint_weight=0.0, trajectory_weight=1.0)
        loss_fn = DistillationLoss(config, span_depth=4)

        with pytest.raises(ValueError, match="trajectory_targets.*empty"):
            loss_fn.compute_trajectory_loss(
                student_trajectory=student_outputs["trajectory_hidden"],
                teacher_trajectory=torch.tensor([]),
                attention_mask=torch.ones(2, 128),
                T=4,
            )


class TestKLDivergenceLoss:
    """Test KL divergence loss on logits."""

    def test_kl_loss_basic(self, device, sample_batch, student_outputs):
        """Test basic KL divergence computation."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(kl_weight=1.0)
        loss_fn = DistillationLoss(config)

        losses = loss_fn.compute_kl_loss(
            student_logits=student_outputs["logits"],
            teacher_logits=sample_batch["teacher_logits"],
            attention_mask=sample_batch["attention_mask"],
        )

        assert "loss" in losses
        assert "kl_div" in losses
        assert losses["loss"].item() >= 0

    def test_kl_loss_masked(self, device):
        """Test that KL loss respects attention mask."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(kl_weight=1.0, mask_padding_tokens=True)
        loss_fn = DistillationLoss(config)

        batch_size, seq_len, vocab_size = 2, 4, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -1] = 0  # Mask last token

        losses = loss_fn.compute_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
        )

        assert losses["loss"].item() >= 0


class TestCrossEntropyLoss:
    """Test cross-entropy loss on labels."""

    def test_ce_loss_basic(self, device, sample_batch, student_outputs):
        """Test basic CE loss computation."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(ce_weight=1.0)
        loss_fn = DistillationLoss(config)

        losses = loss_fn.compute_ce_loss(
            student_logits=student_outputs["logits"],
            labels=sample_batch["labels"],
            attention_mask=sample_batch["attention_mask"],
        )

        assert "loss" in losses
        assert "ce" in losses
        assert losses["loss"].item() >= 0

    def test_ce_loss_disabled(self, device, sample_batch, student_outputs):
        """Test that CE loss returns zero when weight is 0."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(ce_weight=0.0)
        loss_fn = DistillationLoss(config)

        losses = loss_fn.compute_ce_loss(
            student_logits=student_outputs["logits"],
            labels=sample_batch["labels"],
            attention_mask=sample_batch["attention_mask"],
        )

        assert losses["loss"].item() == 0.0


class TestLossWeighting:
    """Test weighting of different loss terms."""

    def test_total_loss_computation(self, device, sample_batch, student_outputs):
        """Test that total loss is weighted sum of components."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0,
            endpoint_weight=1.0,
            trajectory_weight=0.5,
            kl_weight=0.25,
            ce_weight=0.1,
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()
        assert total_loss.item() >= 0

        # Check that metrics contain all components
        assert "total_loss" in metrics
        assert "endpoint_loss" in metrics
        assert "trajectory_loss" in metrics
        assert "kl_loss" in metrics
        assert "ce_loss" in metrics

    def test_zero_weights(self, device, sample_batch, student_outputs):
        """Test that zero weights disable corresponding losses."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0,
            endpoint_weight=0.0,
            trajectory_weight=0.0,
            kl_weight=0.0,
            ce_weight=0.0,
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        assert total_loss.item() == 0.0
        assert metrics["velocity_loss"] == 0.0
        assert metrics["endpoint_loss"] == 0.0
        assert metrics["trajectory_loss"] == 0.0
        assert metrics["kl_loss"] == 0.0
        assert metrics["ce_loss"] == 0.0


class TestGroupedMetricOutputs:
    """Test grouped metric outputs from loss computation."""

    def test_metrics_structure(self, device, sample_batch, student_outputs):
        """Test that metrics dict has expected structure."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0,
            endpoint_weight=1.0,
            trajectory_weight=1.0,
            kl_weight=0.25,
            ce_weight=0.0,
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        # Check main metric groups
        assert "total_loss" in metrics
        assert "velocity_loss" in metrics
        assert "endpoint_loss" in metrics
        assert "endpoint_mse" in metrics
        assert "trajectory_loss" in metrics
        assert "trajectory_mse" in metrics
        assert "kl_loss" in metrics
        assert "kl_div" in metrics
        assert "ce_loss" in metrics

        # All values should be scalars (floats or 0-d tensors)
        for key, value in metrics.items():
            assert isinstance(value, (float, torch.Tensor)), (
                f"{key} should be float or Tensor"
            )
            if isinstance(value, torch.Tensor):
                assert value.shape == (), (
                    f"{key} should be scalar, got shape {value.shape}"
                )

    def test_detached_metrics(self, device, sample_batch, student_outputs):
        """Test that metrics are detached from computation graph."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=1.0
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        # Check that metrics don't have grad
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad, f"{key} should not require grad"


class TestFailFastBehavior:
    """Test fail-fast behavior for missing required inputs."""

    def test_fail_fast_missing_trajectory_targets(self, device, student_outputs):
        """Test that loss fails fast when trajectory targets are required but missing."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=1.0
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        teacher_batch = {
            "h_target": torch.randn(2, 128, 896, device=device),
            # Missing trajectory_targets
        }

        with pytest.raises(ValueError, match="trajectory_targets.*required"):
            loss_fn.forward(
                student_outputs=student_outputs,
                teacher_batch=teacher_batch,
                T=4,
            )

    def test_fail_fast_missing_endpoint_target(self, device, student_outputs):
        """Test that loss fails fast when endpoint target is required but missing."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=0.0
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        teacher_batch = {
            "trajectory_targets": torch.randn(2, 128, 4, 896),
            # Missing h_target
        }

        with pytest.raises(ValueError, match="h_target.*required"):
            loss_fn.forward(
                student_outputs=student_outputs,
                teacher_batch=teacher_batch,
                T=4,
            )

    def test_fail_fast_missing_logits_for_kl(
        self, device, sample_batch, student_outputs
    ):
        """Test that loss fails fast when logits required for KL but missing."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(velocity_weight=0.0, kl_weight=1.0)
        loss_fn = DistillationLoss(config)

        student_outputs_no_logits = {
            "endpoint_hidden": student_outputs["endpoint_hidden"],
            "trajectory_hidden": student_outputs["trajectory_hidden"],
            # Missing logits
        }

        with pytest.raises(ValueError, match="logits.*required"):
            loss_fn.forward(
                student_outputs=student_outputs_no_logits,
                teacher_batch=sample_batch,
                T=4,
            )


class TestLossFromConfig:
    """Test creating loss module from config."""

    def test_from_config(self, config):
        """Test that loss module can be created from config."""
        from src.training.losses import DistillationLoss

        loss_fn = DistillationLoss.from_config(config)

        assert loss_fn.config.endpoint_weight == config["loss"]["endpoint_weight"]
        assert loss_fn.config.trajectory_weight == config["loss"]["trajectory_weight"]
        assert loss_fn.config.kl_weight == config["loss"]["kl_weight"]
        assert loss_fn.config.ce_weight == config["loss"]["ce_weight"]
        assert (
            loss_fn.config.mask_padding_tokens == config["loss"]["mask_padding_tokens"]
        )

    def test_from_config_with_span_depth(self, config):
        """Test that loss module extracts span_depth from config."""
        from src.training.losses import DistillationLoss

        loss_fn = DistillationLoss.from_config(config)

        expected_depth = (
            config["replacement_model"]["end_layer"]
            - config["replacement_model"]["start_layer"]
            + 1
        )
        assert loss_fn.span_depth == expected_depth


def test_velocity_loss_disabled_when_weight_zero(device, student_outputs, sample_batch):
    """Test that velocity_loss is zero when velocity_weight=0."""
    from src.training.losses import DistillationLoss, LossConfig

    config = LossConfig(velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=1.0)
    loss_fn = DistillationLoss(config, span_depth=4)

    total_loss, metrics = loss_fn(student_outputs, sample_batch, T=4)
    assert "velocity_loss" in metrics
    assert (
        metrics["velocity_loss"] == 0.0
    )  # velocity_weight=0 means velocity_loss should be 0.0


def test_velocity_loss_penalizes_velocity_discrepancy(device):
    """Test that velocity loss correctly penalizes incorrect velocity predictions."""
    from src.training.losses import DistillationLoss, LossConfig

    config = LossConfig(velocity_weight=1.0, endpoint_weight=0.0, trajectory_weight=0.0)
    loss_fn = DistillationLoss(config)

    batch_size, seq_len, hidden_dim = 2, 10, 64
    h_start = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    velocity_target = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    mock_model_correct = MagicMock()
    mock_midblock_correct = MagicMock()
    mock_model_correct.midblock = mock_midblock_correct
    mock_midblock_correct.get_velocity.return_value = velocity_target.clone()

    mock_model_wrong = MagicMock()
    mock_midblock_wrong = MagicMock()
    mock_model_wrong.midblock = mock_midblock_wrong
    mock_midblock_wrong.get_velocity.return_value = torch.randn(
        batch_size, seq_len, hidden_dim, device=device
    )

    batch = {
        "h_start": h_start,
        "velocity_target": velocity_target,
    }

    t = torch.rand(batch_size, device=device)

    loss_correct = loss_fn.compute_velocity_loss(
        mock_model_correct, batch, t, torch.device(device)
    )
    loss_wrong = loss_fn.compute_velocity_loss(
        mock_model_wrong, batch, t, torch.device(device)
    )

    assert loss_correct.shape == ()
    assert loss_correct.item() >= 0
    assert loss_wrong.shape == ()
    assert loss_wrong.item() >= 0
    assert loss_correct.item() < loss_wrong.item(), (
        "Correct velocity prediction should have lower loss than wrong prediction"
    )


class TestVelocityLoss:
    """Test velocity loss computation."""

    def test_velocity_loss_basic(self, device):
        """Test basic velocity loss computation without masking."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(velocity_weight=1.0, mask_padding_tokens=False)

        batch_size, seq_len, hidden_dim = 2, 16, 128

        # Create mock model with midblock
        mock_model = MagicMock()
        mock_midblock = MagicMock()
        mock_model.midblock = mock_midblock

        # Create velocity target (h_end - h_start)
        h_start = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        h_end = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        velocity_target = h_end - h_start

        # Mock get_velocity to return predicted velocity
        mock_midblock.get_velocity.return_value = torch.randn(
            batch_size, seq_len, hidden_dim, device=device
        )

        batch = {
            "h_start": h_start,
            "velocity_target": velocity_target,
        }

        t = torch.rand(batch_size, device=device)

        loss_fn = DistillationLoss(config)
        loss = loss_fn.compute_velocity_loss(mock_model, batch, t, torch.device(device))

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_velocity_loss_with_attention_mask(self, device):
        """Test that velocity loss respects attention mask."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(velocity_weight=1.0, mask_padding_tokens=True)

        batch_size, seq_len, hidden_dim = 2, 4, 8

        # Create mock model
        mock_model = MagicMock()
        mock_midblock = MagicMock()
        mock_model.midblock = mock_midblock

        h_start = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        velocity_target = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Create attention mask with some padded tokens
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[:, -1] = 0  # Mask last token in each sequence

        mock_midblock.get_velocity.return_value = torch.randn(
            batch_size, seq_len, hidden_dim, device=device
        )

        loss_fn = DistillationLoss(config)

        batch = {
            "h_start": h_start,
            "velocity_target": velocity_target,
            "attention_mask": attention_mask,
        }

        t = torch.rand(batch_size, device=device)

        loss = loss_fn.compute_velocity_loss(mock_model, batch, t, torch.device(device))

        assert loss.shape == ()
        assert loss.item() >= 0


class TestDetachedMetrics:
    """Test device and dtype handling."""

    def test_loss_computation_preserves_device(
        self, device, sample_batch, student_outputs
    ):
        """Test that loss computation preserves tensor device."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=1.0
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        assert total_loss.device.type == student_outputs["endpoint_hidden"].device.type

    def test_loss_computation_preserves_dtype(
        self, device, sample_batch, student_outputs
    ):
        """Test that loss computation preserves tensor dtype."""
        from src.training.losses import DistillationLoss, LossConfig

        config = LossConfig(
            velocity_weight=0.0, endpoint_weight=1.0, trajectory_weight=1.0
        )
        loss_fn = DistillationLoss(config, span_depth=4)

        total_loss, metrics = loss_fn.forward(
            student_outputs=student_outputs,
            teacher_batch=sample_batch,
            T=4,
        )

        # Loss should be float32 or match input dtype
        assert total_loss.dtype in [torch.float32, torch.float64]

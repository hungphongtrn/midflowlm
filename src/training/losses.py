"""Loss functions for continuous-time velocity distillation.

This module implements the supervision objectives for training the flow-based
hidden-state refiner using continuous-time velocity matching. It supports:
- Velocity MSE: Matches predicted velocity to teacher velocity targets
- Endpoint hidden-state MSE (legacy, optional)
- Trajectory hidden-state loss using alignment policy (legacy, optional)
- KL divergence on logits (optional)
- Cross-entropy on labels (optional)

The loss module now defaults to continuous-time velocity supervision for
architecture training, with legacy endpoint/trajectory losses as optional.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.alignment import TrajectoryAligner, AlignmentPolicy


@dataclass
class LossConfig:
    """Configuration for distillation losses.

    Attributes:
        velocity_weight: Weight for continuous-time velocity MSE loss
        endpoint_weight: Weight for endpoint hidden-state MSE loss (legacy)
        trajectory_weight: Weight for trajectory hidden-state loss (legacy)
        kl_weight: Weight for KL divergence on logits (0.0 to disable)
        ce_weight: Weight for cross-entropy on labels (0.0 to disable)
        mask_padding_tokens: Whether to mask padding tokens in loss computation
    """

    velocity_weight: float = 1.0
    endpoint_weight: float = 0.0
    trajectory_weight: float = 0.0
    kl_weight: float = 0.0
    ce_weight: float = 0.0
    mask_padding_tokens: bool = True


class DistillationLoss(nn.Module):
    """Distillation loss for training the flow-based hidden-state refiner.

    This module combines multiple loss terms with continuous-time velocity
    supervision as the primary training objective:
    1. Velocity MSE: Matches predicted velocity to teacher velocity (default)
    2. Endpoint MSE: Matches final hidden state to teacher endpoint (optional)
    3. Trajectory MSE: Matches intermediate hidden states (optional)
    4. KL Divergence: Matches student logits to teacher logits (optional)
    5. Cross-Entropy: Standard language modeling loss (optional)

    For architecture training (default), only velocity_weight should be non-zero
    and the batch must contain 'velocity_target' from teacher cache.

    Args:
        config: LossConfig with weight settings
        span_depth: Number of layers in the teacher span (for alignment)
        aligner_config: Optional configuration for trajectory alignment
    """

    def __init__(
        self,
        config: LossConfig,
        span_depth: Optional[int] = None,
        aligner_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config = config
        self.span_depth = span_depth
        self.aligner_config = aligner_config or {}

        # Create trajectory aligner if span_depth is provided (legacy)
        if span_depth is not None:
            self.aligner = TrajectoryAligner(
                span_depth=span_depth,
                config=self.aligner_config,
            )
        else:
            self.aligner = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DistillationLoss":
        """Create a DistillationLoss from a configuration dictionary.

        Args:
            config: Configuration dict with 'loss' and 'replacement_model' sections

        Returns:
            DistillationLoss instance configured from config
        """
        loss_config = LossConfig(
            velocity_weight=config["loss"].get("velocity_weight", 1.0),
            endpoint_weight=config["loss"].get("endpoint_weight", 0.0),
            trajectory_weight=config["loss"].get("trajectory_weight", 0.0),
            kl_weight=config["loss"].get("kl_weight", 0.0),
            ce_weight=config["loss"].get("ce_weight", 0.0),
            mask_padding_tokens=config["loss"].get("mask_padding_tokens", True),
        )

        # Compute span_depth from layer range (legacy)
        start_layer = config["replacement_model"]["start_layer"]
        end_layer = config["replacement_model"]["end_layer"]
        span_depth = end_layer - start_layer + 1

        # Get alignment config if present (legacy)
        aligner_config = config["replacement_model"].get("trajectory_alignment", {})

        return cls(
            config=loss_config,
            span_depth=span_depth,
            aligner_config=aligner_config,
        )

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_batch: Dict[str, torch.Tensor],
        T: int,
        model: Optional[nn.Module] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total distillation loss.

        Args:
            student_outputs: Dict with 'endpoint_hidden', 'trajectory_hidden', 'logits'
            teacher_batch: Dict with 'h_target', 'trajectory_targets', 'teacher_logits', 'labels', 'velocity_target'
            T: Number of student refinement steps
            model: Student model (required for velocity loss)
            t: Continuous time values [batch_size] (required for velocity loss)

        Returns:
            Tuple of (total_loss, metrics_dict)
            - total_loss: Scalar tensor for backprop
            - metrics_dict: Dict of detached scalar metrics for logging
        """
        device = self._get_device(student_outputs, teacher_batch)
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        # Architecture-training contract: if teacher_logits not in batch, kl_weight must be 0.0
        if "teacher_logits" not in teacher_batch:
            if self.config.kl_weight > 0.0:
                raise ValueError(
                    "teacher_logits not in teacher_batch but kl_weight > 0.0. "
                    "Either provide teacher_logits or set kl_weight=0.0. "
                    "Architecture training defaults should not use KL loss."
                )

        # Velocity loss (primary training objective)
        if self.config.velocity_weight > 0:
            # Fail fast if required inputs are missing
            if model is None:
                raise ValueError(
                    "model is required for velocity loss but got None. "
                    "Pass the student model to compute velocity loss."
                )
            if t is None:
                raise ValueError(
                    "t (continuous time values) is required for velocity loss but got None. "
                    "Sample t ~ U(0, 1) before calling forward."
                )
            if "velocity_target" not in teacher_batch:
                raise ValueError(
                    "velocity_target is required but missing from teacher_batch. "
                    "Run teacher cache generation with velocity targets."
                )

            velocity_loss = self.compute_velocity_loss(
                model=model,
                teacher_batch=teacher_batch,
                t=t,
                device=device,
            )
            weighted_velocity_loss = velocity_loss * self.config.velocity_weight
            total_loss = total_loss + weighted_velocity_loss

            metrics["velocity_loss"] = velocity_loss.detach().item()
        else:
            metrics["velocity_loss"] = 0.0

        # Endpoint loss (legacy, optional)
        if self.config.endpoint_weight > 0:
            # Fail fast if endpoint target is missing
            if "h_target" not in teacher_batch:
                raise ValueError(
                    "h_target (teacher endpoint hidden state) is required but missing from teacher_batch. "
                    "Endpoint supervision is enabled when endpoint_weight > 0."
                )

            endpoint_losses = self.compute_endpoint_loss(
                student_hidden=student_outputs["endpoint_hidden"],
                teacher_hidden=teacher_batch["h_target"],
                attention_mask=teacher_batch.get("attention_mask"),
            )
            endpoint_loss = endpoint_losses["loss"] * self.config.endpoint_weight
            total_loss = total_loss + endpoint_loss

            metrics["endpoint_loss"] = endpoint_losses["loss"].detach().item()
            metrics["endpoint_mse"] = endpoint_losses["mse"].detach().item()
        else:
            metrics["endpoint_loss"] = 0.0
            metrics["endpoint_mse"] = 0.0

        # Trajectory loss (legacy, optional)
        if self.config.trajectory_weight > 0:
            # Fail fast if trajectory targets are missing
            if "trajectory_targets" not in teacher_batch:
                raise ValueError(
                    "trajectory_targets is required but missing from teacher_batch. "
                    "Trajectory supervision is enabled when trajectory_weight > 0."
                )

            trajectory_losses = self.compute_trajectory_loss(
                student_trajectory=student_outputs["trajectory_hidden"],
                teacher_trajectory=teacher_batch["trajectory_targets"],
                attention_mask=teacher_batch.get("attention_mask"),
                T=T,
            )
            trajectory_loss = trajectory_losses["loss"] * self.config.trajectory_weight
            total_loss = total_loss + trajectory_loss

            metrics["trajectory_loss"] = trajectory_losses["loss"].detach().item()
            metrics["trajectory_mse"] = trajectory_losses["mse"].detach().item()
        else:
            metrics["trajectory_loss"] = 0.0
            metrics["trajectory_mse"] = 0.0

        # KL divergence loss (optional)
        if self.config.kl_weight > 0:
            # Fail fast if logits are missing
            if "logits" not in student_outputs:
                raise ValueError(
                    "student_outputs['logits'] is required for KL loss but is missing. "
                    "Either provide logits or set kl_weight=0."
                )
            if "teacher_logits" not in teacher_batch:
                raise ValueError(
                    "teacher_logits is required for KL loss but is missing from teacher_batch. "
                    "teacher_logits is not part of the default architecture-training cache; "
                    "set kl_weight=0 or add an explicit behavior-training path."
                )

            kl_losses = self.compute_kl_loss(
                student_logits=student_outputs["logits"],
                teacher_logits=teacher_batch["teacher_logits"],
                attention_mask=teacher_batch.get("attention_mask"),
            )
            kl_loss = kl_losses["loss"] * self.config.kl_weight
            total_loss = total_loss + kl_loss

            metrics["kl_loss"] = kl_losses["loss"].detach().item()
            metrics["kl_div"] = kl_losses["kl_div"].detach().item()
        else:
            metrics["kl_loss"] = 0.0
            metrics["kl_div"] = 0.0

        # Cross-entropy loss (optional)
        if self.config.ce_weight > 0:
            ce_losses = self.compute_ce_loss(
                student_logits=student_outputs["logits"],
                labels=teacher_batch["labels"],
                attention_mask=teacher_batch.get("attention_mask"),
            )
            ce_loss = ce_losses["loss"] * self.config.ce_weight
            total_loss = total_loss + ce_loss

            metrics["ce_loss"] = ce_losses["loss"].detach().item()
            metrics["ce"] = ce_losses["ce"].detach().item()
        else:
            metrics["ce_loss"] = 0.0
            metrics["ce"] = 0.0

        metrics["total_loss"] = total_loss.detach().item()

        return total_loss, metrics

    def compute_velocity_loss(
        self,
        model: nn.Module,
        teacher_batch: Dict[str, torch.Tensor],
        t: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute continuous-time velocity MSE loss.

        Constructs h_t = h_start + t * velocity_target and computes MSE
        between model's predicted velocity and target velocity. Supports
        attention-masked loss computation when mask_padding_tokens is enabled.

        Args:
            model: Student model with midblock.get_velocity() method
            teacher_batch: Dict with 'h_start', 'velocity_target', 'attention_mask'
            t: Continuous time values [batch_size], sampled externally by caller
               (typically via trainer.sample_continuous_time())
            device: Device to run computation on

        Returns:
            Velocity MSE loss scalar tensor
        """
        batch_size = teacher_batch["h_start"].shape[0]

        # Get velocity target (straight-line path: h_end - h_start)
        velocity_target = teacher_batch["velocity_target"]

        # Construct h_t = h_start + t * velocity_target (straight-line interpolation)
        # t shape: [batch_size] -> expand to [batch_size, 1, 1] for broadcasting
        t_expanded = t.view(batch_size, 1, 1)
        h_t = teacher_batch["h_start"] + t_expanded * velocity_target

        # Get velocity prediction from model's midblock
        v_pred = model.midblock.get_velocity(
            h_t=h_t,
            h_start=teacher_batch["h_start"],
            attention_mask=teacher_batch.get("attention_mask"),
            t=t,
        )

        # Compute MSE between predicted and target velocity
        squared_error = (v_pred - velocity_target) ** 2

        # Apply masking if enabled
        if (
            self.config.mask_padding_tokens
            and teacher_batch.get("attention_mask") is not None
        ):
            # attention_mask: [batch, seq] -> [batch, seq, 1]
            mask = teacher_batch["attention_mask"].unsqueeze(-1)
            masked_error = squared_error * mask
            loss = masked_error.sum() / (mask.sum() * v_pred.shape[-1]).clamp(min=1.0)
        else:
            loss = squared_error.mean()

        return loss

    def compute_endpoint_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute endpoint hidden-state MSE loss.

        Args:
            student_hidden: Student endpoint hidden states [batch, seq, hidden]
            teacher_hidden: Teacher endpoint hidden states [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq] (1 for valid tokens)

        Returns:
            Dict with 'loss' and 'mse' keys
        """
        # Fail fast if teacher hidden is missing
        if teacher_hidden is None:
            raise ValueError(
                "h_target (teacher endpoint hidden state) is required but got None"
            )

        # Compute squared error
        diff = student_hidden - teacher_hidden
        squared_error = (diff**2).mean(dim=-1)  # [batch, seq]

        # Apply masking if enabled
        if self.config.mask_padding_tokens and attention_mask is not None:
            masked_error = squared_error * attention_mask
            loss = masked_error.sum() / attention_mask.sum().clamp(min=1.0)
        else:
            loss = squared_error.mean()

        return {
            "loss": loss,
            "mse": loss.detach(),  # Same as loss for MSE, but detached
        }

    def compute_trajectory_loss(
        self,
        student_trajectory: torch.Tensor,
        teacher_trajectory: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        T: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute trajectory hidden-state loss with alignment.

        Args:
            student_trajectory: Student trajectory hidden states [batch, seq, T, hidden]
            teacher_trajectory: Teacher trajectory hidden states [batch, seq, depth, hidden]
            attention_mask: Attention mask [batch, seq]
            T: Number of student steps

        Returns:
            Dict with 'loss' and 'mse' keys
        """
        # Fail fast validation
        if teacher_trajectory is None:
            raise ValueError(
                "trajectory_targets is required but got None. "
                "Trajectory supervision is enabled when trajectory_weight > 0."
            )

        if teacher_trajectory.numel() == 0:
            raise ValueError("trajectory_targets is empty")

        # Align teacher trajectory to student steps
        if self.aligner is not None:
            aligned_teacher = self.aligner.align_targets(
                trajectory_targets=teacher_trajectory,
                T=T,
            )
        else:
            # No aligner - assume exact match
            if teacher_trajectory.shape[2] != T:
                raise ValueError(
                    f"Trajectory depth mismatch: teacher has {teacher_trajectory.shape[2]} "
                    f"layers but T={T}. Provide span_depth to enable alignment."
                )
            aligned_teacher = teacher_trajectory

        # Compute MSE over all trajectory steps
        diff = student_trajectory - aligned_teacher
        squared_error = (diff**2).mean(dim=-1)  # [batch, seq, T]

        # Apply masking if enabled
        if self.config.mask_padding_tokens and attention_mask is not None:
            # Expand mask for trajectory dimension
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
            masked_error = squared_error * mask_expanded
            loss = masked_error.sum() / (attention_mask.sum() * T).clamp(min=1.0)
        else:
            loss = squared_error.mean()

        return {
            "loss": loss,
            "mse": loss.detach(),
        }

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute KL divergence loss on logits.

        Args:
            student_logits: Student logits [batch, seq, vocab]
            teacher_logits: Teacher logits [batch, seq, vocab]
            attention_mask: Attention mask [batch, seq]

        Returns:
            Dict with 'loss' and 'kl_div' keys
        """
        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # KL divergence: sum(teacher_probs * (log(teacher_probs) - student_log_probs))
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        )  # [batch, seq, vocab]

        # Sum over vocab dimension
        kl_div = kl_div.sum(dim=-1)  # [batch, seq]

        # Apply masking if enabled
        if self.config.mask_padding_tokens and attention_mask is not None:
            masked_kl = kl_div * attention_mask
            loss = masked_kl.sum() / attention_mask.sum().clamp(min=1.0)
        else:
            loss = kl_div.mean()

        return {
            "loss": loss,
            "kl_div": loss.detach(),
        }

    def compute_ce_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-entropy loss on labels.

        Args:
            student_logits: Student logits [batch, seq, vocab]
            labels: Target labels [batch, seq]
            attention_mask: Attention mask [batch, seq]

        Returns:
            Dict with 'loss' and 'ce' keys
        """
        # Return zero loss if CE is disabled
        if self.config.ce_weight == 0.0:
            return {
                "loss": torch.tensor(0.0, device=student_logits.device),
                "ce": torch.tensor(0.0, device=student_logits.device),
            }

        # Flatten for cross_entropy
        batch_size, seq_len, vocab_size = student_logits.shape
        logits_flat = student_logits.view(-1, vocab_size)  # [batch*seq, vocab]
        labels_flat = labels.view(-1)  # [batch*seq]

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction="none",
        )  # [batch*seq]

        # Reshape back
        ce_loss = ce_loss.view(batch_size, seq_len)  # [batch, seq]

        # Apply masking if enabled
        if self.config.mask_padding_tokens and attention_mask is not None:
            # Also mask out -100 labels (ignore_index)
            valid_mask = attention_mask * (labels != -100).float()
            masked_ce = ce_loss * valid_mask
            loss = masked_ce.sum() / valid_mask.sum().clamp(min=1.0)
        else:
            loss = ce_loss.mean()

        return {
            "loss": loss,
            "ce": loss.detach(),
        }

    def _get_device(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_batch: Dict[str, torch.Tensor],
    ) -> torch.device:
        """Get device from student outputs or teacher batch."""
        # Try student outputs first
        for key in ["endpoint_hidden", "trajectory_hidden", "logits"]:
            if key in student_outputs:
                return student_outputs[key].device
        # Try teacher batch
        for key in ["h_start", "h_target", "velocity_target"]:
            if key in teacher_batch:
                return teacher_batch[key].device
        return torch.device("cpu")

    def get_trainable_weights(self) -> Dict[str, float]:
        """Get current loss weights for logging.

        Returns:
            Dict mapping loss name to weight
        """
        return {
            "velocity": self.config.velocity_weight,
            "endpoint": self.config.endpoint_weight,
            "trajectory": self.config.trajectory_weight,
            "kl": self.config.kl_weight,
            "ce": self.config.ce_weight,
        }

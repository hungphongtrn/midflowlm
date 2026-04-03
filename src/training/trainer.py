"""Primary trainer for iterative midblock training.

This is the DEFAULT trainer for midflowlm, using online calculation instead of caching.
Teacher targets (h_start, h_target, velocity_target, teacher_logits) are extracted
on-the-fly via model.extract_teacher_targets() in one no-grad forward pass.

Key Features:
- Token batches (input_ids, attention_mask) as primary input
- On-the-fly teacher target extraction via extract_teacher_targets()
- No teacher-state caching required (disk-space friendly)
- Velocity + KL loss computation using extracted targets
- Tensorboard logging support with gradient norm tracking
- Gradient accumulation and AMP support

For backward compatibility, OnlineNoCacheTrainer is an alias to this Trainer class.

Note: Cache-based training is still available via CachedTrainer in cached_trainer.py,
but is deprecated and should only be used for legacy workflows.
"""

import logging
import math
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class Trainer:
    """Primary trainer for iterative midblock training with online teacher target extraction.

    This is the DEFAULT trainer for midflowlm. It uses online calculation instead of
    caching, making it disk-space friendly and simpler to use.

    This trainer:
    - Uses token batches (input_ids, attention_mask) only
    - Extracts teacher targets via model.extract_teacher_targets() in one no-grad forward
    - Computes velocity + KL loss using the extracted targets
    - Supports gradient accumulation, AMP, and tensorboard logging

    Args:
        model: FrozenQwenStudent model with extract_teacher_targets() method
        loss_fn: Loss function module
        config: Configuration dictionary
        device: Device to train on
        train_dataloader: Training dataloader (token batches)
        val_dataloader: Validation dataloader (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        device: Union[str, torch.device] = "cuda",
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.global_step = 0
        self.current_epoch = 0
        self.accumulation_step = 0
        self._just_validated = False

        self.optimizer_config = config.get("optimizer", {})
        self.scheduler_config = config.get("scheduler", {})
        self.train_loop_config = config.get("train_loop", {})
        self.model_config = config.get("model", {})
        self.loss_config = config.get("loss", {})
        self.kl_weight = float(self.loss_config.get("kl_weight", 0.0))

        self._setup_precision()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        if self.use_amp:
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.accumulate_grad_batches = self.train_loop_config.get(
            "accumulate_grad_batches", 1
        )
        self.grad_clip_norm = self.optimizer_config.get("grad_clip_norm", 1.0)
        self.train_T_values = self.model_config.get("train_T_values", [4])
        self.train_T_weights = self.model_config.get("train_T_weights", None)

        self.best_val_metric = float("inf")
        self.best_checkpoint_path = None
        self.log_every_n_steps = self.train_loop_config.get("log_every_n_steps", 10)

        self.monitor_key = config.get("logging", {}).get("monitor", "val/total_loss")
        self.monitor_mode = config.get("logging", {}).get("mode", "min")

        # Tensorboard setup
        self.tensorboard_config = config.get("tensorboard", {})
        self.use_tensorboard = self.tensorboard_config.get("enabled", False)
        self.tensorboard_log_dir = self.tensorboard_config.get(
            "log_dir", "./tensorboard"
        )
        self.tensorboard_writer = None

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                Path(self.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(
                    log_dir=self.tensorboard_log_dir
                )
                logger.info(
                    f"Tensorboard enabled: logging to {self.tensorboard_log_dir}"
                )
            except ImportError:
                logger.warning(
                    "Tensorboard not available. Install with: pip install tensorboard"
                )
                self.use_tensorboard = False

        logger.info(f"Initialized OnlineNoCacheTrainer on device: {self.device}")
        logger.info(f"  Precision: {self.precision}, AMP: {self.use_amp}")
        logger.info(f"  Gradient accumulation: {self.accumulate_grad_batches}")
        logger.info(f"  Train T values: {self.train_T_values}")
        logger.info(f"  KL weight: {self.kl_weight}")
        logger.info(f"  CE weight: {self.loss_config.get('ce_weight', 0.0)}")
        logger.info(
            f"  Tensorboard: {'enabled' if self.use_tensorboard else 'disabled'}"
        )
        logger.info(
            f"  Best-checkpoint monitor: {self.monitor_key} ({self.monitor_mode})"
        )

    def _autocast_context(self):
        if not self.use_amp or self.device.type != "cuda":
            return None
        return (
            torch.amp.autocast("cuda", dtype=self.amp_dtype)
            if hasattr(torch, "amp")
            else torch.cuda.amp.autocast(dtype=self.amp_dtype)
        )

    def _setup_precision(self):
        precision_str = self.train_loop_config.get("precision", "fp32")
        if precision_str == "fp32":
            self.precision = "fp32"
            self.use_amp = False
            self.amp_dtype = torch.float32
        elif precision_str == "fp16-mixed":
            self.precision = "fp16"
            self.use_amp = True
            self.amp_dtype = torch.float16
        elif precision_str == "bf16-mixed":
            self.precision = "bf16"
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
        else:
            logger.warning(f"Unknown precision: {precision_str}, defaulting to fp32")
            self.precision = "fp32"
            self.use_amp = False
            self.amp_dtype = torch.float32

    def _create_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.optimizer_config.get("name", "adamw")
        lr = self.optimizer_config.get("learning_rate", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 0.01)
        betas = tuple(self.optimizer_config.get("betas", [0.9, 0.95]))
        eps = self.optimizer_config.get("eps", 1e-8)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if opt_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        elif opt_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        logger.info(f"Created {opt_name} optimizer with lr={lr}")
        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        sched_name = self.scheduler_config.get("name", "cosine_with_warmup")
        if sched_name is None or sched_name.lower() == "none":
            return None

        warmup_steps = self.scheduler_config.get("warmup_steps", 0)

        if sched_name.lower() == "cosine_with_warmup":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, self._get_max_steps() - warmup_steps)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(
                f"Created cosine_with_warmup scheduler with warmup_steps={warmup_steps}"
            )
        elif sched_name.lower() == "linear_with_warmup":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, self._get_max_steps() - warmup_steps)
                )
                return max(0.0, 1.0 - progress)

            scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(
                f"Created linear_with_warmup scheduler with warmup_steps={warmup_steps}"
            )
        else:
            scheduler = None

        return scheduler

    def _get_max_steps(self) -> int:
        max_epochs = self.train_loop_config.get("max_epochs", 3)
        if self.train_dataloader is not None:
            steps_per_epoch = len(self.train_dataloader)
            return max_epochs * steps_per_epoch
        return max_epochs * 1000

    def sample_T(self) -> int:
        if len(self.train_T_values) == 1:
            return self.train_T_values[0]
        if self.train_T_weights is not None:
            return random.choices(
                self.train_T_values, weights=self.train_T_weights, k=1
            )[0]
        else:
            return random.choice(self.train_T_values)

    def sample_continuous_time(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def _is_better(self, new_val: float) -> bool:
        if self.monitor_mode == "min":
            return new_val < self.best_val_metric
        else:
            return new_val > self.best_val_metric

    def _log_to_tensorboard(
        self, metrics: Dict[str, float], prefix: str = "train"
    ) -> None:
        """Log metrics to tensorboard.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        if not self.use_tensorboard or self.tensorboard_writer is None:
            return

        step = self.global_step

        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tensorboard_writer.add_scalar(f"{prefix}/{key}", value, step)

        # Log combined KL + CE loss separately
        kl_loss = metrics.get("kl_loss", 0.0)
        ce_loss = metrics.get("ce_loss", 0.0)
        if kl_loss > 0 or ce_loss > 0:
            combined_behavior_loss = kl_loss + ce_loss
            self.tensorboard_writer.add_scalar(
                f"{prefix}/combined_behavior_loss", combined_behavior_loss, step
            )

        # Flush periodically
        if step % 100 == 0:
            self.tensorboard_writer.flush()

    def train_step(
        self, batch: Dict[str, torch.Tensor], T: Optional[int] = None
    ) -> Dict[str, float]:
        """Execute one training step with on-the-fly teacher target extraction.

        Args:
            batch: Batch with input_ids and attention_mask (token batch)
            T: Number of refinement steps (sampled if None)

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        if T is None:
            T = self.sample_T()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logger.debug(f"Extracting teacher targets for batch...")
        teacher_targets = self.model.extract_teacher_targets(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logger.debug(f"Teacher targets extracted successfully")
        h_start = teacher_targets["h_start"]
        h_target = teacher_targets["h_target"]
        velocity_target = teacher_targets["velocity_target"]
        teacher_logits = teacher_targets["teacher_logits"]

        # For CE loss, pass input_ids directly. The loss function will extract
        # targets as input_ids[:, 1:] and use logits[:, :-1] for predictions.

        sample_continuous_time = self.train_loop_config.get(
            "sample_continuous_time", False
        )
        if sample_continuous_time:
            batch_size = input_ids.shape[0]
            t = self.sample_continuous_time(batch_size, self.device)
        else:
            t = None

        autocast_context = self._autocast_context()
        if autocast_context is not None:
            with autocast_context:
                student_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=T,
                    return_dict=True,
                )
                total_loss, loss_metrics = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_batch={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "h_start": h_start,
                        "h_target": h_target,
                        "velocity_target": velocity_target,
                        "teacher_logits": teacher_logits,
                        "labels": input_ids,
                    },
                    T=T,
                    model=self.model,
                    t=t,
                )
        else:
            student_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=T,
                return_dict=True,
            )
            total_loss, loss_metrics = self.loss_fn(
                student_outputs=student_outputs,
                teacher_batch={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "h_start": h_start,
                    "h_target": h_target,
                    "velocity_target": velocity_target,
                    "teacher_logits": teacher_logits,
                    "labels": input_ids,
                },
                T=T,
                model=self.model,
                t=t,
            )

        if self.accumulate_grad_batches > 1:
            total_loss = total_loss / self.accumulate_grad_batches

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        self.accumulation_step += 1

        if self.accumulation_step >= self.accumulate_grad_batches:
            # Compute gradient norm before clipping
            grad_norm = 0.0
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            # Calculate gradient norm
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.detach().data.norm(2).item() ** 2
            grad_norm = grad_norm**0.5

            # Clip gradients if enabled
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.accumulation_step = 0
            self.global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            self._last_grad_norm = grad_norm
            self._just_validated = False
        else:
            grad_norm = 0.0
            self._last_grad_norm = grad_norm

        metrics = {
            "loss": loss_metrics.get("total_loss", total_loss.item()),
            "velocity_loss": loss_metrics.get("velocity_loss", 0.0),
            "kl_loss": loss_metrics.get("kl_loss", 0.0),
            "ce_loss": loss_metrics.get("ce_loss", 0.0),
            "T": T,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": self._last_grad_norm,
        }

        if sample_continuous_time and t is not None:
            metrics["t_mean"] = t.mean().item()
            metrics["t_std"] = t.std().item()

        if (
            self.accumulation_step == 0
            and self.use_tensorboard
            and self.tensorboard_writer is not None
        ):
            self._log_to_tensorboard(metrics)

        return metrics

    def val_step(
        self, batch: Dict[str, torch.Tensor], T: Optional[int] = None
    ) -> Dict[str, float]:
        """Execute one validation step with on-the-fly teacher target extraction.

        Args:
            batch: Batch with input_ids and attention_mask
            T: Number of refinement steps (sampled if None)

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        if T is None:
            T = self.sample_T()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        teacher_targets = self.model.extract_teacher_targets(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h_start = teacher_targets["h_start"]
        h_target = teacher_targets["h_target"]
        velocity_target = teacher_targets["velocity_target"]
        teacher_logits = teacher_targets["teacher_logits"]

        # For CE loss, pass input_ids directly. The loss function will extract
        # targets as input_ids[:, 1:] and use logits[:, :-1] for predictions.

        sample_continuous_time = self.train_loop_config.get(
            "sample_continuous_time", False
        )
        if sample_continuous_time:
            batch_size = input_ids.shape[0]
            t = self.sample_continuous_time(batch_size, self.device)
        else:
            t = None

        with torch.no_grad():
            autocast_context = self._autocast_context()
            if autocast_context is not None:
                with autocast_context:
                    student_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_steps=T,
                        return_dict=True,
                    )
                    total_loss, loss_metrics = self.loss_fn(
                        student_outputs=student_outputs,
                        teacher_batch={
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "h_start": h_start,
                            "h_target": h_target,
                            "velocity_target": velocity_target,
                            "teacher_logits": teacher_logits,
                            "labels": input_ids,
                        },
                        T=T,
                        model=self.model,
                        t=t,
                    )
            else:
                student_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=T,
                    return_dict=True,
                )
                total_loss, loss_metrics = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_batch={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "h_start": h_start,
                        "h_target": h_target,
                        "velocity_target": velocity_target,
                        "teacher_logits": teacher_logits,
                        "labels": input_ids,
                    },
                    T=T,
                    model=self.model,
                    t=t,
                )

        metrics = {
            "loss": loss_metrics.get("total_loss", total_loss.item()),
            "velocity_loss": loss_metrics.get("velocity_loss", 0.0),
            "kl_loss": loss_metrics.get("kl_loss", 0.0),
            "ce_loss": loss_metrics.get("ce_loss", 0.0),
            "T": T,
        }

        return metrics

    def validate(
        self, dataloader: Optional[Any] = None, max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        if dataloader is None:
            dataloader = self.val_dataloader
        if dataloader is None:
            logger.warning("No validation dataloader provided, skipping validation")
            return {}

        self.model.eval()
        all_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            metrics = self.val_step(batch)
            for key, value in metrics.items():
                all_metrics[key].append(value)

        avg_metrics = {
            f"val/{key}": sum(values) / len(values)
            for key, values in all_metrics.items()
        }

        # Log validation metrics to tensorboard
        if self.use_tensorboard and self.tensorboard_writer is not None:
            val_metrics_for_tb = {
                key.replace("val/", ""): value for key, value in avg_metrics.items()
            }
            self._log_to_tensorboard(val_metrics_for_tb, prefix="val")

            # Log combined behavior loss for validation
            kl_loss = avg_metrics.get("val/kl_loss", 0.0)
            ce_loss = avg_metrics.get("val/ce_loss", 0.0)
            if kl_loss > 0 or ce_loss > 0:
                combined_behavior_loss = kl_loss + ce_loss
                self.tensorboard_writer.add_scalar(
                    "val/combined_behavior_loss",
                    combined_behavior_loss,
                    self.global_step,
                )
                self.tensorboard_writer.flush()

        return avg_metrics

    def fit(self, max_epochs: Optional[int] = None) -> None:
        if self.train_dataloader is None:
            raise ValueError("No training dataloader provided")

        if max_epochs is None:
            max_epochs = self.train_loop_config.get("max_epochs", 3)

        val_check_interval = self.train_loop_config.get("val_check_interval", 250)
        checkpoint_dir = Path(
            self.train_loop_config.get("checkpoint_dir", "./checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting OnlineNoCache training for {max_epochs} epochs")
        logger.info(
            f"Initial state: global_step={self.global_step}, epoch={self.current_epoch}, accumulation_step={self.accumulation_step}"
        )

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            logger.info(f"Starting epoch {epoch + 1} - waiting for first batch...")

            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch_idx == 0:
                    logger.info(
                        f"Received first batch of epoch {epoch + 1}, starting training steps..."
                    )
                metrics = self.train_step(batch)

                if (
                    self.accumulation_step == 0
                    and self.global_step % self.log_every_n_steps == 0
                ):
                    log_str = f"Step {self.global_step}: "
                    log_str += f"loss={metrics['loss']:.4f}, "
                    log_str += (
                        f"velocity_loss={metrics.get('velocity_loss', 0.0):.4f}, "
                    )
                    log_str += f"kl_loss={metrics.get('kl_loss', 0.0):.4f}, "
                    log_str += f"ce_loss={metrics.get('ce_loss', 0.0):.4f}, "
                    log_str += f"grad_norm={metrics.get('grad_norm', 0.0):.4f}, "
                    log_str += f"T={metrics['T']}, "
                    log_str += f"lr={metrics['lr']:.6f}"
                    logger.info(log_str)

                if (
                    self.global_step % val_check_interval == 0
                    and self.global_step > 0
                    and not self._just_validated
                ):
                    val_metrics = self.validate()
                    self._just_validated = True
                    log_str = f"Validation at step {self.global_step}: "
                    for key, value in val_metrics.items():
                        log_str += f"{key}={value:.4f} "
                    logger.info(log_str)

                    monitor_value = val_metrics.get(
                        self.monitor_key,
                        val_metrics.get("val/loss", float("inf")),
                    )
                    if self._is_better(monitor_value):
                        self.best_val_metric = monitor_value
                        best_path = checkpoint_dir / "best.ckpt"
                        self.best_checkpoint_path = self.save_checkpoint(best_path)
                        logger.info(
                            f"New best checkpoint saved ({self.monitor_key}={monitor_value:.4f})"
                        )

            val_metrics = self.validate()
            logger.info(f"End of epoch {epoch + 1} validation: {val_metrics}")

            monitor_value = val_metrics.get(
                self.monitor_key,
                val_metrics.get("val/loss", float("inf")),
            )
            if self._is_better(monitor_value):
                self.best_val_metric = monitor_value
                best_path = checkpoint_dir / "best.ckpt"
                self.best_checkpoint_path = self.save_checkpoint(best_path)
                logger.info(
                    f"New best checkpoint saved at epoch end ({self.monitor_key}={monitor_value:.4f})"
                )

        logger.info("OnlineNoCache training complete!")

        # Close tensorboard writer
        if self.use_tensorboard and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            logger.info("Tensorboard writer closed")

    def close(self) -> None:
        """Close resources (tensorboard writer, etc.)."""
        if self.use_tensorboard and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            logger.info("Tensorboard writer closed")

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "accumulation_step": self.accumulation_step,
            "best_val_metric": self.best_val_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.accumulation_step = checkpoint.get("accumulation_step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", float("inf"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Loaded checkpoint from {path} at step {self.global_step}, "
            f"epoch {self.current_epoch}, accumulation_step {self.accumulation_step}"
        )

    def warm_start_from_checkpoint(self, path: Union[str, Path]) -> None:
        """Warm-start from checkpoint by loading model weights only.

        Unlike load_checkpoint(), this method does NOT restore:
        - optimizer state (fresh optimizer)
        - scheduler state (fresh scheduler)
        - global_step (remains 0)
        - current_epoch (remains 0)
        - accumulation_step (remains 0)

        This is useful when you want to initialize a training run from a pretrained
        checkpoint but with a fresh optimizer (e.g., for ablation studies).

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(
            f"Warm-started from checkpoint {path}: model weights loaded, "
            f"optimizer/scheduler/global_step remain fresh"
        )

        del checkpoint
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# Backward compatibility alias
OnlineNoCacheTrainer = Trainer

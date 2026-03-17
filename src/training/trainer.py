"""Raw PyTorch training loop for iterative midblock.

This module implements:
- Training step with gradient accumulation
- Validation step
- Checkpoint save/load
- Variable T sampling from config
- AMP/bf16 mixed precision support
- Logging of train and val metrics
"""

import json
import math
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Raw PyTorch trainer for iterative midblock training.
    
    This trainer supports:
    - Fixed-T training (T = depth of span)
    - Variable-T training with sampling from config distribution
    - AMP/bf16 mixed precision
    - Gradient accumulation
    - Checkpoint save/load with optimizer/scheduler state
    
    Args:
        model: The student model to train
        loss_fn: Loss function module
        config: Configuration dictionary
        device: Device to train on
        train_dataloader: Training dataloader
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
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.accumulation_step = 0
        
        # Extract config sections
        self.optimizer_config = config.get("optimizer", {})
        self.scheduler_config = config.get("scheduler", {})
        self.train_loop_config = config.get("train_loop", {})
        self.model_config = config.get("model", {})
        
        # Setup precision
        self._setup_precision()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup gradient scaler for AMP
        if self.use_amp:
            if hasattr(torch.amp, 'GradScaler'):
                # New API: torch.amp.GradScaler('cuda')
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                # Fallback to old API for older PyTorch versions
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Extract training settings
        self.accumulate_grad_batches = self.train_loop_config.get("accumulate_grad_batches", 1)
        self.grad_clip_norm = self.optimizer_config.get("grad_clip_norm", 1.0)
        
        # T sampling config
        self.train_T_values = self.model_config.get("train_T_values", [4])
        self.train_T_weights = self.model_config.get("train_T_weights", None)
        
        # Validation tracking
        self.best_val_metric = float('inf')
        self.best_checkpoint_path = None
        
        # Logging
        self.log_every_n_steps = self.train_loop_config.get("log_every_n_steps", 10)
        
        logger.info(f"Initialized Trainer on device: {self.device}")
        logger.info(f"  Precision: {self.precision}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  Gradient accumulation: {self.accumulate_grad_batches}")
        logger.info(f"  Train T values: {self.train_T_values}")
    
    def _setup_precision(self):
        """Setup precision settings for training."""
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
        """Create optimizer from config."""
        opt_name = self.optimizer_config.get("name", "adamw")
        lr = self.optimizer_config.get("learning_rate", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 0.01)
        betas = tuple(self.optimizer_config.get("betas", [0.9, 0.95]))
        eps = self.optimizer_config.get("eps", 1e-8)
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif opt_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif opt_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        logger.info(f"Created {opt_name} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler from config."""
        sched_name = self.scheduler_config.get("name", "cosine_with_warmup")
        
        if sched_name is None or sched_name.lower() == "none":
            return None
        
        warmup_steps = self.scheduler_config.get("warmup_steps", 0)
        
        if sched_name.lower() == "cosine_with_warmup":
            # Cosine annealing with warmup
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, self._get_max_steps() - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Created cosine_with_warmup scheduler with warmup_steps={warmup_steps}")
        elif sched_name.lower() == "linear_with_warmup":
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                # Linear decay
                progress = float(current_step - warmup_steps) / float(max(1, self._get_max_steps() - warmup_steps))
                return max(0.0, 1.0 - progress)
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Created linear_with_warmup scheduler with warmup_steps={warmup_steps}")
        elif sched_name.lower() == "constant":
            scheduler = None
        else:
            logger.warning(f"Unknown scheduler: {sched_name}, using constant LR")
            scheduler = None
        
        return scheduler
    
    def _get_max_steps(self) -> int:
        """Estimate maximum number of training steps."""
        max_epochs = self.train_loop_config.get("max_epochs", 3)
        
        if self.train_dataloader is not None:
            steps_per_epoch = len(self.train_dataloader)
            return max_epochs * steps_per_epoch
        
        # Fallback estimate
        return max_epochs * 1000
    
    def sample_T(self) -> int:
        """Sample a T value from the configured distribution.
        
        Returns:
            Number of refinement steps T
        """
        if len(self.train_T_values) == 1:
            return self.train_T_values[0]
        
        # Sample according to weights
        if self.train_T_weights is not None:
            return random.choices(
                self.train_T_values,
                weights=self.train_T_weights,
                k=1
            )[0]
        else:
            # Uniform sampling
            return random.choice(self.train_T_values)
    
    def train_step(self, batch: Dict[str, torch.Tensor], T: Optional[int] = None) -> Dict[str, float]:
        """Execute one training step.
        
        Args:
            batch: Batch of data with teacher targets
            T: Number of refinement steps (sampled if None)
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Sample T if not provided
        if T is None:
            T = self.sample_T()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with optional AMP
        if self.use_amp:
            # Use torch.amp.autocast for newer PyTorch, fallback to torch.cuda.amp.autocast
            autocast_context = torch.amp.autocast('cuda', dtype=self.amp_dtype) if hasattr(torch, 'amp') else torch.cuda.amp.autocast(dtype=self.amp_dtype)
            with autocast_context:
                student_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    num_steps=T,
                    return_dict=True,
                )
                
                # Compute loss
                total_loss, loss_metrics = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_batch=batch,
                    T=T,
                )
        else:
            student_outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                num_steps=T,
                return_dict=True,
            )
            
            # Compute loss
            total_loss, loss_metrics = self.loss_fn(
                student_outputs=student_outputs,
                teacher_batch=batch,
                T=T,
            )
        
        # Scale loss for gradient accumulation
        if self.accumulate_grad_batches > 1:
            total_loss = total_loss / self.accumulate_grad_batches
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        self.accumulation_step += 1
        
        # Optimizer step if accumulation is complete
        if self.accumulation_step >= self.accumulate_grad_batches:
            if self.grad_clip_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )
            
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.accumulation_step = 0
            self.global_step += 1
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Prepare metrics
        metrics = {
            "loss": loss_metrics.get("total_loss", total_loss.item()),
            "endpoint_loss": loss_metrics.get("endpoint_loss", 0.0),
            "trajectory_loss": loss_metrics.get("trajectory_loss", 0.0),
            "kl_loss": loss_metrics.get("kl_loss", 0.0),
            "ce_loss": loss_metrics.get("ce_loss", 0.0),
            "T": T,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        
        return metrics
    
    def val_step(self, batch: Dict[str, torch.Tensor], T: Optional[int] = None) -> Dict[str, float]:
        """Execute one validation step.
        
        Args:
            batch: Batch of data with teacher targets
            T: Number of refinement steps (sampled if None)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Sample T if not provided
        if T is None:
            T = self.sample_T()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            # Forward pass with optional AMP
            if self.use_amp:
                # Use torch.amp.autocast for newer PyTorch, fallback to torch.cuda.amp.autocast
                autocast_context = torch.amp.autocast('cuda', dtype=self.amp_dtype) if hasattr(torch, 'amp') else torch.cuda.amp.autocast(dtype=self.amp_dtype)
                with autocast_context:
                    student_outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        num_steps=T,
                        return_dict=True,
                    )
                    
                    # Compute loss
                    total_loss, loss_metrics = self.loss_fn(
                        student_outputs=student_outputs,
                        teacher_batch=batch,
                        T=T,
                    )
            else:
                student_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    num_steps=T,
                    return_dict=True,
                )
                
                # Compute loss
                total_loss, loss_metrics = self.loss_fn(
                    student_outputs=student_outputs,
                    teacher_batch=batch,
                    T=T,
                )
        
        # Prepare metrics
        metrics = {
            "loss": loss_metrics.get("total_loss", total_loss.item()),
            "endpoint_loss": loss_metrics.get("endpoint_loss", 0.0),
            "trajectory_loss": loss_metrics.get("trajectory_loss", 0.0),
            "kl_loss": loss_metrics.get("kl_loss", 0.0),
            "ce_loss": loss_metrics.get("ce_loss", 0.0),
            "T": T,
        }
        
        return metrics
    
    def validate(self, dataloader: Optional[Any] = None, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run validation over the validation dataloader.
        
        Args:
            dataloader: Validation dataloader (uses self.val_dataloader if None)
            max_batches: Maximum number of batches to validate on
            
        Returns:
            Dictionary of averaged metrics
        """
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
        
        # Average metrics
        avg_metrics = {f"val/{key}": sum(values) / len(values) 
                      for key, values in all_metrics.items()}
        
        return avg_metrics
    
    def fit(self, max_epochs: Optional[int] = None) -> None:
        """Run full training loop.
        
        Args:
            max_epochs: Maximum number of epochs (uses config if None)
        """
        if self.train_dataloader is None:
            raise ValueError("No training dataloader provided")
        
        if max_epochs is None:
            max_epochs = self.train_loop_config.get("max_epochs", 3)
        
        val_check_interval = self.train_loop_config.get("val_check_interval", 250)
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    log_str = f"Step {self.global_step}: "
                    log_str += f"loss={metrics['loss']:.4f}, "
                    log_str += f"T={metrics['T']}, "
                    log_str += f"lr={metrics['lr']:.6f}"
                    logger.info(log_str)
                
                # Validation
                if self.global_step % val_check_interval == 0 and self.global_step > 0:
                    val_metrics = self.validate()
                    log_str = f"Validation at step {self.global_step}: "
                    for key, value in val_metrics.items():
                        log_str += f"{key}={value:.4f} "
                    logger.info(log_str)
                    
                    # Save best checkpoint
                    monitor_key = self.config.get("logging", {}).get("monitor", "val/loss")
                    monitor_value = val_metrics.get(monitor_key, val_metrics.get("val/loss", float('inf')))
                    
                    if monitor_value < self.best_val_metric:
                        self.best_val_metric = monitor_value
                        checkpoint_dir = Path(self.train_loop_config.get("checkpoint_dir", "./checkpoints"))
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = checkpoint_dir / "best.ckpt"
                        self.save_checkpoint(checkpoint_path)
                        logger.info(f"Saved best checkpoint to {checkpoint_path}")
            
            # End of epoch validation
            val_metrics = self.validate()
            logger.info(f"End of epoch {epoch + 1} validation: {val_metrics}")
        
        logger.info("Training complete!")
    
    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
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
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", float('inf'))
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}, epoch {self.current_epoch}")
    
    def save_midblock_only(self, path: Union[str, Path]) -> Path:
        """Save only the midblock weights (for efficient checkpointing).
        
        Args:
            path: Path to save midblock
            
        Returns:
            Path to saved midblock
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only midblock state
        if hasattr(self.model, 'midblock') and self.model.midblock is not None:
            torch.save(self.model.midblock.state_dict(), path)
            logger.info(f"Saved midblock to {path}")
        else:
            logger.warning("Model has no midblock to save")
        
        return path

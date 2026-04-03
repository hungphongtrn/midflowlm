#!/usr/bin/env python3
"""Training script for v0 iterative midblock.

This script runs the full training loop for the Qwen iterative midblock student.
It supports:
- Loading from teacher cache
- Fixed-T or variable-T training
- Resume from checkpoint
- Fast dev run for smoke testing
- Limited batch runs for quick validation

Usage:
    python scripts/train_v0.py --config configs/v0_onemotif.yaml
    python scripts/train_v0.py --config configs/v0_onemotif.yaml --fast-dev-run
    python scripts/train_v0.py --config configs/v0_onemotif.yaml --resume-from-checkpoint ./checkpoints/best.ckpt
    python scripts/train_v0.py --config configs/v0_onemotif.yaml --limit-train-batches 10 --limit-val-batches 5
"""

import argparse
import logging
import random
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.training.losses import DistillationLoss
from src.training.trainer import Trainer
from src.training.data import (
    create_cache_dataloader,
    get_cache_info,
    validate_cache_compatibility,
)
from src.training.teacher_state import (
    resolve_teacher_state_mode,
    validate_teacher_state_config,
    TeacherStateMode,
)
from src.data.dataset_factory import get_experiment_dataloaders


def format_train_batch_log(batch_idx: int, metrics: Dict[str, float]) -> str:
    """Format a training batch log line with loss components."""
    return (
        f"  Batch {batch_idx}: "
        f"loss={metrics.get('loss', 0.0):.4f}, "
        f"velocity_loss={metrics.get('velocity_loss', 0.0):.4f}, "
        f"kl_loss={metrics.get('kl_loss', 0.0):.4f}, "
        f"ce_loss={metrics.get('ce_loss', 0.0):.4f}, "
        f"T={metrics.get('T', 0)}"
    )


def get_teacher_logits_source(config: Dict[str, Any]) -> str:
    """Resolve how KL teacher logits should be sourced for this run."""
    loss_config = config.get("loss", {})
    kl_weight = float(loss_config.get("kl_weight", 0.0))
    return loss_config.get(
        "teacher_logits_source",
        "cache" if kl_weight > 0.0 else "none",
    )


class StructuredTrainingLogger:
    """Structured JSON logger for AI debugging of training processes.

    This logger creates a machine-readable log file with events, metrics,
    and traces that enable AI agents to debug training issues effectively.

    Log format: JSON Lines (.jsonl) with one JSON object per line
    Each log entry contains:
    - timestamp: ISO format timestamp
    - event_type: Type of event (config, step, validation, checkpoint, error, system, summary)
    - step: Global training step (if applicable)
    - epoch: Epoch number (if applicable)
    - data: Event-specific data
    """

    def __init__(self, log_path: Path):
        """Initialize the structured logger.

        Args:
            log_path: Path to the JSONL log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._step_history: list = []
        self._error_count = 0

    def _write_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """Write a structured event to the log file.

        Args:
            event_type: Type of event
            data: Event data dictionary
            step: Optional global step number
            epoch: Optional epoch number
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        if step is not None:
            event["step"] = step
        if epoch is not None:
            event["epoch"] = epoch

        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def log_config(
        self, config: Dict[str, Any], cli_args: Optional[Dict[str, Any]] = None
    ):
        """Log configuration at the start of training.

        Args:
            config: Training configuration dictionary
            cli_args: Optional CLI arguments dictionary
        """
        event_data = {
            "config": config,
            "cli_args": cli_args or {},
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }
        self._write_event("config", event_data)

    def log_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        batch_info: Optional[Dict[str, Any]] = None,
    ):
        """Log a training step.

        Args:
            step: Global step number
            epoch: Current epoch
            metrics: Dictionary of metrics (loss, lr, etc.)
            batch_info: Optional batch information (size, T value, etc.)
        """
        event_data = {"metrics": metrics, "batch_info": batch_info or {}}
        self._write_event("step", event_data, step=step, epoch=epoch)
        self._step_history.append({"step": step, "epoch": epoch, "metrics": metrics})

    def log_validation(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        perplexity: Optional[float] = None,
    ):
        """Log validation results.

        Args:
            step: Global step number
            epoch: Current epoch
            metrics: Validation metrics dictionary
            perplexity: Optional perplexity value
        """
        event_data = {"metrics": metrics, "perplexity": perplexity}
        self._write_event("validation", event_data, step=step, epoch=epoch)

    def log_checkpoint(
        self,
        step: int,
        epoch: int,
        checkpoint_path: str,
        is_best: bool = False,
        checkpoint_type: str = "full",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log checkpoint save event.

        Args:
            step: Global step number
            epoch: Current epoch
            checkpoint_path: Path to saved checkpoint
            is_best: Whether this is the best checkpoint
            checkpoint_type: Type of checkpoint (full, midblock_only, etc.)
            metadata: Optional checkpoint metadata
        """
        event_data = {
            "checkpoint_path": str(checkpoint_path),
            "is_best": is_best,
            "checkpoint_type": checkpoint_type,
            "metadata": metadata or {},
        }
        self._write_event("checkpoint", event_data, step=step, epoch=epoch)

    def log_error(
        self,
        step: Optional[int],
        epoch: Optional[int],
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log an error event.

        Args:
            step: Global step number (if applicable)
            epoch: Current epoch (if applicable)
            error_type: Type of error (exception, nan_loss, etc.)
            error_message: Error message
            traceback: Optional traceback string
            context: Optional context information
        """
        self._error_count += 1
        event_data = {
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback,
            "context": context or {},
            "error_count": self._error_count,
        }
        self._write_event("error", event_data, step=step, epoch=epoch)

    def log_system(
        self,
        step: Optional[int],
        epoch: Optional[int],
        memory_stats: Optional[Dict[str, Any]] = None,
        gpu_stats: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, float]] = None,
    ):
        """Log system state.

        Args:
            step: Global step number
            epoch: Current epoch
            memory_stats: Memory statistics
            gpu_stats: GPU statistics
            timing: Timing information (step_time, etc.)
        """
        event_data = {
            "memory_stats": memory_stats or {},
            "gpu_stats": gpu_stats or {},
            "timing": timing or {},
        }
        self._write_event("system", event_data, step=step, epoch=epoch)

    def log_summary(
        self,
        total_steps: int,
        total_epochs: int,
        final_metrics: Dict[str, Any],
        best_val_metric: Optional[float] = None,
        training_duration_seconds: Optional[float] = None,
    ):
        """Log training summary at the end.

        Args:
            total_steps: Total training steps
            total_epochs: Total epochs completed
            final_metrics: Final metrics dictionary
            best_val_metric: Best validation metric achieved
            training_duration_seconds: Total training duration
        """
        event_data = {
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "final_metrics": final_metrics,
            "best_val_metric": best_val_metric,
            "training_duration_seconds": training_duration_seconds,
            "error_count": self._error_count,
            "step_count": len(self._step_history),
        }
        self._write_event("summary", event_data)

    def log_model_info(
        self,
        model_summary: Dict[str, Any],
        param_summary: Dict[str, int],
        trainable_params: int,
        frozen_params: int,
    ):
        """Log model information.

        Args:
            model_summary: Model architecture summary
            param_summary: Parameter count summary
            trainable_params: Number of trainable parameters
            frozen_params: Number of frozen parameters
        """
        event_data = {
            "model_summary": model_summary,
            "param_summary": param_summary,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "total_params": trainable_params + frozen_params,
        }
        self._write_event("model_info", event_data)

    def log_data_info(
        self,
        train_batches: int,
        val_batches: int,
        cache_info: Optional[Dict[str, Any]] = None,
    ):
        """Log data information.

        Args:
            train_batches: Number of training batches
            val_batches: Number of validation batches
            cache_info: Optional cache information
        """
        event_data = {
            "train_batches": train_batches,
            "val_batches": val_batches,
            "cache_info": cache_info or {},
        }
        self._write_event("data_info", event_data)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> tuple:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for structured JSON logs
        experiment_name: Experiment name for log file naming

    Returns:
        Tuple of (logger, structured_logger)
    """
    # Setup console/file logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup structured JSON logger for AI debugging
    structured_logger = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_part = experiment_name if experiment_name else "unknown"
        log_file = log_dir_path / f"train_{experiment_part}_{timestamp}.jsonl"

        structured_logger = StructuredTrainingLogger(log_file)
        logging.info(f"Structured logging to: {log_file}")

    return logging.getLogger(__name__), structured_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_timestamped_experiment_name(base_name: str) -> str:
    """Generate timestamped experiment name.

    Args:
        base_name: Base experiment name (e.g., "v0_qwen_iterative_midblock")

    Returns:
        Timestamped name like "00-21-19-03-2026-v0_qwen_iterative_midblock"
    """
    timestamp = datetime.now().strftime("%H-%M-%d-%m-%Y")
    return f"{timestamp}-{base_name}"


def update_config_with_timestamp(config: dict) -> dict:
    """Update config with timestamped experiment name.

    Modifies checkpoint_dir and log_dir to include timestamp.

    Args:
        config: Configuration dictionary

    Returns:
        Updated config with timestamped paths
    """
    base_name = config.get("experiment_name", "experiment")
    timestamped_name = get_timestamped_experiment_name(base_name)

    # Update checkpoint_dir
    train_loop = config.get("train_loop", {})
    checkpoint_dir = train_loop.get("checkpoint_dir", "./checkpoints")
    # Replace base_name with timestamped_name in the path
    checkpoint_dir = checkpoint_dir.replace(base_name, timestamped_name)
    train_loop["checkpoint_dir"] = checkpoint_dir
    config["train_loop"] = train_loop

    # Update log_dir
    logging_config = config.get("logging", {})
    log_dir = logging_config.get("log_dir", "./logs")
    log_dir = log_dir.replace(base_name, timestamped_name)
    logging_config["log_dir"] = log_dir
    config["logging"] = logging_config

    # Update experiment_name
    config["experiment_name"] = timestamped_name

    return config


def create_student_model(config: dict, device: str) -> FrozenQwenStudent:
    """Create the student model from config.

    Args:
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        FrozenQwenStudent instance
    """
    model_config = config["model"]
    replacement_config = config["replacement_model"]

    # Determine dtype from precision setting
    train_loop_config = config.get("train_loop", {})
    precision = train_loop_config.get("precision", "fp32")

    if precision == "bf16-mixed":
        dtype = torch.bfloat16
    elif precision == "fp16-mixed":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=dtype,
        bypass_mode=False,
    )

    return model


def create_teacher_model(config: dict) -> FrozenQwenStudent:
    """Create the frozen original model used for online KL logits."""
    model_config = config["model"]
    replacement_config = config["replacement_model"]

    train_loop_config = config.get("train_loop", {})
    precision = train_loop_config.get("precision", "fp32")

    if precision == "bf16-mixed":
        dtype = torch.bfloat16
    elif precision == "fp16-mixed":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device="cpu",
        dtype=dtype,
        bypass_mode=True,
    )


def create_loss_function(config: dict) -> DistillationLoss:
    """Create the loss function from config.

    Args:
        config: Configuration dictionary

    Returns:
        DistillationLoss instance
    """
    loss_fn = DistillationLoss.from_config(config)
    return loss_fn


def create_dataloaders(config: dict, cache_dir: str) -> tuple:
    """Create train and validation dataloaders.

    Routes to the appropriate dataloader based on teacher_state.mode:
    - offline_cache: Uses create_cache_dataloader (cache-based)
    - online_no_cache: Uses get_experiment_dataloaders (token dataset)
    - online_write_through_cache: Uses get_experiment_dataloaders (token dataset)

    Args:
        config: Configuration dictionary
        cache_dir: Directory containing cache files (used for offline_cache mode)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    teacher_state_mode = resolve_teacher_state_mode(config)

    if teacher_state_mode == TeacherStateMode.OFFLINE_CACHE.value:
        validate_cache_compatibility(config, cache_dir)

        data_config = config.get("data", {})
        seed = config.get("seed", 1337)

        batch_size = data_config.get("batch_size", 8)
        num_workers = data_config.get("num_workers", 0)
        pin_memory = data_config.get("pin_memory", False)

        train_dataloader = create_cache_dataloader(
            cache_dir=cache_dir,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            split="train",
        )

        val_dataloader = create_cache_dataloader(
            cache_dir=cache_dir,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            split="val",
        )

        return train_dataloader, val_dataloader

    elif teacher_state_mode in (
        TeacherStateMode.ONLINE_NO_CACHE.value,
        TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE.value,
    ):
        dataloaders = _create_online_dataloaders(config)
        return dataloaders["train"], dataloaders["val"]

    else:
        raise ValueError(f"Unknown teacher_state mode: {teacher_state_mode}")


def _create_online_dataloaders(config: dict) -> dict:
    """Create dataloaders for online teacher_state modes.

    Reuses get_experiment_dataloaders from dataset_factory for token dataset loading.
    The config is adapted to namespace-compatible format expected by dataset_factory.

    Args:
        config: Configuration dictionary with data.loader and data.mixture_components

    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    from transformers import AutoTokenizer

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    batch_size = data_config.get("batch_size", 8)

    class ConfigAdapter:
        def __init__(self, config_dict):
            self._config = config_dict

        def __getattr__(self, name):
            if name == "data":
                return DataConfigAdapter(self._config.get("data", {}))
            if name == "model":
                return ModelConfigAdapter(self._config.get("model", {}))
            return self._config.get(name)

    class DataConfigAdapter:
        def __init__(self, data_dict):
            self._data = data_dict
            for k, v in data_dict.items():
                setattr(self, k, v)

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return self._data.get(name)

    class ModelConfigAdapter:
        def __init__(self, model_dict):
            self._model = model_dict
            for k, v in model_dict.items():
                setattr(self, k, v)

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return self._model.get(name)

    adapted_config = ConfigAdapter(config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloaders = get_experiment_dataloaders(
        config=adapted_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    return dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train v0 iterative midblock")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to train on (cuda/cpu)"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast dev loop (1 train, 1 val step)",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit number of validation batches",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--no-structured-logs",
        action="store_true",
        help="Disable structured JSON logging",
    )

    args = parser.parse_args()

    # Load config first to get log_dir
    config = load_config(args.config)

    # Add timestamp to experiment name
    config = update_config_with_timestamp(config)

    # Determine log directory from config
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "./logs")
    experiment_name = config.get("experiment_name", "unknown")

    # Setup logging
    logger, structured_logger = setup_logging(
        args.log_level,
        log_dir=None if args.no_structured_logs else log_dir,
        experiment_name=experiment_name,
    )

    # Log configuration
    logger.info(f"Loading config from {args.config}")
    logger.info(f"Experiment name: {config['experiment_name']}")

    if structured_logger:
        structured_logger.log_config(config=config, cli_args=vars(args))

    # Set seed
    seed = config.get("seed", 1337)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Resolve and validate teacher_state mode
    teacher_state_mode = resolve_teacher_state_mode(config)
    logger.info(f"Teacher state mode: {teacher_state_mode}")

    try:
        validate_teacher_state_config(config)
    except ValueError as e:
        error_msg = f"Teacher state config validation failed: {str(e)}"
        logger.error(error_msg)
        if structured_logger:
            structured_logger.log_error(
                step=None,
                epoch=None,
                error_type="teacher_state_validation",
                error_message=error_msg,
                context={"teacher_state_mode": teacher_state_mode, "config": config},
            )
        sys.exit(1)

    # Get cache directory (only relevant for offline_cache mode)
    cache_config = config.get("teacher_cache", {})
    cache_dir = cache_config.get("cache_dir", "./cache")

    cache_info_dict = {}
    if teacher_state_mode == TeacherStateMode.OFFLINE_CACHE.value:
        if not Path(cache_dir).exists():
            error_msg = f"Cache directory not found: {cache_dir}"
            logger.error(error_msg)
            logger.error("Please run scripts/build_teacher_cache.py first")
            if structured_logger:
                structured_logger.log_error(
                    step=None,
                    epoch=None,
                    error_type="missing_cache",
                    error_message=error_msg,
                    context={"cache_dir": cache_dir},
                )
            sys.exit(1)

        try:
            cache_info_dict = get_cache_info(cache_dir)
            logger.info(f"Cache info: {cache_info_dict}")
        except Exception as e:
            logger.warning(f"Could not load cache info: {e}")
    elif teacher_state_mode == TeacherStateMode.ONLINE_NO_CACHE.value:
        logger.info(
            "Online no-cache mode: token-batch dataloaders enabled; live teacher extraction remains pending"
        )
    elif teacher_state_mode == TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE.value:
        logger.info(
            "Online write-through cache mode: token-batch dataloaders enabled; live extraction and cache writing remain pending"
        )

    # Create model
    logger.info("Creating student model...")
    try:
        model = create_student_model(config, device)
    except Exception as e:
        error_msg = f"Failed to create model: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=None,
                epoch=None,
                error_type="model_creation",
                error_message=error_msg,
                traceback=tb,
                context={"config": config.get("model", {})},
            )
        raise

    teacher_model = None
    teacher_logits_source = get_teacher_logits_source(config)
    if teacher_logits_source == "online":
        logger.info("Creating frozen teacher model for online KL logits...")
        try:
            teacher_model = create_teacher_model(config)
        except Exception as e:
            error_msg = f"Failed to create teacher model: {str(e)}"
            logger.error(error_msg)
            import traceback

            tb = traceback.format_exc()
            if structured_logger:
                structured_logger.log_error(
                    step=None,
                    epoch=None,
                    error_type="teacher_model_creation",
                    error_message=error_msg,
                    traceback=tb,
                    context={"config": config.get("model", {})},
                )
            raise

    param_summary = model.get_parameter_summary()
    logger.info(f"Model parameters: {param_summary}")

    if structured_logger:
        # Calculate trainable vs frozen params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        structured_logger.log_model_info(
            model_summary={
                "name": config.get("model", {}).get("name", "unknown"),
                "start_layer": config.get("replacement_model", {}).get("start_layer"),
                "end_layer": config.get("replacement_model", {}).get("end_layer"),
                "max_steps_T": config.get("model", {}).get("max_steps_T"),
                "device": device,
                "teacher_logits_source": teacher_logits_source,
                "uses_online_teacher_logits": teacher_logits_source == "online",
            },
            param_summary=param_summary,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
        )

    # Create loss function
    logger.info("Creating loss function...")
    try:
        loss_fn = create_loss_function(config)
        loss_fn = loss_fn.to(device)
    except Exception as e:
        error_msg = f"Failed to create loss function: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=None,
                epoch=None,
                error_type="loss_creation",
                error_message=error_msg,
                traceback=tb,
                context={"config": config.get("loss", {})},
            )
        raise

    # Create dataloaders - route based on teacher_state mode
    logger.info("Creating dataloaders...")
    try:
        train_dataloader, val_dataloader = create_dataloaders(config, cache_dir)
    except Exception as e:
        error_msg = f"Failed to create dataloaders: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=None,
                epoch=None,
                error_type="dataloader_creation",
                error_message=error_msg,
                traceback=tb,
                context={
                    "data_config": config.get("data", {}),
                    "cache_dir": cache_dir,
                    "teacher_state_mode": teacher_state_mode,
                },
            )
        raise

    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")

    if structured_logger:
        structured_logger.log_data_info(
            train_batches=len(train_dataloader),
            val_batches=len(val_dataloader),
            cache_info=cache_info_dict,
        )

    # Create trainer
    logger.info("Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            config=config,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            teacher_model=teacher_model,
        )
    except Exception as e:
        error_msg = f"Failed to create trainer: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=None,
                epoch=None,
                error_type="trainer_creation",
                error_message=error_msg,
                traceback=tb,
                context={"config": config},
            )
        raise

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        try:
            trainer.load_checkpoint(args.resume_from_checkpoint)
            if structured_logger:
                structured_logger.log_checkpoint(
                    step=trainer.global_step,
                    epoch=trainer.current_epoch,
                    checkpoint_path=args.resume_from_checkpoint,
                    checkpoint_type="resume",
                    metadata={"source": args.resume_from_checkpoint},
                )
        except Exception as e:
            error_msg = f"Failed to resume from checkpoint: {str(e)}"
            logger.error(error_msg)
            import traceback

            tb = traceback.format_exc()
            if structured_logger:
                structured_logger.log_error(
                    step=None,
                    epoch=None,
                    error_type="checkpoint_resume",
                    error_message=error_msg,
                    traceback=tb,
                    context={"checkpoint_path": args.resume_from_checkpoint},
                )
            raise

    # Fast dev run
    if args.fast_dev_run:
        logger.info("Running fast dev loop (1 train step, 1 val step)...")

        try:
            # One train step
            train_batch = next(iter(train_dataloader))
            train_metrics = trainer.train_step(train_batch)
            logger.info(f"Train step metrics: {train_metrics}")

            if structured_logger:
                structured_logger.log_step(
                    step=trainer.global_step,
                    epoch=0,
                    metrics=train_metrics,
                    batch_info={
                        "batch_size": train_batch.get(
                            "input_ids", torch.tensor([])
                        ).shape[0]
                    },
                )

            # One val step
            val_batch = next(iter(val_dataloader))
            val_metrics = trainer.val_step(val_batch)
            logger.info(f"Val step metrics: {val_metrics}")

            if structured_logger:
                structured_logger.log_validation(
                    step=trainer.global_step, epoch=0, metrics=val_metrics
                )

            logger.info("Fast dev run complete!")
        except Exception as e:
            error_msg = f"Fast dev run failed: {str(e)}"
            logger.error(error_msg)
            import traceback

            tb = traceback.format_exc()
            if structured_logger:
                structured_logger.log_error(
                    step=trainer.global_step,
                    epoch=trainer.current_epoch,
                    error_type="fast_dev_run",
                    error_message=error_msg,
                    traceback=tb,
                )
            raise
        return

    # Track training start time for summary
    import time

    training_start_time = time.time()

    # Limited batch run
    if args.limit_train_batches or args.limit_val_batches:
        logger.info("Running limited batch training...")

        max_epochs = config.get("train_loop", {}).get("max_epochs", 1)

        # Compute baseline perplexity before training
        try:
            baseline_ppl = trainer.compute_baseline_perplexity()
        except Exception as e:
            error_msg = f"Failed to compute baseline perplexity: {str(e)}"
            logger.error(error_msg)
            import traceback

            tb = traceback.format_exc()
            if structured_logger:
                structured_logger.log_error(
                    step=trainer.global_step,
                    epoch=0,
                    error_type="baseline_perplexity",
                    error_message=error_msg,
                    traceback=tb,
                )
            raise

        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")

            # Training with limited batches
            for batch_idx, batch in enumerate(train_dataloader):
                if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                    break

                try:
                    metrics = trainer.train_step(batch)

                    if batch_idx % 10 == 0:
                        logger.info(format_train_batch_log(batch_idx, metrics))

                    if structured_logger and batch_idx % 10 == 0:
                        structured_logger.log_step(
                            step=trainer.global_step,
                            epoch=epoch,
                            metrics=metrics,
                            batch_info={
                                "batch_idx": batch_idx,
                                "batch_size": batch.get(
                                    "input_ids", torch.tensor([])
                                ).shape[0],
                            },
                        )

                        # Log system stats periodically
                        if torch.cuda.is_available():
                            structured_logger.log_system(
                                step=trainer.global_step,
                                epoch=epoch,
                                gpu_stats={
                                    "allocated_mb": torch.cuda.memory_allocated()
                                    / 1024**2,
                                    "reserved_mb": torch.cuda.memory_reserved()
                                    / 1024**2,
                                },
                            )

                except Exception as e:
                    error_msg = f"Training step failed at batch {batch_idx}: {str(e)}"
                    logger.error(error_msg)
                    import traceback

                    tb = traceback.format_exc()
                    if structured_logger:
                        structured_logger.log_error(
                            step=trainer.global_step,
                            epoch=epoch,
                            error_type="training_step",
                            error_message=error_msg,
                            traceback=tb,
                            context={"batch_idx": batch_idx},
                        )
                    raise

            # Validation with limited batches
            logger.info("Running validation...")
            try:
                val_metrics = trainer.validate(max_batches=args.limit_val_batches)
                logger.info(f"Validation metrics: {val_metrics}")
            except Exception as e:
                error_msg = f"Validation failed: {str(e)}"
                logger.error(error_msg)
                import traceback

                tb = traceback.format_exc()
                if structured_logger:
                    structured_logger.log_error(
                        step=trainer.global_step,
                        epoch=epoch,
                        error_type="validation",
                        error_message=error_msg,
                        traceback=tb,
                    )
                raise

            # Compute perplexity after epoch
            try:
                epoch_ppl = trainer.compute_epoch_perplexity(epoch + 1)
                logger.info(f"=== PPL CHANGE: {epoch_ppl - baseline_ppl:+.2f} ===")

                if structured_logger:
                    structured_logger.log_validation(
                        step=trainer.global_step,
                        epoch=epoch,
                        metrics=val_metrics,
                        perplexity=epoch_ppl,
                    )
            except Exception as e:
                error_msg = f"Failed to compute epoch perplexity: {str(e)}"
                logger.error(error_msg)
                import traceback

                tb = traceback.format_exc()
                if structured_logger:
                    structured_logger.log_error(
                        step=trainer.global_step,
                        epoch=epoch,
                        error_type="epoch_perplexity",
                        error_message=error_msg,
                        traceback=tb,
                    )
                raise

            # Save checkpoint
            checkpoint_dir = Path(
                config.get("train_loop", {}).get("checkpoint_dir", "./checkpoints")
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.ckpt"

            try:
                trainer.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                if structured_logger:
                    structured_logger.log_checkpoint(
                        step=trainer.global_step,
                        epoch=epoch,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_type="epoch",
                        metadata={"epoch": epoch + 1},
                    )
            except Exception as e:
                error_msg = f"Failed to save checkpoint: {str(e)}"
                logger.error(error_msg)
                import traceback

                tb = traceback.format_exc()
                if structured_logger:
                    structured_logger.log_error(
                        step=trainer.global_step,
                        epoch=epoch,
                        error_type="checkpoint_save",
                        error_message=error_msg,
                        traceback=tb,
                        context={"checkpoint_path": str(checkpoint_path)},
                    )
                raise

        training_duration = time.time() - training_start_time

        if structured_logger:
            structured_logger.log_summary(
                total_steps=trainer.global_step,
                total_epochs=max_epochs,
                final_metrics={
                    "last_val_metrics": val_metrics,
                    "final_ppl": epoch_ppl if "epoch_ppl" in locals() else None,
                    "baseline_ppl": baseline_ppl
                    if "baseline_ppl" in locals()
                    else None,
                },
                best_val_metric=trainer.best_val_metric,
                training_duration_seconds=training_duration,
            )

        logger.info("Limited batch training complete!")
        return

    # Full training
    logger.info("Starting full training...")

    try:
        # Compute baseline perplexity
        baseline_ppl = trainer.compute_baseline_perplexity()

        # Run training - we need to hook into the fit() method for detailed logging
        # For now, capture what's available after fit()
        trainer.fit()

        # Log system info after training
        if structured_logger:
            final_epoch = trainer.current_epoch
            final_step = trainer.global_step

            # Log final validation
            val_metrics = trainer.validate()
            epoch_ppl = trainer.compute_epoch_perplexity(final_epoch)

            structured_logger.log_validation(
                step=final_step,
                epoch=final_epoch,
                metrics=val_metrics,
                perplexity=epoch_ppl,
            )

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                error_type="training",
                error_message=error_msg,
                traceback=tb,
            )

            # Log partial summary
            training_duration = time.time() - training_start_time
            structured_logger.log_summary(
                total_steps=trainer.global_step,
                total_epochs=trainer.current_epoch,
                final_metrics={},
                best_val_metric=trainer.best_val_metric,
                training_duration_seconds=training_duration,
            )
        raise

    # Save final checkpoint
    checkpoint_dir = Path(
        config.get("train_loop", {}).get("checkpoint_dir", "./checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "final.ckpt"

    try:
        trainer.save_checkpoint(final_path)
        logger.info(f"Saved final checkpoint to {final_path}")

        if structured_logger:
            structured_logger.log_checkpoint(
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                checkpoint_path=str(final_path),
                checkpoint_type="final",
            )

            # Log final summary
            training_duration = time.time() - training_start_time
            structured_logger.log_summary(
                total_steps=trainer.global_step,
                total_epochs=trainer.current_epoch,
                final_metrics={
                    "best_val_metric": trainer.best_val_metric,
                    "epoch_perplexities": trainer.epoch_ppls
                    if hasattr(trainer, "epoch_ppls")
                    else {},
                },
                best_val_metric=trainer.best_val_metric,
                training_duration_seconds=training_duration,
            )
    except Exception as e:
        error_msg = f"Failed to save final checkpoint: {str(e)}"
        logger.error(error_msg)
        import traceback

        tb = traceback.format_exc()
        if structured_logger:
            structured_logger.log_error(
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                error_type="final_checkpoint_save",
                error_message=error_msg,
                traceback=tb,
            )
        raise

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

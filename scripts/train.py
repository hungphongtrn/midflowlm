#!/usr/bin/env python3
"""PRIMARY training script for iterative midblock training with mixed corpus support.

This is the DEFAULT training script for midflowlm. It uses online calculation
instead of teacher-state caching, making it disk-space friendly and simpler to use.

Teacher targets (h_start, h_target, velocity_target, teacher_logits) are extracted
on-the-fly via model.extract_teacher_targets() in one no-grad forward pass.

Supports both TinyStories and mixed corpus datasets via dataset_factory.

Usage:
    # Default: online calculation mode (recommended)
    python scripts/train.py --config configs/v0_online_no_cache_mixed_ce_kl.yaml

    # Fast dev run for testing
    python scripts/train.py --config configs/v0_online_no_cache_mixed_ce_kl.yaml --fast-dev-run

    # Limited batch run
    python scripts/train.py --config configs/v0_online_no_cache_mixed_ce_kl.yaml --limit-train-batches 10

    # Resume from checkpoint
    python scripts/train.py --config configs/v0_online_no_cache_mixed_ce_kl.yaml --resume-from-checkpoint ./checkpoints/best.ckpt

Note: Cache-based training is available but deprecated. Use CachedTrainer from
src.training.cached_trainer if you absolutely need cache support.
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional
from types import SimpleNamespace

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer

# Force unbuffered output for live logging
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

# SDPA (Scaled Dot Product Attention) is enabled by default for better performance
# Only disable if experiencing CUDA hangs during training
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.training.trainer import Trainer
from src.training.losses import DistillationLoss
from src.data.dataset_factory import get_experiment_dataloaders


def setup_logging(
    log_level: str = "INFO", log_dir: Optional[str] = None
) -> logging.Logger:
    """Setup logging with both console and file handlers.

    Configures the root logger so that all modules (including trainers)
    automatically log to both console and file with live streaming.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files. If None, only console logging is used.

    Returns:
        Configured logger instance
    """
    # Configure the root logger so all child loggers inherit the settings
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers on root logger
    root_logger.handlers = []

    # Console handler with explicit flush for live output
    class LineBufferedStreamHandler(logging.StreamHandler):
        """StreamHandler that flushes after every emit for live output."""

        def emit(self, record):
            super().emit(record)
            self.flush()

    console_handler = LineBufferedStreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_dir is provided
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "train.log"

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Also log to a separate error-only file
        error_file = log_path / "error.log"
        error_handler = logging.FileHandler(error_file, mode="a")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        root_logger.info(f"Logging to console and files: {log_file}, {error_file}")

    # Return the module logger (which inherits from root)
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Setup logging with both console and file handlers.

    Configures the root logger so that all modules (including trainers)
    automatically log to both console and file.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files. If None, only console logging is used.

    Returns:
        Configured logger instance
    """
    # Configure the root logger so all child loggers inherit the settings
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers on root logger
    root_logger.handlers = []

    # Console handler - line-buffered for live output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    # Force line buffering by not using buffering
    console_handler.flush = sys.stdout.flush
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_dir is provided
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "train.log"

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Also log to a separate error-only file
        error_file = log_path / "error.log"
        error_handler = logging.FileHandler(error_file, mode="a")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        root_logger.info(f"Logging to console and files: {log_file}, {error_file}")

    # Return the module logger (which inherits from root)
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def find_checkpoint_path(checkpoint_source: str) -> Optional[str]:
    """Find checkpoint file to resume from.

    Args:
        checkpoint_source: Path to checkpoint file or directory

    Returns:
        Path to checkpoint file, or None if not found
    """
    if not checkpoint_source:
        return None

    path = Path(checkpoint_source)

    # If it's a file, use it directly
    if path.is_file():
        return str(path)

    # If it's a directory, look for checkpoint files
    if path.is_dir():
        # Priority: best.ckpt > last.ckpt > any .ckpt file
        candidates = ["best.ckpt", "last.ckpt"]
        for candidate in candidates:
            ckpt_path = path / candidate
            if ckpt_path.exists():
                return str(ckpt_path)

        # Find any .ckpt file
        ckpt_files = list(path.glob("*.ckpt"))
        if ckpt_files:
            # Sort by modification time (most recent first)
            ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(ckpt_files[0])

    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_student_model(config: dict, device: str) -> FrozenQwenStudent:
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

    model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=dtype,
        bypass_mode=False,
    )

    # Enable gradient checkpointing if configured
    if train_loop_config.get("gradient_checkpointing", False):
        import logging

        logging.getLogger(__name__).info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    return model


def create_loss_function(config: dict) -> DistillationLoss:
    loss_fn = DistillationLoss.from_config(config)
    return loss_fn


def dict_to_namespace(d: dict, preserve_keys: list = None) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for attribute access.

    Args:
        d: Dictionary to convert
        preserve_keys: List of keys whose values should remain as dicts (not converted)
    """
    if preserve_keys is None:
        preserve_keys = ["mixture_components"]

    if isinstance(d, dict):
        result = {}
        for k, v in d.items():
            if k in preserve_keys:
                # Keep as-is (for mixture_components list of dicts)
                result[k] = v
            else:
                result[k] = dict_to_namespace(v, preserve_keys)
        return SimpleNamespace(**result)
    elif isinstance(d, list):
        return [dict_to_namespace(item, preserve_keys) for item in d]
    else:
        return d


def main():
    parser = argparse.ArgumentParser(
        description="Online-no-cache iterative midblock training with mixed corpus support"
    )
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
    parser.add_argument(
        "--init-from-checkpoint",
        type=str,
        default=None,
        help="Warm-start from checkpoint (loads model weights only, fresh optimizer/scheduler)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    config = load_config(args.config)

    # Get log directory from config
    log_dir = config.get("logging", {}).get("log_dir")
    logger = setup_logging(args.log_level, log_dir)

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Experiment: {config.get('experiment_name', 'unknown')}")

    seed = config.get("seed", 1337)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision"),
        trust_remote_code=True,
    )
    # Note: Qwen3.5 already has proper PAD token (<|endoftext|>) distinct from EOS (<|im_end|>)
    # Only set pad_token if model truly lacks one (not the case for Qwen)
    if tokenizer.pad_token is None:
        logger.warning(
            f"Tokenizer {model_config['name']} has no pad_token, setting to eos_token"
        )
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        f"Tokenizer: PAD={tokenizer.pad_token} (ID:{tokenizer.pad_token_id}), "
        f"EOS={tokenizer.eos_token} (ID:{tokenizer.eos_token_id})"
    )

    logger.info("Creating student model...")
    model = create_student_model(config, device)

    param_summary = model.get_parameter_summary()
    logger.info(f"Model parameters: {param_summary}")

    init_from_checkpoint = args.init_from_checkpoint
    if not init_from_checkpoint:
        init_from_checkpoint = config.get("train_loop", {}).get("init_from_checkpoint")

    if init_from_checkpoint:
        warmstart_path = find_checkpoint_path(init_from_checkpoint)
        if warmstart_path:
            logger.info("=" * 60)
            logger.info(f"WARM-STARTING FROM CHECKPOINT (fresh optimizer/scheduler)")
            logger.info(f"Loading model weights from: {warmstart_path}")
            logger.info("Optimizer, scheduler, and global_step will be FRESH")
            logger.info("=" * 60)
            checkpoint = torch.load(
                warmstart_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model weights loaded successfully for warm-start")
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            logger.info("GPU memory cache cleared after warm-start loading")
        else:
            logger.warning(
                f"Could not find checkpoint for warm-start: {init_from_checkpoint}"
            )

    logger.info("Creating loss function...")
    loss_fn = create_loss_function(config)
    loss_fn = loss_fn.to(device)

    logger.info("Creating dataloaders via dataset factory...")
    data_config = config.get("data", {})
    batch_size = data_config.get("batch_size", 4)

    # Convert config dict to namespace for attribute access (required by dataset factory)
    config_ns = dict_to_namespace(config)

    # Use dataset factory to support both tinystories and mixture loaders
    dataloaders = get_experiment_dataloaders(
        config=config_ns,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders.get("val", None)

    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    if val_dataloader is not None:
        logger.info(f"Val dataloader: {len(val_dataloader)} batches")

    logger.info("Creating Trainer...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Determine checkpoint to resume from
    # Priority: CLI arg > config value > None
    resume_source = args.resume_from_checkpoint
    if not resume_source:
        resume_source = config.get("train_loop", {}).get("resume_from_checkpoint")

    checkpoint_path = find_checkpoint_path(resume_source)

    if checkpoint_path:
        logger.info("=" * 60)
        logger.info(f"RESUMING TRAINING FROM CHECKPOINT")
        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(
            f"Previous step: {trainer.global_step}, epoch: {trainer.current_epoch}"
        )
        logger.info("=" * 60)
        trainer.load_checkpoint(checkpoint_path)
        logger.info(
            f"Resumed at step {trainer.global_step}, epoch {trainer.current_epoch + 1}"
        )
        # Clear cache and run garbage collection after loading checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc

        gc.collect()
        logger.info("GPU memory cache cleared after checkpoint loading")
    elif resume_source:
        logger.warning(f"Could not find checkpoint to resume from: {resume_source}")

    if args.fast_dev_run:
        logger.info("Running fast dev loop (1 train step, 1 val step)...")

        try:
            train_batch = next(iter(train_dataloader))
            train_metrics = trainer.train_step(train_batch)
            logger.info(f"Train step metrics: {train_metrics}")

            val_batch = next(iter(val_dataloader))
            val_metrics = trainer.val_step(val_batch)
            logger.info(f"Val step metrics: {val_metrics}")

            logger.info("Fast dev run complete!")
        except Exception as e:
            logger.error(f"Fast dev run failed: {e}")
            raise
        return

    if args.limit_train_batches or args.limit_val_batches:
        logger.info("Running limited batch training...")
        max_epochs = config.get("train_loop", {}).get("max_epochs", 1)

        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")

            for batch_idx, batch in enumerate(train_dataloader):
                if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                    break

                try:
                    metrics = trainer.train_step(batch)
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"  Batch {batch_idx}: loss={metrics['loss']:.4f}, "
                            f"velocity_loss={metrics.get('velocity_loss', 0.0):.4f}, "
                            f"kl_loss={metrics.get('kl_loss', 0.0):.4f}, "
                            f"ce_loss={metrics.get('ce_loss', 0.0):.4f}, "
                            f"grad_norm={metrics.get('grad_norm', 0.0):.4f}, "
                            f"T={metrics['T']}"
                        )
                except Exception as e:
                    logger.error(f"Training step failed at batch {batch_idx}: {e}")
                    raise

            val_metrics = trainer.validate(max_batches=args.limit_val_batches)
            logger.info(f"Validation: {val_metrics}")

        logger.info("Limited batch training complete!")
        return

    logger.info("Starting full training...")
    training_start_time = time.time()

    try:
        trainer.fit()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    training_duration = time.time() - training_start_time
    logger.info(f"Training complete! Duration: {training_duration:.1f}s")

    checkpoint_dir = Path(
        config.get("train_loop", {}).get("checkpoint_dir", "./checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(final_path)
    logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()

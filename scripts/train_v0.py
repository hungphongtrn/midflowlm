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
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.training.losses import DistillationLoss
from src.training.trainer import Trainer
from src.training.data import create_cache_dataloader, get_cache_info


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    Args:
        config: Configuration dictionary
        cache_dir: Directory containing cache files

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    data_config = config.get("data", {})
    seed = config.get("seed", 1337)

    # Determine batch size
    batch_size = data_config.get("batch_size", 8)
    num_workers = data_config.get("num_workers", 0)
    pin_memory = data_config.get("pin_memory", False)

    # Create train dataloader
    train_dataloader = create_cache_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        split="train",
    )

    # Create val dataloader
    val_dataloader = create_cache_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 1,  # Different seed for val
        num_workers=num_workers,
        pin_memory=pin_memory,
        split="val",
    )

    return train_dataloader, val_dataloader


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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

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

    # Get cache directory
    cache_config = config.get("teacher_cache", {})
    cache_dir = cache_config.get("cache_dir", "./cache")

    if not Path(cache_dir).exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        logger.error("Please run scripts/build_teacher_cache.py first")
        sys.exit(1)

    # Print cache info
    try:
        cache_info = get_cache_info(cache_dir)
        logger.info(f"Cache info: {cache_info}")
    except Exception as e:
        logger.warning(f"Could not load cache info: {e}")

    # Create model
    logger.info("Creating student model...")
    model = create_student_model(config, device)

    param_summary = model.get_parameter_summary()
    logger.info(f"Model parameters: {param_summary}")

    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = create_loss_function(config)
    loss_fn = loss_fn.to(device)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(config, cache_dir)
    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)

    # Fast dev run
    if args.fast_dev_run:
        logger.info("Running fast dev loop (1 train step, 1 val step)...")

        # One train step
        train_batch = next(iter(train_dataloader))
        train_metrics = trainer.train_step(train_batch)
        logger.info(f"Train step metrics: {train_metrics}")

        # One val step
        val_batch = next(iter(val_dataloader))
        val_metrics = trainer.val_step(val_batch)
        logger.info(f"Val step metrics: {val_metrics}")

        logger.info("Fast dev run complete!")
        return

    # Limited batch run
    if args.limit_train_batches or args.limit_val_batches:
        logger.info("Running limited batch training...")

        max_epochs = config.get("train_loop", {}).get("max_epochs", 1)

        # Compute baseline perplexity before training
        baseline_ppl = trainer.compute_baseline_perplexity()

        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")

            # Training with limited batches
            for batch_idx, batch in enumerate(train_dataloader):
                if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                    break

                metrics = trainer.train_step(batch)

                if batch_idx % 10 == 0:
                    logger.info(
                        f"  Batch {batch_idx}: loss={metrics['loss']:.4f}, T={metrics['T']}"
                    )

            # Validation with limited batches
            logger.info("Running validation...")
            val_metrics = trainer.validate(max_batches=args.limit_val_batches)
            logger.info(f"Validation metrics: {val_metrics}")

            # Compute perplexity after epoch
            epoch_ppl = trainer.compute_epoch_perplexity(epoch + 1)
            logger.info(f"=== PPL CHANGE: {epoch_ppl - baseline_ppl:+.2f} ===")

            # Save checkpoint
            checkpoint_dir = Path(
                config.get("train_loop", {}).get("checkpoint_dir", "./checkpoints")
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info("Limited batch training complete!")
        return

    # Full training
    logger.info("Starting full training...")
    trainer.fit()

    # Save final checkpoint
    checkpoint_dir = Path(
        config.get("train_loop", {}).get("checkpoint_dir", "./checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(final_path)
    logger.info(f"Saved final checkpoint to {final_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

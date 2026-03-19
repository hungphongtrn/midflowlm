#!/usr/bin/env python3
"""Build teacher cache with full trajectory targets.

This script generates offline teacher cache for TinyStories dataset using Qwen teacher model.
It caches h_start, full span trajectory states, h_target, and optional logits for each sample.

Usage:
    python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --limit 8
    python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --split train
    python scripts/build_teacher_cache.py --config configs/v0_onemotif.yaml --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.teacher_cache import (
    TeacherCacheWriter,
    generate_batch_cache,
    load_metadata,
)
from src.data.dataset_factory import get_experiment_dataloaders
from src.model.qwen_parity import QwenInspector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_namespace(config: dict) -> argparse.Namespace:
    """Convert config dict to Namespace for compatibility with existing code."""

    class ConfigNamespace:
        pass

    ns = ConfigNamespace()

    # Create nested namespaces
    for section_name, section_data in config.items():
        if isinstance(section_data, dict):
            section_ns = ConfigNamespace()
            for key, value in section_data.items():
                setattr(section_ns, key, value)
            setattr(ns, section_name, section_ns)
        else:
            setattr(ns, section_name, section_data)

    return ns


def build_cache_for_split(
    config_dict: dict,
    config: argparse.Namespace,
    split: str,
    cache_dir: str,
    store_logits: bool,
    model_name: str,
    model_revision: Optional[str],
    start_layer: int,
    end_layer: int,
    seq_len: int,
    overwrite: bool,
    device: str,
    batch_size: int,
    limit: Optional[int],
) -> int:
    """Build teacher cache for a specific split.

    Args:
        config_dict: Raw configuration dictionary
        config: Config namespace
        split: Dataset split to cache (train/val/test)
        cache_dir: Base cache directory
        store_logits: Whether to store logits
        model_name: Name of the teacher model
        model_revision: Model revision/commit hash
        start_layer: First layer of replacement span
        end_layer: Last layer of replacement span
        seq_len: Sequence length
        overwrite: Whether to overwrite existing cache
        device: Device for inference
        batch_size: Batch size for processing
        limit: Maximum number of samples to cache

    Returns:
        Number of samples cached
    """
    from src.data.teacher_cache import resolve_split_cache_dir

    # Resolve split-specific cache directory
    split_cache_dir = resolve_split_cache_dir(cache_dir, split)

    logger.info(f"Building teacher cache for {model_name}")
    logger.info(f"  Span: layers {start_layer}-{end_layer}")
    logger.info(f"  Cache dir: {split_cache_dir}")
    logger.info(f"  Store logits: {store_logits}")
    logger.info(f"  Split: {split}")
    if limit:
        logger.info(f"  Limit: {limit} samples")

    # Initialize cache writer for this split
    writer = TeacherCacheWriter(
        cache_dir=split_cache_dir,
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        store_logits=store_logits,
        model_revision=model_revision,
        overwrite=overwrite,
    )

    # Check if cache already exists
    try:
        existing_metadata = load_metadata(split_cache_dir)
        logger.info(
            f"Found existing cache with {existing_metadata.num_samples} samples"
        )
        if not overwrite:
            logger.info("Use --overwrite to rebuild cache")
            return existing_metadata.num_samples
    except FileNotFoundError:
        pass

    # Initialize teacher inspector
    logger.info("Loading teacher model...")
    inspector = QwenInspector(
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        dtype=torch.float32,
    )

    # Get dataloaders
    logger.info("Loading dataset...")
    dataloaders = get_experiment_dataloaders(
        config=config,
        batch_size=batch_size,
    )

    if split not in dataloaders:
        raise ValueError(
            f"Split '{split}' not found. Available: {list(dataloaders.keys())}"
        )

    dataloader = dataloaders[split]

    # Process samples
    num_samples = 0
    num_shards = limit if limit else len(dataloader.dataset)

    logger.info(f"Processing {num_shards} samples...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching samples")):
        # Check if we've reached the limit before processing
        if limit and num_samples >= limit:
            break

        # Calculate how many samples to process from this batch
        batch_size_actual = batch["input_ids"].shape[0]
        if limit:
            remaining = limit - num_samples
            if remaining <= 0:
                break
            if remaining < batch_size_actual:
                # Truncate batch to only process remaining samples
                batch = {
                    "input_ids": batch["input_ids"][:remaining],
                    "attention_mask": batch["attention_mask"][:remaining],
                }
                batch_size_actual = remaining

        # Generate cache for the entire batch at once (leverages GPU parallelism)
        try:
            cache_list = generate_batch_cache(
                batch=batch,
                inspector=inspector,
                device=device,
                store_logits=store_logits,
            )

            # Write each sample from the batch
            for cache_data in cache_list:
                writer.write_shard(
                    sample_data=cache_data,
                    shard_idx=num_samples,
                    num_shards=num_shards,
                )
                num_samples += 1

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            raise

    # Write metadata
    writer.write_metadata(num_samples=num_samples)

    logger.info(f"Cache build complete for {split}: {num_samples} samples cached")
    logger.info(f"Cache location: {split_cache_dir}")

    return num_samples


def build_cache_with_split(
    config_path: str,
    val_split_ratio: float = 0.05,
    limit: Optional[int] = None,
    batch_size: int = 1,
    overwrite: bool = False,
    device: Optional[str] = None,
) -> None:
    """Build teacher cache with automatic train/val split.

    Args:
        config_path: Path to configuration YAML file
        val_split_ratio: Ratio of samples to use for validation (default: 0.05)
        limit: Maximum number of samples to cache (None for all)
        batch_size: Batch size for processing
        overwrite: Whether to overwrite existing cache
        device: Device for model inference (auto-detected if None)
    """
    # Load configuration
    config_dict = load_config(config_path)
    config = create_namespace(config_dict)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Extract cache configuration
    cache_config = config_dict.get("teacher_cache", {})
    cache_dir = cache_config.get("cache_dir", "./cache/teacher_cache")
    store_logits = cache_config.get("store_logits", True)

    # Extract model configuration
    model_name = config_dict["model"]["name"]
    model_revision = config_dict["model"].get("revision")

    # Extract replacement span configuration
    start_layer = config_dict["replacement_model"]["start_layer"]
    end_layer = config_dict["replacement_model"]["end_layer"]
    seq_len = config_dict["data"]["seq_len"]

    logger.info(f"Building teacher cache with auto-split for {model_name}")
    logger.info(f"  Span: layers {start_layer}-{end_layer}")
    logger.info(f"  Base cache dir: {cache_dir}")
    logger.info(f"  Store logits: {store_logits}")
    logger.info(f"  Val split ratio: {val_split_ratio}")
    if limit:
        logger.info(f"  Limit: {limit} samples")

    # Initialize teacher inspector
    logger.info("Loading teacher model...")
    inspector = QwenInspector(
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        dtype=torch.float32,
    )

    # Get dataloaders
    logger.info("Loading dataset...")
    dataloaders = get_experiment_dataloaders(
        config=config,
        batch_size=batch_size,
    )

    # Use 'train' split for processing and then split internally
    if "train" not in dataloaders:
        raise ValueError(
            f"Split 'train' not found. Available: {list(dataloaders.keys())}"
        )

    dataloader = dataloaders["train"]

    # Calculate split sizes
    total_samples = limit if limit else len(dataloader.dataset)
    val_samples = int(total_samples * val_split_ratio)
    train_samples = total_samples - val_samples

    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"  Train: {train_samples}")
    logger.info(f"  Val: {val_samples}")

    # Resolve split-specific cache directories
    from src.data.teacher_cache import resolve_split_cache_dir

    train_cache_dir = resolve_split_cache_dir(cache_dir, "train")
    val_cache_dir = resolve_split_cache_dir(cache_dir, "val")

    # Initialize cache writers for each split
    train_writer = TeacherCacheWriter(
        cache_dir=train_cache_dir,
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        store_logits=store_logits,
        model_revision=model_revision,
        overwrite=overwrite,
    )

    val_writer = TeacherCacheWriter(
        cache_dir=val_cache_dir,
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        store_logits=store_logits,
        model_revision=model_revision,
        overwrite=overwrite,
    )

    # Process samples
    train_count = 0
    val_count = 0
    total_processed = 0

    logger.info(f"Processing {total_samples} samples...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching samples")):
        # Check if we've reached the limit before processing
        if limit and total_processed >= limit:
            break

        # Calculate how many samples to process from this batch
        batch_size_actual = batch["input_ids"].shape[0]
        if limit:
            remaining = limit - total_processed
            if remaining <= 0:
                break
            if remaining < batch_size_actual:
                # Truncate batch to only process remaining samples
                batch = {
                    "input_ids": batch["input_ids"][:remaining],
                    "attention_mask": batch["attention_mask"][:remaining],
                }
                batch_size_actual = remaining

        # Generate cache for the entire batch at once (leverages GPU parallelism)
        try:
            cache_list = generate_batch_cache(
                batch=batch,
                inspector=inspector,
                device=device,
                store_logits=store_logits,
            )

            # Write each sample from the batch to appropriate split
            for cache_data in cache_list:
                # First N samples go to train, rest go to val
                if train_count < train_samples:
                    train_writer.write_shard(
                        sample_data=cache_data,
                        shard_idx=train_count,
                        num_shards=train_samples,
                    )
                    train_count += 1
                else:
                    val_writer.write_shard(
                        sample_data=cache_data,
                        shard_idx=val_count,
                        num_shards=val_samples,
                    )
                    val_count += 1
                total_processed += 1

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            raise

    # Write metadata for each split
    train_writer.write_metadata(num_samples=train_count)
    val_writer.write_metadata(num_samples=val_count)

    logger.info(f"Cache build complete:")
    logger.info(f"  Train: {train_count} samples -> {train_cache_dir}")
    logger.info(f"  Val: {val_count} samples -> {val_cache_dir}")


def build_cache(
    config_path: str,
    split: str = "train",
    limit: Optional[int] = None,
    batch_size: int = 1,
    overwrite: bool = False,
    device: Optional[str] = None,
) -> None:
    """Build teacher cache for the specified configuration.

    Args:
        config_path: Path to configuration YAML file
        split: Dataset split to cache (train/val/test/all)
        limit: Maximum number of samples to cache (None for all)
        batch_size: Batch size for processing (currently only supports 1)
        overwrite: Whether to overwrite existing cache
        device: Device for model inference (auto-detected if None)
    """
    # Load configuration
    config_dict = load_config(config_path)
    config = create_namespace(config_dict)

    # Extract cache configuration
    cache_config = config_dict.get("teacher_cache", {})
    val_split_ratio = cache_config.get("val_split_ratio", 0.05)

    # Handle 'all' split option
    if split == "all":
        build_cache_with_split(
            config_path=config_path,
            val_split_ratio=val_split_ratio,
            limit=limit,
            batch_size=batch_size,
            overwrite=overwrite,
            device=device,
        )
        return

    # Extract cache configuration
    cache_dir = cache_config.get("cache_dir", "./cache/teacher_cache")
    store_logits = cache_config.get("store_logits", True)

    # Extract model configuration
    model_name = config_dict["model"]["name"]
    model_revision = config_dict["model"].get("revision")

    # Extract replacement span configuration
    start_layer = config_dict["replacement_model"]["start_layer"]
    end_layer = config_dict["replacement_model"]["end_layer"]
    seq_len = config_dict["data"]["seq_len"]

    # Build cache for the specified split
    build_cache_for_split(
        config_dict=config_dict,
        config=config,
        split=split,
        cache_dir=cache_dir,
        store_logits=store_logits,
        model_name=model_name,
        model_revision=model_revision,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        overwrite=overwrite,
        device=device if device else ("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
        limit=limit,
    )


def verify_cache(
    cache_dir: str, num_samples: int = 1, split: Optional[str] = None
) -> None:
    """Verify cache contents by loading and checking samples.

    Args:
        cache_dir: Directory containing cache files
        num_samples: Number of samples to verify
        split: Specific split to verify (train/val), or None to verify root cache dir
    """
    from src.data.teacher_cache import resolve_split_cache_dir

    # Resolve split-specific directory if split is specified
    if split:
        cache_dir = str(resolve_split_cache_dir(cache_dir, split))
        logger.info(f"Verifying {split} cache at {cache_dir}...")
    else:
        logger.info(f"Verifying cache at {cache_dir}...")

    # Load metadata
    metadata = load_metadata(cache_dir)
    logger.info(f"Cache metadata:")
    logger.info(f"  Model: {metadata.model_name}")
    logger.info(f"  Span: layers {metadata.start_layer}-{metadata.end_layer}")
    logger.info(f"  Total samples: {metadata.num_samples}")
    logger.info(f"  Store logits: {metadata.store_logits}")
    logger.info(f"  Target type: {getattr(metadata, 'target_type', 'unknown')}")
    logger.info(
        f"  Training rule: {getattr(metadata, 'training_state_rule', 'unknown')}"
    )

    # Load and verify samples
    from src.data.teacher_cache import load_shard

    num_to_check = min(num_samples, metadata.num_samples)
    for i in range(num_to_check):
        try:
            data = load_shard(
                cache_dir=cache_dir,
                shard_idx=i,
                num_shards=metadata.num_samples,
            )

            logger.info(f"Sample {i}:")
            logger.info(f"  input_ids shape: {data['input_ids'].shape}")
            logger.info(f"  h_start shape: {data['h_start'].shape}")
            logger.info(f"  velocity_target shape: {data['velocity_target'].shape}")
            logger.info(f"  h_target shape: {data['h_target'].shape}")
            if "teacher_logits" in data:
                logger.info(f"  teacher_logits shape: {data['teacher_logits'].shape}")

            # Verify velocity_target = h_target - h_start
            expected_velocity = data["h_target"] - data["h_start"]
            if torch.allclose(
                data["velocity_target"], expected_velocity, rtol=1e-5, atol=1e-6
            ):
                logger.info(
                    f"  ✓ velocity_target correctly computed: h_target - h_start"
                )
            else:
                logger.warning(f"  ✗ velocity_target mismatch with h_target - h_start!")

        except Exception as e:
            logger.error(f"Error verifying sample {i}: {e}")
            raise

    logger.info("Cache verification complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build teacher cache with full trajectory targets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v0_onemotif.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to cache (use 'all' to auto-split train/val)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to cache (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (cuda/cpu)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cache after building",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing cache, don't build",
    )

    args = parser.parse_args()

    if args.verify_only:
        # Load config to get cache_dir
        config = load_config(args.config)
        cache_dir = config["teacher_cache"]["cache_dir"]

        # If split is 'all', verify both train and val splits
        if args.split == "all":
            verify_cache(cache_dir, num_samples=args.limit or 3, split="train")
            verify_cache(cache_dir, num_samples=args.limit or 3, split="val")
        else:
            verify_cache(cache_dir, num_samples=args.limit or 3, split=args.split)
    else:
        build_cache(
            config_path=args.config,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            device=args.device,
        )

        if args.verify:
            config = load_config(args.config)
            cache_dir = config["teacher_cache"]["cache_dir"]

            # If split is 'all', verify both train and val splits
            if args.split == "all":
                verify_cache(cache_dir, num_samples=args.limit or 3, split="train")
                verify_cache(cache_dir, num_samples=args.limit or 3, split="val")
            else:
                verify_cache(cache_dir, num_samples=args.limit or 3, split=args.split)


if __name__ == "__main__":
    main()

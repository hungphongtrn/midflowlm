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
    generate_sample_cache,
    load_metadata,
    resolve_split_cache_dir,
)
from src.data.tinystories import get_tinystories_dataloaders
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
        split: Dataset split to cache (train/val/test)
        limit: Maximum number of samples to cache (None for all)
        batch_size: Batch size for processing (currently only supports 1)
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
    cache_root = cache_config.get("cache_dir", "./cache/teacher_cache")
    cache_dir = resolve_split_cache_dir(cache_root, split)
    store_logits = cache_config.get("store_logits", False)

    # Extract model configuration
    model_name = config_dict["model"]["name"]
    model_revision = config_dict["model"].get("revision")

    # Extract replacement span configuration
    start_layer = config_dict["replacement_model"]["start_layer"]
    end_layer = config_dict["replacement_model"]["end_layer"]
    seq_len = config_dict["data"]["seq_len"]

    logger.info(f"Building teacher cache for {model_name}")
    logger.info(f"  Span: layers {start_layer}-{end_layer}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Store logits: {store_logits}")
    logger.info(f"  Split: {split}")
    if limit:
        logger.info(f"  Limit: {limit} samples")

    # Initialize cache writer
    writer = TeacherCacheWriter(
        cache_dir=cache_dir,
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
        existing_metadata = load_metadata(cache_dir)
        logger.info(
            f"Found existing cache with {existing_metadata.num_samples} samples"
        )
        if not overwrite:
            logger.info("Use --overwrite to rebuild cache")
            return
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
    dataloaders = get_tinystories_dataloaders(
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
        if limit and batch_idx >= limit:
            break

        # Process each sample in batch (currently batch_size=1)
        for i in range(batch["input_ids"].shape[0]):
            sample = {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
            }

            # Generate cache for this sample
            try:
                cache_data = generate_sample_cache(
                    sample=sample,
                    inspector=inspector,
                    device=device,
                    store_logits=store_logits,
                )

                # Write shard
                writer.write_shard(
                    sample_data=cache_data,
                    shard_idx=num_samples,
                    num_shards=num_shards,
                )

                num_samples += 1

            except Exception as e:
                logger.error(f"Error processing sample {num_samples}: {e}")
                raise

    # Write metadata
    writer.write_metadata(num_samples=num_samples)

    logger.info(f"Cache build complete: {num_samples} samples cached")
    logger.info(f"Cache location: {cache_dir}")


def verify_cache(cache_dir: str, num_samples: int = 1) -> None:
    """Verify cache contents by loading and checking samples.

    Args:
        cache_dir: Directory containing cache files
        num_samples: Number of samples to verify
    """
    logger.info(f"Verifying cache at {cache_dir}...")

    # Load metadata
    metadata = load_metadata(cache_dir)
    logger.info(f"Cache metadata:")
    logger.info(f"  Model: {metadata.model_name}")
    logger.info(f"  Span: layers {metadata.start_layer}-{metadata.end_layer}")
    logger.info(f"  Total samples: {metadata.num_samples}")
    logger.info(f"  Store logits: {metadata.store_logits}")

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
            logger.info(
                f"  trajectory_targets count: {len(data['trajectory_targets'])}"
            )
            logger.info(f"  h_target shape: {data['h_target'].shape}")
            if "teacher_logits" in data:
                logger.info(f"  teacher_logits shape: {data['teacher_logits'].shape}")

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
        choices=["train", "val", "test"],
        help="Dataset split to cache",
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
        cache_root = config["teacher_cache"]["cache_dir"]
        cache_dir = resolve_split_cache_dir(cache_root, args.split)
        verify_cache(cache_dir, num_samples=args.limit or 3)
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
            cache_root = config["teacher_cache"]["cache_dir"]
            cache_dir = resolve_split_cache_dir(cache_root, args.split)
            verify_cache(cache_dir, num_samples=args.limit or 3)


if __name__ == "__main__":
    main()

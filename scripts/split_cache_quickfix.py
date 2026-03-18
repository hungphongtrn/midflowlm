#!/usr/bin/env python3
"""Quick fix script to split existing cache into train/val folders.

Val set = 5% (1000 samples), Train set = 95% (19000 samples)
"""

import json
import shutil
from pathlib import Path
import sys


def main():
    cache_dir = Path("./cache/tinystories_qwen_boundary_states")
    val_size = 1000
    train_size = 19000

    print("=== Splitting cache into train/val folders ===")
    print(f"Cache dir: {cache_dir}")
    print(f"Val size: {val_size} samples")
    print(f"Train size: {train_size} samples")

    # Create directories
    train_dir = cache_dir / "train"
    val_dir = cache_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Get all shard files sorted
    all_shards = sorted(cache_dir.glob("shard_*_of_*.safetensors"))
    total_shards = len(all_shards)

    print(f"\nFound {total_shards} total shards")

    if total_shards == 0:
        print("No shards to process!")
        sys.exit(1)

    # Split into val and train
    val_shards = all_shards[:val_size]
    train_shards = all_shards[val_size:]

    print(f"\n=== Moving {len(val_shards)} shards to val/ ===")
    for i, shard_path in enumerate(val_shards):
        new_name = f"shard_{i:04d}_of_{len(val_shards):04d}.safetensors"
        dest = val_dir / new_name
        shutil.move(str(shard_path), str(dest))
        if (i + 1) % 100 == 0:
            print(f"  Moved {i + 1}/{len(val_shards)} val shards...")

    print(f"\n=== Moving {len(train_shards)} shards to train/ ===")
    for i, shard_path in enumerate(train_shards):
        new_name = f"shard_{i:04d}_of_{len(train_shards):04d}.safetensors"
        dest = train_dir / new_name
        shutil.move(str(shard_path), str(dest))
        if (i + 1) % 1000 == 0:
            print(f"  Moved {i + 1}/{len(train_shards)} train shards...")

    # Create metadata for val
    print("\n=== Creating metadata.json for val ===")
    val_metadata = {
        "model_name": "Qwen/Qwen3.5-0.8B",
        "model_revision": None,
        "start_layer": 8,
        "end_layer": 11,
        "span_depth": 4,
        "seq_len": 128,
        "store_logits": False,
        "num_samples": len(val_shards),
    }
    with open(val_dir / "metadata.json", "w") as f:
        json.dump(val_metadata, f, indent=2)

    # Create metadata for train
    print("=== Creating metadata.json for train ===")
    train_metadata = {
        "model_name": "Qwen/Qwen3.5-0.8B",
        "model_revision": None,
        "start_layer": 8,
        "end_layer": 11,
        "span_depth": 4,
        "seq_len": 128,
        "store_logits": False,
        "num_samples": len(train_shards),
    }
    with open(train_dir / "metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    # Remove old metadata from root
    old_metadata = cache_dir / "metadata.json"
    if old_metadata.exists():
        old_metadata.unlink()
        print("Removed old metadata.json from root")

    # Verification
    print("\n=== Verifying split ===")
    val_count = len(list(val_dir.glob("*.safetensors")))
    train_count = len(list(train_dir.glob("*.safetensors")))
    print(f"Val shards: {val_count}")
    print(f"Train shards: {train_count}")
    print(f"Total: {val_count + train_count}")

    print("\n=== Done! Cache split complete ===")
    print(f"Train: {train_dir} ({train_count} samples)")
    print(f"Val: {val_dir} ({val_count} samples)")


if __name__ == "__main__":
    main()

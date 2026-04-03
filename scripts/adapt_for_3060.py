#!/usr/bin/env python3
"""
Generate 3060-compatible versions of v0.1 matrix configs.
Adjusts batch_size=1, grad_accum=16 for 12GB VRAM.

Usage:
    python3 scripts/adapt_for_3060.py
"""

import yaml
import os
from pathlib import Path

CONFIGS_DIR = Path("configs/v0_1_matrix")
OUTPUT_DIR = Path("configs/v0_1_matrix_3060")


def adapt_config(config_path: Path) -> dict:
    """Adapt a config for 3060 (12GB)."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get original name and modify
    orig_name = config["experiment_name"]
    config["experiment_name"] = orig_name + "_3060"

    # Adjust batch settings for 3060
    config["data"]["batch_size"] = 1  # Reduced from 2
    config["data"]["num_workers"] = 0  # Safer for local
    config["data"]["pin_memory"] = False  # Save memory

    # Increase grad accum to maintain effective batch
    config["train_loop"]["accumulate_grad_batches"] = 16  # Was 8

    # Update wandb tags
    if "wandb" in config:
        config["wandb"]["project"] = "midflowlm-v0-1-3060"
        if "3060" not in config["wandb"].get("tags", []):
            config["wandb"]["tags"].append("3060")

    # Update paths
    safe_name = config["experiment_name"].replace("-", "_")
    output_base = f"./outputs/{safe_name}"
    config["train_loop"]["checkpoint_dir"] = f"{output_base}/checkpoints"
    config["logging"]["log_dir"] = f"{output_base}/logs"
    if "tensorboard" in config:
        config["tensorboard"]["log_dir"] = f"{output_base}/tensorboard"
    if "teacher_cache" in config:
        config["teacher_cache"]["cache_dir"] = f"./cache/{safe_name}"

    return config


def main():
    print("Generating 3060-compatible configs...")
    print("=" * 50)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all configs
    config_files = list(CONFIGS_DIR.glob("*.yaml"))
    config_files = [f for f in config_files if f.name != "README.md"]

    for config_path in sorted(config_files):
        adapted = adapt_config(config_path)
        output_path = OUTPUT_DIR / config_path.name

        with open(output_path, "w") as f:
            # Header
            f.write(f"# {adapted['experiment_name']}\n")
            f.write(f"# 3060-adapted version of {config_path.name}\n")
            f.write(f"# Changes: batch_size=1, grad_accum=16, num_workers=0\n")
            f.write(f"#\n\n")
            yaml.dump(adapted, f, default_flow_style=False, sort_keys=False)

        print(f"  ✓ {output_path.name}")

    # Create README
    readme = OUTPUT_DIR / "README.md"
    with open(readme, "w") as f:
        f.write("# v0.1 Experiment Matrix - 3060 Adapted\n\n")
        f.write("These configs are adapted for RTX 3060 12GB.\n\n")
        f.write("Changes from base configs:\n")
        f.write("- batch_size: 2 → 1\n")
        f.write("- accumulate_grad_batches: 8 → 16\n")
        f.write("- num_workers: 2 → 0\n")
        f.write("- pin_memory: true → false\n")
        f.write("\nEffective batch size remains 16.\n\n")
        f.write("Usage:\n")
        f.write("```bash\n")
        f.write("# Single experiment\n")
        f.write(
            "python3 scripts/train.py --config configs/v0_1_matrix_3060/CONFIG.yaml\n"
        )
        f.write("\n")
        f.write("# Full matrix (sequential on 3060 - ~40-60 hours)\n")
        f.write("for cfg in configs/v0_1_matrix_3060/*.yaml; do\n")
        f.write('  python3 scripts/train.py --config "$cfg"\n')
        f.write("done\n")
        f.write("```\n")

    print(f"  ✓ {readme.name}")
    print("")
    print(f"Generated {len(config_files)} configs in {OUTPUT_DIR}/")
    print("")
    print("Note: Running full matrix on 3060 will take ~40-60 hours.")
    print("Consider using Vast for parallel execution on 3x GPUs.")


if __name__ == "__main__":
    main()

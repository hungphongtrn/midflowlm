#!/usr/bin/env python3
"""Adapt v0_1_matrix configs for RTX 3090 (24GB VRAM).

Changes:
- batch_size: 2 (from 3)
- num_workers: 2 (from 4)
- accumulate_grad_batches: 8 (from 5) to maintain effective batch size
- project name suffix: -3090
"""

import yaml
import os
from pathlib import Path

SOURCE_DIR = Path("configs/v0_1_matrix")
TARGET_DIR = Path("configs/v0_1_matrix_3090")


def adapt_config(source_path: Path, target_path: Path):
    with open(source_path) as f:
        config = yaml.safe_load(f)

    # Update experiment name
    orig_name = config["experiment_name"]
    config["experiment_name"] = orig_name + "_3090"

    # Update cache dir
    if "cache_dir" in config.get("teacher_cache", {}):
        config["teacher_cache"]["cache_dir"] = config["teacher_cache"][
            "cache_dir"
        ].replace(orig_name, config["experiment_name"])

    # Reduce batch size for 3090
    config["data"]["batch_size"] = 2
    config["data"]["num_workers"] = 2

    # Update grad accumulation to maintain effective batch size
    # Original: bs=3, accum=5 -> effective=15
    # New: bs=2, accum=8 -> effective=16 (close enough)
    config["train_loop"]["accumulate_grad_batches"] = 8

    # Update checkpoint and log dirs
    orig_ckpt = config["train_loop"]["checkpoint_dir"]
    config["train_loop"]["checkpoint_dir"] = orig_ckpt.replace(
        orig_name, config["experiment_name"]
    )

    orig_log = config["logging"]["log_dir"]
    config["logging"]["log_dir"] = orig_log.replace(
        orig_name, config["experiment_name"]
    )

    if config["logging"].get("tensorboard", {}).get("log_dir"):
        orig_tb = config["logging"]["tensorboard"]["log_dir"]
        config["logging"]["tensorboard"]["log_dir"] = orig_tb.replace(
            orig_name, config["experiment_name"]
        )

    # Update wandb project
    config["wandb"]["project"] = "midflowlm-v0-1-3090"

    # Add 3090 tag
    if "tags" in config["wandb"]:
        config["wandb"]["tags"].append("3090")

    # Add comment header
    lines = [
        f"# {config['experiment_name']}",
        f"# 3090-adapted version of {orig_name}.yaml",
        f"# Changes: batch_size=2, grad_accum=8, num_workers=2",
        f"#",
        "",
    ]

    with open(target_path, "w") as f:
        f.write("\n".join(lines))
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {target_path}")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for config_file in sorted(SOURCE_DIR.glob("*.yaml")):
        target_path = TARGET_DIR / config_file.name
        adapt_config(config_file, target_path)

    print(f"\nAll configs adapted for RTX 3090 in {TARGET_DIR}")
    print("Key changes: batch_size=2, num_workers=2, grad_accum=8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate all v0.1 experiment configs without full training.
Tests that each config can:
1. Load the model
2. Create the dataloaders
3. Run one forward/backward pass

Usage:
    python3 scripts/validate_configs.py
    python3 scripts/validate_configs.py --configs configs/v0_1_matrix_3060/*.yaml
"""

import argparse
import yaml
import sys
import traceback
from pathlib import Path


def validate_config(config_path: Path, limit_batches: int = 2) -> tuple[bool, str]:
    """
    Validate a single config by running a minimal training loop.
    Returns (success, error_message)
    """
    print(f"\n{'=' * 60}")
    print(f"Validating: {config_path.name}")
    print("=" * 60)

    try:
        # Import here to catch import errors per config
        import torch
        from scripts.train import main as train_main

        # Check GPU
        if not torch.cuda.is_available():
            return False, "No GPU available"

        # Parse args for the train script
        args = argparse.Namespace(
            config=str(config_path),
            device=None,
            fast_dev_run=False,
            limit_train_batches=limit_batches,
            limit_val_batches=1,
            resume_from_checkpoint=None,
            init_from_checkpoint=None,
            log_level="WARNING",
        )

        # Store original sys.argv and replace
        original_argv = sys.argv
        sys.argv = [
            "train.py",
            "--config",
            str(config_path),
            "--limit-train-batches",
            str(limit_batches),
            "--limit-val-batches",
            "1",
            "--log-level",
            "WARNING",
        ]

        try:
            # Run training with limited batches
            train_main()
            return True, ""
        except Exception as e:
            return False, str(e)
        finally:
            sys.argv = original_argv

    except Exception as e:
        return False, f"Setup error: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Validate v0.1 experiment configs")
    parser.add_argument(
        "--configs", nargs="+", help="Specific config files to validate"
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=2,
        help="Number of batches to run per config (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/v0_1_matrix_3060",
        help="Directory containing configs to validate",
    )
    args = parser.parse_args()

    # Get list of configs
    if args.configs:
        config_files = [Path(p) for p in args.configs]
    else:
        config_dir = Path(args.output_dir)
        config_files = list(config_dir.glob("*.yaml"))
        config_files = [f for f in config_files if f.name != "README.md"]

    if not config_files:
        print(f"No config files found in {args.output_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"Validating {len(config_files)} experiment configs")
    print(f"Running {args.limit_batches} batches per config")
    print("=" * 60)

    # Track results
    passed = []
    failed = []

    for config_file in sorted(config_files):
        success, error = validate_config(config_file, args.limit_batches)

        if success:
            passed.append(config_file.name)
            print(f"✓ {config_file.name}: PASSED")
        else:
            failed.append((config_file.name, error))
            print(f"✗ {config_file.name}: FAILED")
            print(f"  Error: {error[:200]}...")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total: {len(config_files)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed configs:")
        for name, error in failed:
            print(f"  - {name}")
        print("\nDetailed errors saved to: logs/validation_errors.log")

        # Save detailed errors
        Path("logs").mkdir(exist_ok=True)
        with open("logs/validation_errors.log", "w") as f:
            for name, error in failed:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"{name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(error)
                f.write("\n")

        sys.exit(1)
    else:
        print("\n✓ All configs validated successfully!")
        print("\nYour 3060 setup can run all experiments.")
        print("Note: Full training on 3060 will take ~40-60 hours total.")
        print("Consider using Vast with 3x3090 for parallel execution.")
        sys.exit(0)


if __name__ == "__main__":
    main()

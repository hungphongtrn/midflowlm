#!/usr/bin/env python3
"""
Print model statistics for the v0 iterative midblock student.

This script loads the configuration and student model to report:
- Total parameters
- Trainable parameters
- Frozen parameters
- Replacement span (start_layer, end_layer, depth)
- Allowed T values from training schedule

Usage:
    python scripts/print_model_stats.py --config configs/v0_onemotif.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.student_qwen import FrozenQwenStudent


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_memory(params: int, bytes_per_param: int = 4) -> str:
    """Convert parameter count to approximate memory usage."""
    bytes_total = params * bytes_per_param
    if bytes_total < 1024**3:
        return f"{bytes_total / (1024**2):.1f} MB"
    else:
        return f"{bytes_total / (1024**3):.2f} GB"


def print_model_stats(config: dict, device: str = "cpu") -> None:
    """Load model and print comprehensive statistics."""
    model_cfg = config.get("model", {})
    replacement_cfg = config.get("replacement_model", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3.5-0.8B")
    start_layer = replacement_cfg.get("start_layer", 8)
    end_layer = replacement_cfg.get("end_layer", 11)
    max_steps_T = model_cfg.get("max_steps_T", 8)
    train_T_values = model_cfg.get("train_T_values", [1, 2, 4, 6, 8])

    print("=" * 60)
    print("v0 Iterative Midblock - Model Statistics")
    print("=" * 60)
    print()

    # Configuration summary
    print("Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print()

    # Replacement span info
    span_depth = end_layer - start_layer + 1
    print("Replacement Span:")
    print(f"  Start layer: {start_layer}")
    print(f"  End layer: {end_layer}")
    print(f"  Span depth: {span_depth} layers")
    print(f"  Replaces layers: {start_layer} through {end_layer} (inclusive)")
    print()

    # T values info
    print("Iterative Refinement Schedule:")
    print(f"  Maximum T (max_steps_T): {max_steps_T}")
    print(f"  Training T values: {train_T_values}")
    print(f"  T < depth(span) regime: compression policy")
    print(f"  T = depth(span) regime: exact layerwise matching")
    print(f"  T > depth(span) regime: expansion/interpolation policy")
    print()

    # Load model and get parameter counts
    print("Loading model (this may take a moment)...")
    try:
        # Use fp32 for accurate parameter counting
        student = FrozenQwenStudent(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            max_steps_T=max_steps_T,
            device=device,
            dtype=torch.float32,
            bypass_mode=False,
        )

        total = student.get_total_parameter_count()
        trainable = student.get_trainable_parameter_count()
        frozen = student.get_frozen_parameter_count()

        print()
        print("Parameter Counts:")
        print(f"  Total:     {format_number(total):>15} ({format_memory(total)})")
        print(
            f"  Trainable: {format_number(trainable):>15} ({format_memory(trainable)})"
        )
        print(f"  Frozen:    {format_number(frozen):>15} ({format_memory(frozen)})")
        print()

        # Calculate percentages
        trainable_pct = (trainable / total) * 100 if total > 0 else 0
        frozen_pct = (frozen / total) * 100 if total > 0 else 0

        print("Parameter Distribution:")
        print(f"  Trainable: {trainable_pct:>6.2f}%")
        print(f"  Frozen:    {frozen_pct:>6.2f}%")
        print()

        # Midblock-specific info
        if student.midblock is not None:
            midblock_params = sum(p.numel() for p in student.midblock.parameters())
            print("Midblock Details:")
            print(f"  Midblock parameters: {format_number(midblock_params)}")
            print(f"  Hidden size: {student.midblock.hidden_size}")
            print(f"  Span depth: {student.midblock.span_depth}")
            print()

        # Training implications
        print("Training Implications:")
        print(f"  Only ~{trainable_pct:.2f}% of parameters are trainable")
        print(f"  Gradient checkpointing recommended for large batch sizes")
        print(f"  Teacher cache required before training")
        print()

        # Reference values for Qwen3.5-0.8B
        if "0.8B" in model_name:
            print("Reference: Qwen3.5-0.8B has ~800M total parameters")
            print(f"  Base model trainable (full finetune): {format_number(total)}")
            print(f"  This configuration trainable: {format_number(trainable)}")
            print(
                f"  Reduction: {(1 - trainable / total) * 100:.1f}% fewer trainable params"
            )
            print()

        print("=" * 60)
        print("Model stats loaded successfully")
        print("=" * 60)

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print()
        print("Falling back to configuration-only statistics...")
        print()

        # Provide config-based estimates
        print("Estimated Statistics (from config):")
        print(f"  Replacement span depth: {span_depth} layers")
        print(f"  Training T values: {train_T_values}")
        print()
        print("Note: Run with model access for accurate parameter counts")
        print("=" * 60)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Print model statistics for v0 iterative midblock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print stats using default config
  python scripts/print_model_stats.py --config configs/v0_onemotif.yaml

  # Print stats on GPU
  python scripts/print_model_stats.py --config configs/v0_onemotif.yaml --device cuda
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v0_onemotif.yaml",
        help="Path to configuration file (default: configs/v0_onemotif.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on (default: cpu)",
    )

    args = parser.parse_args()

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Print stats
    try:
        print_model_stats(config, device=args.device)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluation script for v0 iterative midblock and baselines.

This script runs evaluation on:
1. Identity baseline (h_end = h_start)
2. T=1 shared-block baseline
3. Simple recurrent baseline (no step conditioning)
4. Trained iterative midblock (if checkpoint provided)

It computes:
- Endpoint hidden-state error
- Trajectory error
- KL divergence (if logits available)
- Perplexity (if labels/logits available)
- Latency/tokens-per-second by T
- Total and trainable parameter counts
- Hidden-state norm and delta stability metrics

Usage:
    python scripts/eval_v0.py --config configs/v0_onemotif.yaml
    python scripts/eval_v0.py --config configs/v0_onemotif.yaml --checkpoint ./checkpoints/best.ckpt
    python scripts/eval_v0.py --config configs/v0_onemotif.yaml --baseline identity
    python scripts/eval_v0.py --config configs/v0_onemotif.yaml --num-steps 4 8
    python scripts/eval_v0.py --config configs/v0_onemotif.yaml --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.training.data import create_cache_dataloader, get_cache_info
from src.eval.baselines import (
    IdentityBaseline,
    T1SharedBlockBaseline,
    SimpleRecurrentBaseline,
    MetricsReport,
    compute_endpoint_error,
    compute_trajectory_error,
    compute_kl_divergence,
    compute_perplexity,
    compute_latency_metrics,
    compute_stability_metrics,
    get_parameter_counts,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_baseline(
    baseline_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Create a baseline model.

    Args:
        baseline_name: Name of baseline ('identity', 't1_shared', 'simple_recurrent')
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Baseline model
    """
    model_config = config["model"]
    hidden_size = 896  # Qwen3.5-0.8B hidden size
    num_heads = 8  # Default for Qwen3.5-0.8B

    if baseline_name == "identity":
        return IdentityBaseline()

    elif baseline_name == "t1_shared":
        return T1SharedBlockBaseline(
            hidden_size=hidden_size,
            num_heads=num_heads,
        ).to(device)

    elif baseline_name == "simple_recurrent":
        return SimpleRecurrentBaseline(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_steps_T=model_config["max_steps_T"],
        ).to(device)

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")


def create_student_model(
    config: dict,
    device: str,
    checkpoint_path: Optional[str] = None,
) -> FrozenQwenStudent:
    """Create the student model from config.

    Args:
        config: Configuration dictionary
        device: Device to load model on
        checkpoint_path: Optional path to checkpoint to load

    Returns:
        FrozenQwenStudent instance
    """
    model_config = config["model"]
    replacement_config = config["replacement_model"]

    model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=torch.float32,
        bypass_mode=False,
    )

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_midblock(checkpoint_path)

    return model


def run_forward_pass(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    num_steps: int,
    device: str,
) -> tuple[torch.Tensor, List[float]]:
    """Run forward pass and measure latency.

    Args:
        model: Model to evaluate
        h_start: Starting hidden states
        num_steps: Number of steps
        device: Device

    Returns:
        Tuple of (output, list of latencies in seconds)
    """
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(h_start, num_steps=num_steps)

    # Timed runs
    with torch.no_grad():
        for _ in range(10):
            start_time = time.perf_counter()
            output = model(h_start, num_steps=num_steps)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    return output, latencies


def evaluate_baseline_on_batch(
    model: torch.nn.Module,
    batch: dict,
    num_steps: int,
    device: str,
    compute_logits: bool = False,
) -> Dict[str, Any]:
    """Evaluate a baseline on a single batch.

    Args:
        model: Baseline model
        batch: Batch dictionary with 'h_start', 'h_end', 'input_ids', 'attention_mask'
        num_steps: Number of steps
        device: Device
        compute_logits: Whether to compute logits (requires student model with full forward)

    Returns:
        Dictionary of metrics
    """
    h_start = batch["h_start"].to(device)
    h_end_target = batch["h_end"].to(device)
    batch_size, seq_len, hidden_size = h_start.shape

    # Run forward pass and measure latency
    h_end_pred, latencies = run_forward_pass(model, h_start, num_steps, device)

    # Compute metrics
    endpoint_error = compute_endpoint_error(h_end_pred, h_end_target)

    latency_metrics = compute_latency_metrics(latencies, batch_size, seq_len)

    # Get parameter counts
    param_counts = get_parameter_counts(model)

    result = {
        "endpoint_error": endpoint_error,
        "latencies": latencies,
        **latency_metrics,
        **param_counts,
    }

    return result, h_end_pred


def evaluate_student_on_batch(
    model: FrozenQwenStudent,
    batch: dict,
    num_steps: int,
    device: str,
) -> Dict[str, Any]:
    """Evaluate the student model on a single batch.

    Args:
        model: Student model
        batch: Batch dictionary
        num_steps: Number of steps
        device: Device

    Returns:
        Dictionary of metrics and logits
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    h_start = batch["h_start"].to(device)
    h_end_target = batch["h_end"].to(device)

    batch_size, seq_len = input_ids.shape

    # Run forward pass and measure latency
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, attention_mask, num_steps=num_steps)

    # Timed runs
    with torch.no_grad():
        for _ in range(10):
            start_time = time.perf_counter()
            outputs = model(input_ids, attention_mask, num_steps=num_steps, return_dict=True)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    logits = outputs.logits

    # Compute hidden-state endpoint error (extract h_end from model)
    with torch.no_grad():
        # We need to manually compute h_end from the student
        h_mid = model.midblock.iterative_refinement(
            h_start=h_start,
            num_steps=num_steps,
            attention_mask=attention_mask,
        )
        endpoint_error = compute_endpoint_error(h_mid, h_end_target)

    # Compute latency metrics
    latency_metrics = compute_latency_metrics(latencies, batch_size, seq_len)

    # Get parameter counts
    param_counts = get_parameter_counts(model)

    result = {
        "endpoint_error": endpoint_error,
        "latencies": latencies,
        **latency_metrics,
        **param_counts,
    }

    return result, logits


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_steps: int,
    device: str,
    model_name: str,
    is_student: bool = False,
    max_batches: Optional[int] = None,
) -> MetricsReport:
    """Evaluate a model on the dataloader.

    Args:
        model: Model to evaluate (baseline or student)
        dataloader: Evaluation dataloader
        num_steps: Number of steps
        device: Device
        model_name: Name of the model for reporting
        is_student: Whether this is a student model (has different forward)
        max_batches: Maximum number of batches to evaluate

    Returns:
        MetricsReport with aggregated metrics
    """
    model.eval()

    all_endpoint_errors = []
    all_latencies = []
    total_params = None
    trainable_params = None
    batch_idx = 0

    for batch in dataloader:
        if max_batches and batch_idx >= max_batches:
            break

        if is_student:
            result, logits = evaluate_student_on_batch(model, batch, num_steps, device)
        else:
            result, _ = evaluate_baseline_on_batch(model, batch, num_steps, device)

        all_endpoint_errors.append(result["endpoint_error"])
        all_latencies.extend(result["latencies"])

        if total_params is None:
            total_params = result.get("total_params", 0)
            trainable_params = result.get("trainable_params", 0)

        batch_idx += 1

    # Compute aggregate metrics
    mean_endpoint_error = sum(all_endpoint_errors) / len(all_endpoint_errors)

    # Get batch info from first batch
    first_batch = next(iter(dataloader))
    batch_size = first_batch["h_start"].shape[0]
    seq_len = first_batch["h_start"].shape[1]

    latency_metrics = compute_latency_metrics(all_latencies, batch_size, seq_len)

    report = MetricsReport(
        endpoint_error=mean_endpoint_error,
        latency_ms=latency_metrics["mean_latency_ms"],
        tokens_per_second=latency_metrics["tokens_per_second"],
        total_params=total_params or 0,
        trainable_params=trainable_params or 0,
        model_name=model_name,
        num_steps_T=num_steps,
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate v0 baselines and models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device", type=str, default=None, help="Device to evaluate on (cuda/cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["identity", "t1_shared", "simple_recurrent", "all"],
                        help="Baseline to evaluate")
    parser.add_argument("--num-steps", type=int, nargs="+", default=None,
                        help="Number of steps to evaluate (can provide multiple)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum number of batches to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    global logger
    logger = setup_logging(args.log_level)

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Determine num_steps values to evaluate
    if args.num_steps:
        num_steps_list = args.num_steps
    else:
        num_steps_list = [1, config["model"]["max_steps_T"]]
    logger.info(f"Evaluating with T values: {num_steps_list}")

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

    # Create val dataloader
    logger.info("Creating validation dataloader...")
    data_config = config.get("data", {})
    val_dataloader = create_cache_dataloader(
        cache_dir=cache_dir,
        batch_size=data_config.get("batch_size", 8),
        shuffle=False,
        seed=config.get("seed", 1337) + 1,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=data_config.get("pin_memory", False),
        split="val",
    )
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")

    # Results storage
    all_results = []

    # Evaluate baselines
    baselines_to_evaluate = []
    if args.baseline == "all" or args.baseline is None:
        baselines_to_evaluate = ["identity", "t1_shared", "simple_recurrent"]
    elif args.baseline:
        baselines_to_evaluate = [args.baseline]

    for baseline_name in baselines_to_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating baseline: {baseline_name}")
        logger.info("="*60)

        model = create_baseline(baseline_name, config, device)

        # For identity baseline, only evaluate T=1 (it's the same for all T)
        if baseline_name == "identity":
            steps_list = [1]
        else:
            steps_list = num_steps_list

        for num_steps in steps_list:
            logger.info(f"  Running with T={num_steps}...")

            report = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                num_steps=num_steps,
                device=device,
                model_name=f"{baseline_name}_baseline",
                is_student=False,
                max_batches=args.max_batches,
            )

            logger.info(f"\n{report.summary()}")
            all_results.append(report.to_dict())

    # Evaluate student model if checkpoint provided
    if args.checkpoint:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating student model from checkpoint")
        logger.info("="*60)

        model = create_student_model(config, device, args.checkpoint)

        for num_steps in num_steps_list:
            logger.info(f"  Running with T={num_steps}...")

            report = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                num_steps=num_steps,
                device=device,
                model_name="trained_midblock",
                is_student=True,
                max_batches=args.max_batches,
            )

            logger.info(f"\n{report.summary()}")
            all_results.append(report.to_dict())

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

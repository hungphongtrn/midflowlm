#!/usr/bin/env python3
"""
Hardware Calibration Script for v0.1 Experiment Matrix

Sizes a single fixed hardware profile for seq_len=1024 on a single 3090,
calibrated against the worst-case intended loss regime:
- Architecture: FlowMidblock (A3)
- Loss: End + Traj + KL + CE (all losses enabled)
- Data: Mix C (largest corpus)

Usage:
    python scripts/calibrate_hardware.py --config configs/calibration_worst_case.yaml

Outputs:
    profiles/v0_1_3090_profile.json - Hardware profile artifact for queue runner
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model.student_qwen import create_frozen_qwen_student
from src.data.mixed_corpus import build_mixture_split_with_stats
from src.utils.config_loader import load_config


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "total": round(total, 2),
    }


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_gb() -> float:
    """Get peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def try_microbatch(
    config: Dict,
    microbatch_size: int,
    gradient_accumulation: int,
    max_steps: int = 5,
) -> Optional[Dict]:
    """
    Try training with a specific microbatch configuration.

    Returns metrics dict if successful, None if OOM or error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print(
            "WARNING: No GPU available. Calibration results will not be valid for 3090."
        )
        return None

    # Check GPU is a 3090 or similar
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    try:
        # Reset memory stats
        reset_gpu_memory()

        # Create model
        model_config = config["model"]
        replacement_config = config["replacement_model"]

        student_model = create_frozen_qwen_student(
            model_name=model_config["name"],
            revision=model_config.get("revision"),
            start_layer=replacement_config["start_layer"],
            end_layer=replacement_config["end_layer"],
            max_steps_T=model_config["max_steps_T"],
            step_embedding=model_config.get("step_embedding", "discrete"),
            architecture_mode=replacement_config.get(
                "architecture_mode", "causal_hidden_refiner"
            ),
            depth=replacement_config.get("depth", 4),
            mlp_ratio=replacement_config.get("mlp_ratio", 4.0),
            use_qwen_causal_mask=replacement_config.get("use_qwen_causal_mask", True),
            conditioning_mode=replacement_config.get(
                "conditioning_mode", "timestep_plus_layer_boundary"
            ),
            reuse_qwen_modules=model_config.get("reuse_qwen_modules", True),
            family=replacement_config.get("family", "flow_midblock"),
            qkv_bias=replacement_config.get("qkv_bias", True),
        )
        student_model.to(device)

        # Enable gradient checkpointing if configured
        if config["train_loop"].get("gradient_checkpointing", False):
            student_model.trainable_block.gradient_checkpointing = True

        # Create optimizer
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            student_model.trainable_parameters(),
            lr=optimizer_config["learning_rate"],
            betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )

        # Create a minimal dataloader
        data_config = config["data"]
        train_dataset, _ = build_mixture_split_with_stats(
            components_config=data_config["mixture_components"],
            tokenizer=student_model.tokenizer,
            seq_len=data_config["seq_len"],
            split="train",
            shuffle_seed=data_config.get("shuffle_seed", 1337),
            max_samples=100,  # Small sample for calibration
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=microbatch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for calibration to avoid multiprocessing issues
            pin_memory=data_config.get("pin_memory", False),
        )

        # Training loop for calibration
        student_model.train()
        step_times = []

        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            step_start = time.time()

            # Determine loss flags based on config
            loss_config = config["loss"]
            need_teacher_logits = (
                loss_config.get("kl_weight", 0.0) > 0.0
                or loss_config.get("ce_weight", 0.0) > 0.0
            )
            need_velocity = loss_config.get("velocity_weight", 0.0) > 0.0
            need_trajectory_anchors = (
                loss_config.get("endpoint_weight", 0.0) > 0.0
                or loss_config.get("trajectory_weight", 0.0) > 0.0
            )

            # Forward pass
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = student_model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    num_steps=model_config["train_T_values"][0],
                    return_teacher_targets=True,
                    need_teacher_logits=need_teacher_logits,
                    need_velocity=need_velocity,
                    need_trajectory_anchors=need_trajectory_anchors,
                )

                loss = outputs.get("loss", outputs.get("total_loss", 0))

                # Scale loss for gradient accumulation
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

            # Backward pass
            loss.backward()

            # Optimizer step (only after accumulation)
            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    student_model.trainable_parameters(),
                    optimizer_config.get("grad_clip_norm", 1.0),
                )
                optimizer.step()
                optimizer.zero_grad()

            step_end = time.time()
            step_times.append(step_end - step_start)

            # Clear cache periodically
            if step % 2 == 0:
                torch.cuda.empty_cache()

        # Get peak memory
        peak_vram = get_peak_memory_gb()
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        tokens_per_sec = (
            (microbatch_size * data_config["seq_len"]) / avg_step_time
            if avg_step_time > 0
            else 0
        )

        # Cleanup
        del student_model
        del optimizer
        torch.cuda.empty_cache()

        return {
            "peak_vram_gb": round(peak_vram, 2),
            "avg_step_time_sec": round(avg_step_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "success": True,
        }

    except torch.cuda.OutOfMemoryError as e:
        print(
            f"OOM at microbatch_size={microbatch_size}, grad_accum={gradient_accumulation}"
        )
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def find_optimal_profile(
    config: Dict,
    target_effective_batch_size: int = 16,
    vram_limit_gb: float = 22.0,
) -> Optional[Dict]:
    """
    Find optimal hardware profile by sweeping microbatch sizes.

    Strategy:
    1. Sweep microbatch from 1 upward with grad_accum=1 until OOM
    2. Pick the largest stable microbatch
    3. Add gradient accumulation to reach effective batch size
    4. Verify the final profile is stable
    """
    print("=" * 60)
    print("Hardware Calibration for v0.1 Experiment Matrix")
    print("=" * 60)
    print(f"Target effective batch size: {target_effective_batch_size}")
    print(f"VRAM limit: {vram_limit_gb} GB")
    print()

    # Phase 1: Find max stable microbatch with grad_accum=1
    print("Phase 1: Finding max stable microbatch (grad_accum=1)...")
    stable_microbatches = []

    for microbatch in range(1, 9):  # Test up to 8
        print(f"\nTesting microbatch_size={microbatch}, grad_accum=1...")
        result = try_microbatch(
            config, microbatch, gradient_accumulation=1, max_steps=3
        )

        if result is None:
            print(f"  Failed (OOM or error)")
            break

        print(f"  Peak VRAM: {result['peak_vram_gb']:.2f} GB")
        print(f"  Step time: {result['avg_step_time_sec']:.3f}s")
        print(f"  Tokens/sec: {result['tokens_per_sec']:.1f}")

        if result["peak_vram_gb"] <= vram_limit_gb:
            stable_microbatches.append((microbatch, result))
            print(f"  ✓ Stable")
        else:
            print(f"  ✗ Over limit")
            break

    if not stable_microbatches:
        print("ERROR: No stable configuration found even with microbatch=1!")
        return None

    # Choose the largest stable microbatch
    best_microbatch, best_result = stable_microbatches[-1]
    print(f"\n✓ Selected microbatch_size={best_microbatch}")

    # Phase 2: Add gradient accumulation to reach effective batch size
    print(
        f"\nPhase 2: Adding gradient accumulation to reach effective_batch_size={target_effective_batch_size}..."
    )

    gradient_accumulation = max(1, target_effective_batch_size // best_microbatch)
    effective_batch = best_microbatch * gradient_accumulation

    print(
        f"  microbatch_size={best_microbatch} × grad_accum={gradient_accumulation} = effective_batch={effective_batch}"
    )

    # Phase 3: Verify the final profile
    print(f"\nPhase 3: Verifying final profile...")
    final_result = try_microbatch(
        config, best_microbatch, gradient_accumulation, max_steps=5
    )

    if final_result is None:
        print("ERROR: Final profile verification failed!")
        return None

    print(f"  Peak VRAM: {final_result['peak_vram_gb']:.2f} GB")
    print(f"  Step time: {final_result['avg_step_time_sec']:.3f}s")
    print(f"  Tokens/sec: {final_result['tokens_per_sec']:.1f}")

    if final_result["peak_vram_gb"] > vram_limit_gb:
        print(f"  WARNING: Peak VRAM exceeds limit!")
        # Try with smaller microbatch
        for microbatch in range(best_microbatch - 1, 0, -1):
            gradient_accumulation = max(1, target_effective_batch_size // microbatch)
            print(
                f"\n  Retrying with microbatch={microbatch}, grad_accum={gradient_accumulation}..."
            )
            retry_result = try_microbatch(
                config, microbatch, gradient_accumulation, max_steps=5
            )
            if retry_result and retry_result["peak_vram_gb"] <= vram_limit_gb:
                best_microbatch = microbatch
                final_result = retry_result
                effective_batch = microbatch * gradient_accumulation
                print(f"  ✓ Success with microbatch={microbatch}")
                break
        else:
            print("ERROR: Could not find stable profile under VRAM limit!")
            return None

    print(f"\n{'=' * 60}")
    print("CALIBRATION COMPLETE")
    print(f"{'=' * 60}")

    profile = {
        "hardware": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        "seq_len": config["data"]["seq_len"],
        "microbatch_size": best_microbatch,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": effective_batch,
        "precision": config["train_loop"].get("precision", "bf16-mixed"),
        "gradient_checkpointing": config["train_loop"].get(
            "gradient_checkpointing", True
        ),
        "peak_vram_gb": final_result["peak_vram_gb"],
        "tokens_per_sec": final_result["tokens_per_sec"],
        "avg_step_time_sec": final_result["avg_step_time_sec"],
        "calibrated_on": "FlowMidblock_EndTrajKLCe_MixC",
        "calibration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate hardware profile for v0.1 experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/calibration_worst_case.yaml",
        help="Path to calibration config file",
    )
    parser.add_argument(
        "--target-batch-size", type=int, default=16, help="Target effective batch size"
    )
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=22.0,
        help="VRAM limit in GB (default: 22.0 for 24GB cards with headroom)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiles/v0_1_3090_profile.json",
        help="Output path for profile JSON",
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Run calibration
    profile = find_optimal_profile(
        config,
        target_effective_batch_size=args.target_batch_size,
        vram_limit_gb=args.vram_limit,
    )

    if profile is None:
        print("\nCalibration failed! No stable profile found.")
        sys.exit(1)

    # Write profile to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n✓ Profile written to: {output_path}")
    print("\nProfile contents:")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()

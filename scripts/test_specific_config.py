#!/usr/bin/env python3
"""
Test specific hardware configuration.
Usage: python3 scripts/test_specific_config.py --microbatch 3 --grad-accum 5
"""

import argparse
import json
import time
import os
import sys
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from transformers import AutoTokenizer

# Set environment variables for maximum CPU utilization
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["HF_DATASETS_PARALLELISM"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.data.mixed_corpus import build_mixture_split_with_stats, tokenize_function


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


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


def test_config(config_path, microbatch_size, gradient_accumulation, max_steps=5):
    """Test a specific microbatch + grad_accum configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    model_config = config["model"]
    replacement_config = config["replacement_model"]
    data_config = config["data"]

    print(f"Testing microbatch={microbatch_size}, grad_accum={gradient_accumulation}")
    print(f"Effective batch size: {microbatch_size * gradient_accumulation}")
    print()

    # Reset memory
    reset_gpu_memory()

    # Create model
    student_model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=str(device),
        family=replacement_config.get("family", "flow_midblock"),
    )
    student_model.to(device)

    if config["train_loop"].get("gradient_checkpointing", False):
        student_model.gradient_checkpointing_enable()

    # Create optimizer
    optimizer_config = config["optimizer"]
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
    )

    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_ns = SimpleNamespace(
        data=SimpleNamespace(
            mixture_components=data_config["mixture_components"],
            seq_len=data_config["seq_len"],
            shuffle_seed=data_config.get("shuffle_seed", 1337),
        )
    )

    train_dataset, _ = build_mixture_split_with_stats(
        config=config_ns,
        split="train",
        tokenizer=tokenizer,
        seq_len=data_config["seq_len"],
    )

    tokenize_fn = lambda examples: tokenize_function(
        examples, tokenizer=tokenizer, seq_len=data_config["seq_len"]
    )

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    cpu_count = multiprocessing.cpu_count()
    num_gpus = max(1, torch.cuda.device_count())
    optimal_workers = max(2, (cpu_count - 2) // num_gpus)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Training loop
    student_model.train()
    step_times = []

    try:
        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            step_start = time.time()

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=model_config["train_T_values"][0],
                    return_dict=True,
                )
                loss = outputs["logits"].sum()

                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

            loss.backward()

            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    optimizer_config.get("grad_clip_norm", 1.0),
                )
                optimizer.step()
                optimizer.zero_grad()

            step_end = time.time()
            step_times.append(step_end - step_start)

        peak_vram = get_peak_memory_gb()
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        tokens_per_sec = (
            (microbatch_size * data_config["seq_len"]) / avg_step_time
            if avg_step_time > 0
            else 0
        )

        print(f"✓ SUCCESS!")
        print(f"  Peak VRAM: {peak_vram:.2f} GB")
        print(f"  Step time: {avg_step_time:.3f}s")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")

        return {
            "success": True,
            "microbatch_size": microbatch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": microbatch_size * gradient_accumulation,
            "peak_vram_gb": round(peak_vram, 2),
            "avg_step_time_sec": round(avg_step_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM at step {step if 'step' in locals() else 'unknown'}")
        print(f"  Error: {e}")
        torch.cuda.empty_cache()
        return {"success": False, "error": "OOM"}
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        del student_model
        del optimizer
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Test specific hardware config")
    parser.add_argument("--config", default="configs/calibration_worst_case.yaml")
    parser.add_argument("--microbatch", type=int, default=3, help="Microbatch size")
    parser.add_argument(
        "--grad-accum", type=int, default=5, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Number of training steps"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("Testing Specific Configuration")
    print("=" * 60)
    print()

    result = test_config(args.config, args.microbatch, args.grad_accum, args.max_steps)

    if args.output and result["success"]:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

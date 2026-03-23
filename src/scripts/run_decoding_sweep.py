#!/usr/bin/env python3
"""Run decoding sweep comparing greedy vs sampling settings."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.text_checkpoint_sweep import load_texts, run_text_sweep


def main():
    checkpoint = (
        "outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt"
    )
    config = "configs/v0_onemotif.yaml"

    # Test configurations: (name, temperature, top_p)
    configs = [
        ("greedy", 0.0, 1.0),
        ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0),
        ("temp_0.7_top_p_0.9", 0.7, 0.9),
    ]

    texts = [
        "The robot looked at the night sky and wondered",
        "Once upon a time in a small village,",
        "Write a short story about a brave cat who",
    ]

    num_steps_list = [1, 4, 8]
    max_new_tokens = 48

    print("=" * 80)
    print("DECODING SWEEP: Testing repetition fix with sampling")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint}")
    print(f"num_steps: {num_steps_list}")
    print(f"max_new_tokens: {max_new_tokens}")
    print()

    results = {}

    for config_name, temp, top_p in configs:
        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config_name} (temp={temp}, top_p={top_p})")
        print(f"{'=' * 80}\n")

        payload = run_text_sweep(
            config_path=config,
            checkpoint_path=checkpoint,
            texts=texts,
            num_steps=num_steps_list,
            max_new_tokens=max_new_tokens,
            device="cuda",
            temperature=temp,
            top_p=top_p,
        )

        print(payload["table"])
        print()
        print(f"Repetition metrics for {config_name}:")
        for key, value in payload["repetition_metrics"].items():
            print(f"  {key}: {value:.3f}")

        results[config_name] = {
            "temperature": temp,
            "top_p": top_p,
            "metrics": payload["repetition_metrics"],
            "comparisons": payload["comparisons"],
        }

    # Save results
    output_path = Path("outputs/decoding_sweep_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}")

    # Summary
    print("\nSUMMARY - Repetition Metrics by Configuration:")
    print("-" * 80)
    print(f"{'Config':<20} {'2-gram':<10} {'3-gram':<10} {'4-gram':<10}")
    print("-" * 80)
    for config_name, data in results.items():
        m = data["metrics"]
        print(
            f"{config_name:<20} {m['mean_repeat_2gram_ratio']:<10.3f} {m['mean_repeat_3gram_ratio']:<10.3f} {m['mean_repeat_4gram_ratio']:<10.3f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal test to validate repetition fix."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.text_checkpoint_sweep import (
    greedy_generate,
    create_student,
    create_tokenizer,
    load_config,
)


def test_repetition_fix():
    """Test that temperature sampling breaks repetition loops."""
    config_path = "configs/v0_onemotif.yaml"
    checkpoint_path = (
        "outputs/11-12-19-03-2026-v0_qwen_iterative_midblock/checkpoints/final.ckpt"
    )

    config = load_config(config_path)
    model = create_student(config, device="cuda", bypass_mode=False)
    tokenizer = create_tokenizer(config["model"]["name"])

    # Load checkpoint
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompt = "The robot looked at the night sky and wondered"

    print("Testing repetition fix...")
    print(f"Prompt: {prompt}")
    print()

    # Test greedy (should show repetition)
    print("1. GREEDY (temp=0.0):")
    result_greedy = greedy_generate(
        model, tokenizer, prompt, num_steps=4, max_new_tokens=48, temperature=0.0
    )
    print(f"   Generated: {result_greedy.generated_text[:150]}...")

    # Test sampling (should break repetition)
    print("\n2. SAMPLING (temp=0.7):")
    result_sampling = greedy_generate(
        model, tokenizer, prompt, num_steps=4, max_new_tokens=48, temperature=0.7
    )
    print(f"   Generated: {result_sampling.generated_text[:150]}...")

    # Validation
    greedy_repeats = result_greedy.generated_text.count("wondered what it was doing")
    sampling_repeats = result_sampling.generated_text.count(
        "wondered what it was doing"
    )

    print(f"\nValidation:")
    print(f"  Greedy - phrase repeats: {greedy_repeats}")
    print(f"  Sampling - phrase repeats: {sampling_repeats}")

    if greedy_repeats >= 2 and sampling_repeats < greedy_repeats:
        print("\n✓ PASS: Temperature sampling reduces repetition")
        return True
    else:
        print("\n✗ FAIL: Repetition not sufficiently reduced")
        return False


if __name__ == "__main__":
    success = test_repetition_fix()
    sys.exit(0 if success else 1)

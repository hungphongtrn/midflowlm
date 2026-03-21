#!/usr/bin/env python3
"""Inference script to observe trained model behaviors on MMLU-Pro with more freedom.

Uses higher temperature and higher max tokens to observe model generation patterns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.mmlu_pro_behavior import (
    PromptBehavior,
    create_mmlu_pro_prompt,
    create_tokenizer,
    generate_behavior_completion,
    load_config,
    load_mmlu_pro_val,
)
from src.eval.text_checkpoint_sweep import (
    create_student,
    load_student_checkpoint,
    validate_num_steps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run free-form inference on MMLU-Pro prompts with trained models"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--solver-method", type=str, default="euler")
    parser.add_argument("--prompt-behavior", type=str, default="default", choices=["default", "stripped", "closed_think"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output
    if output_path is None:
        checkpoint_stem = Path(args.checkpoint).stem
        output_path = f"results/mmlu_pro_free_inference_{checkpoint_stem}.jsonl"

    config = load_config(args.config)
    max_steps_t = int(config["model"]["max_steps_T"])
    num_steps = args.num_steps or max_steps_t
    validate_num_steps([num_steps], max_steps_t)

    tokenizer = create_tokenizer(config["model"]["name"])
    questions = load_mmlu_pro_val(num_samples=args.num_samples, seed=args.seed)

    trained_model = create_student(config, device=device, bypass_mode=False)
    load_student_checkpoint(trained_model, args.checkpoint)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sample_index, question in enumerate(questions):
        prompt_text = create_mmlu_pro_prompt(
            question["question"],
            question["options"],
            tokenizer,
            prompt_behavior=args.prompt_behavior,
        )

        generation = generate_behavior_completion(
            model=trained_model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            num_steps=num_steps,
            max_new_tokens=args.max_new_tokens,
            is_student=True,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_on_eos=True,
            solver_method=args.solver_method,
        )

        record = {
            "sample_index": sample_index,
            "category": question.get("category", "unknown"),
            "question": question["question"],
            "options": question["options"],
            "correct_answer": question["correct_answer"],
            "num_steps": num_steps,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "solver_method": args.solver_method,
            "prompt_text": prompt_text,
            "generated_text": generation["generated_text"],
            "completion_length": len(generation["generated_token_ids"]),
            "stopped_on_eos": generation["stopped_on_eos"],
            "latency_ms": generation["latency_ms"],
        }

        results.append(record)
        output_file.open("a", encoding="utf-8").write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

        print(
            f"[{sample_index + 1}/{len(questions)}] Generated {record['completion_length']} tokens "
            f"({record['stopped_on_eos'] and 'stopped on EOS' or 'max tokens reached'}) | "
            f"Category: {record['category']}"
        )

    print(f"\nSaved {len(results)} records to {output_file}")


if __name__ == "__main__":
    main()

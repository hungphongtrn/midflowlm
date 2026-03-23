#!/usr/bin/env python3
"""Run a checkpoint on multiple texts across different iteration counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side original vs trained checkpoint outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_checkpoint_text_sweep.py \
      --checkpoint outputs/v0_qwen_iterative_midblock/checkpoints/final.ckpt \
      --text "Once upon a time" \
      --text "The moon was bright tonight" \
      --num-steps 1 4 8

  python scripts/run_checkpoint_text_sweep.py \
      --checkpoint outputs/v0_qwen_iterative_midblock/checkpoints/final.ckpt \
      --text-file prompts.txt \
      --num-steps 1 2 4 8 \
      --max-new-tokens 32 \
      --output outputs/text_sweep.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v0_onemotif.yaml",
        help="Path to the YAML config used to build the student model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trainer checkpoint, model state dict, or midblock state dict.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Prompt text to evaluate. Repeat this flag to add multiple prompts.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Optional file with one prompt per line.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Iteration counts to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of new tokens to greedily generate for each prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        choices=["cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--solver-method",
        type=str,
        default="euler",
        choices=["euler", "rk4", "dopri5", "adaptive_heun", "adaptive_lsoda"],
        help="ODE solver method for midblock integration.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, >0 = sampling). Higher values increase diversity but may reduce coherence.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (1.0 = disabled). Lower values restrict to high-probability tokens.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output file containing both raw results and a rendered table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.eval.text_checkpoint_sweep import load_texts, run_text_sweep

    texts = load_texts(cli_texts=args.text, text_file=args.text_file)

    payload = run_text_sweep(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        texts=texts,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_path=args.output,
        solver_method=args.solver_method,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Loaded checkpoint: {payload['checkpoint']['path']}")
    print(f"Checkpoint format: {payload['checkpoint']['checkpoint_format']}")
    print(f"Device: {payload['device']}")
    print(f"Solver method: {payload['solver_method']}")
    print(f"Temperature: {payload['temperature']}")
    print(f"Top-p: {payload['top_p']}")
    print(f"Prompts: {len(texts)}")
    print(f"Tested num_steps: {payload['num_steps']}")
    print(f"Configured max_steps_T: {payload['max_steps_T']}")

    if payload["warnings"]:
        print()
        print("Warnings:")
        for warning in payload["warnings"]:
            print(f"- {warning}")

    print()
    print(payload["table"])

    if args.output:
        print(f"Saved JSON report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

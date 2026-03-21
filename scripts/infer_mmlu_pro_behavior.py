#!/usr/bin/env python3
"""Inspect longer-form trained-model behavior on MMLU-Pro prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.mmlu_pro_behavior import (
    PromptBehavior,
    format_behavior_summary,
    run_mmlu_pro_behavior_observation,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavior-oriented MMLU-Pro inference with longer generation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, nargs="+", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--solver-method", type=str, default="euler")
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--no-stop-on-eos", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--prompt-behavior", type=str, default="default", choices=["default", "stripped", "closed_think"])
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output
    if output_path is None:
        checkpoint_stem = Path(args.checkpoint).stem
        output_path = f"results/mmlu_pro_behavior_{checkpoint_stem}.jsonl"

    payload = run_mmlu_pro_behavior_observation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        output_path=output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_teacher=not args.no_teacher,
        solver_method=args.solver_method,
        stop_on_eos=not args.no_stop_on_eos,
        prompt_behavior=args.prompt_behavior,
    )

    for warning in payload["warnings"]:
        logger.warning(warning)

    logger.info("Saved transcripts to %s", payload["output_path"])
    logger.info("\n%s", format_behavior_summary(payload["summaries"]))


if __name__ == "__main__":
    main()

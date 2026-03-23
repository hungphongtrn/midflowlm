#!/usr/bin/env python3
"""Downstream task evaluation on MMLU-Pro using chat templates.

This script evaluates models on MMLU-Pro multiple-choice questions using
Qwen chat templates for prompting. It computes accuracy metrics instead
of just perplexity/loss.

Usage:
    python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml
    python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --checkpoint ./checkpoints/best.ckpt
    python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --baseline identity
    python scripts/eval_mmlu_pro.py --config configs/v0_onemotif.yaml --num-samples 70 --num-steps 4 8
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent
from src.eval.baselines import (
    IdentityBaseline,
    T1SharedBlockBaseline,
    SimpleRecurrentBaseline,
)


@dataclass
class MMLUProResult:
    """Result for a single MMLU-Pro question."""

    question: str
    options: List[str]
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    num_steps: int
    model_name: str
    # Detailed input/output information
    prompt_text: str
    prompt_tokens: List[int]
    raw_output_token: int
    raw_output_text: str
    category: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "num_steps": self.num_steps,
            "model_name": self.model_name,
            "prompt_text": self.prompt_text,
            "prompt_tokens": self.prompt_tokens,
            "raw_output_token": self.raw_output_token,
            "raw_output_text": self.raw_output_text,
            "category": self.category,
        }


@dataclass
class MMLUProReport:
    """Aggregated results for MMLU-Pro evaluation."""

    accuracy: float
    num_correct: int
    num_total: int
    model_name: str
    num_steps: int
    avg_latency_ms: float
    detailed_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "model_name": self.model_name,
            "num_steps": self.num_steps,
            "avg_latency_ms": self.avg_latency_ms,
            "detailed_results": self.detailed_results,
        }

    def summary(self) -> str:
        return (
            f"Model: {self.model_name} (T={self.num_steps})\n"
            f"Accuracy: {self.accuracy:.2%} ({self.num_correct}/{self.num_total})\n"
            f"Avg Latency: {self.avg_latency_ms:.2f} ms"
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


def load_mmlu_pro_val(
    split: str = "validation", num_samples: int = 70, seed: int = 42
) -> List[Dict[str, Any]]:
    """Load MMLU-Pro validation dataset.

    Args:
        split: Dataset split to load
        num_samples: Number of samples to use
        seed: Random seed for sampling

    Returns:
        List of question dictionaries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading MMLU-Pro dataset (split={split}, n={num_samples})...")

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    # Sample random subset
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    questions = []
    for idx in indices:
        item = dataset[idx]
        # Parse options - they come as a list of strings like "A. option text"
        options_raw = item.get("options", [])
        options = []
        for opt in options_raw:
            # Extract option text after the letter prefix (e.g., "A. ")
            match = re.match(r"^([A-J])\.\s*(.+)$", opt.strip())
            if match:
                options.append(match.group(2))
            else:
                options.append(opt)

        questions.append(
            {
                "question": item["question"],
                "options": options,
                "correct_answer": item["answer"],  # This is the letter (A, B, C, etc.)
                "category": item.get("category", "unknown"),
            }
        )

    logger.info(f"Loaded {len(questions)} questions from MMLU-Pro")
    return questions


def create_mmlu_pro_prompt(
    question: str, options: List[str], tokenizer: AutoTokenizer
) -> str:
    """Create a chat-formatted prompt for MMLU-Pro question.

    Args:
        question: The question text
        options: List of option texts
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted prompt string
    """
    # Build options text
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join(
        [f"{letter}. {opt}" for letter, opt in zip(option_letters, options)]
    )

    # Create user message
    user_content = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, etc.).

Question: {question}

Options:
{options_text}

Answer:"""

    # Apply chat template
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers multiple choice questions. Respond with only the letter of the correct answer.",
        },
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def extract_answer(text: str, valid_options: List[str]) -> str:
    """Extract the answer letter from model output.

    Args:
        text: Model-generated text
        valid_options: List of valid option letters

    Returns:
        Extracted answer letter or "INVALID"
    """
    text = text.strip().upper()

    # Create a set for faster lookup
    valid_set = set(valid_options)

    # Try to find single letter at start
    if len(text) > 0 and text[0] in valid_set:
        return text[0]

    # Try to find pattern like "A." or "(A)" or "A)" at start
    match = re.match(r"^[\(\[]?([A-J])[\)\]\.]?\s*", text)
    if match and match.group(1) in valid_set:
        return match.group(1)

    # Try to find standalone option letters (not part of other words)
    # Look for patterns like " answer is X" or "X is correct"
    for opt in valid_options:
        # Pattern: word boundary + option + word boundary
        pattern = r"\b" + re.escape(opt) + r"\b"
        if re.search(pattern, text):
            return opt

    return "INVALID"


def evaluate_question(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: Dict[str, Any],
    num_steps: int,
    device: str,
    model_name: str,
    is_student: bool = True,
) -> Tuple[MMLUProResult, float]:
    """Evaluate a single MMLU-Pro question.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        question: Question dictionary
        num_steps: Number of steps for iterative models
        device: Device
        model_name: Name of the model
        is_student: Whether this is a student model

    Returns:
        Tuple of (MMLUProResult, latency_ms)
    """
    # Create prompt
    prompt = create_mmlu_pro_prompt(
        question["question"], question["options"], tokenizer
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate with timing
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        if is_student:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=num_steps,
            )
        else:
            logits = model(input_ids, num_steps=num_steps)

    # Get next token prediction
    next_token_logits = logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # Decode - both with and without special tokens
    raw_output_text = tokenizer.decode(next_token, skip_special_tokens=False)
    generated_text = tokenizer.decode(next_token, skip_special_tokens=True)

    # Extract answer
    valid_options = [chr(ord("A") + i) for i in range(len(question["options"]))]
    predicted = extract_answer(generated_text, valid_options)

    result = MMLUProResult(
        question=question["question"],
        options=question["options"],
        correct_answer=question["correct_answer"],
        predicted_answer=predicted,
        is_correct=(predicted == question["correct_answer"]),
        num_steps=num_steps,
        model_name=model_name,
        prompt_text=prompt,
        prompt_tokens=input_ids[0].tolist(),
        raw_output_token=int(next_token[0].item()),
        raw_output_text=raw_output_text,
        category=question.get("category", "unknown"),
    )

    return result, latency_ms


def evaluate_model_on_mmlu_pro(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    questions: List[Dict[str, Any]],
    num_steps: int,
    device: str,
    model_name: str,
    is_student: bool = True,
) -> MMLUProReport:
    """Evaluate a model on the full MMLU-Pro set.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        questions: List of question dictionaries
        num_steps: Number of steps
        device: Device
        model_name: Name of the model
        is_student: Whether this is a student model

    Returns:
        MMLUProReport with aggregated results
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluating {model_name} with T={num_steps} on {len(questions)} questions..."
    )

    results = []
    latencies = []

    for idx, question in enumerate(questions):
        result, latency = evaluate_question(
            model=model,
            tokenizer=tokenizer,
            question=question,
            num_steps=num_steps,
            device=device,
            model_name=model_name,
            is_student=is_student,
        )
        results.append(result)
        latencies.append(latency)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(questions)} questions")

    # Compute metrics
    num_correct = sum(1 for r in results if r.is_correct)
    num_total = len(results)
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Convert results to dicts for serialization
    detailed_results = [r.to_dict() for r in results]

    report = MMLUProReport(
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=num_total,
        model_name=model_name,
        num_steps=num_steps,
        avg_latency_ms=avg_latency,
        detailed_results=detailed_results,
    )

    return report


def create_baseline(
    baseline_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Create a baseline model.

    Args:
        baseline_name: Name of baseline
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Baseline model
    """
    model_config = config["model"]
    hidden_size = 896  # Qwen3.5-0.8B hidden size
    num_heads = 8

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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle trainer checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded trainer checkpoint (global_step={checkpoint.get('global_step', 'N/A')})"
            )
        else:
            # Try loading as midblock state dict
            model.load_midblock(checkpoint_path)

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU-Pro")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to evaluate on (cuda/cpu)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["identity", "t1_shared", "simple_recurrent", "all"],
        help="Baseline to evaluate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=None,
        help="Number of steps to evaluate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=70,
        help="Number of MMLU-Pro samples to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON"
    )
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

    # Load tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load MMLU-Pro questions
    questions = load_mmlu_pro_val(
        split="validation",
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Results storage
    all_results = []

    # Skip baselines - they don't support text generation in this context
    # Baselines are for hidden-state evaluation only (eval_v0.py)
    if args.baseline:
        logger.warning(
            "Baselines don't support text generation. Skipping baseline evaluation."
        )
        logger.warning("Use scripts/eval_v0.py for hidden-state baseline comparison.")

    # Evaluate student model if checkpoint provided
    if args.checkpoint:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating student model from checkpoint")
        logger.info("=" * 60)

        try:
            model = create_student_model(config, device, args.checkpoint)

            for num_steps in num_steps_list:
                logger.info(f"  Running with T={num_steps}...")

                report = evaluate_model_on_mmlu_pro(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    num_steps=num_steps,
                    device=device,
                    model_name="trained_midblock",
                    is_student=True,
                )

                logger.info(f"\n{report.summary()}")
                all_results.append(report.to_dict())
        except RuntimeError as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error(
                "The checkpoint may be incompatible with the current model architecture."
            )
            logger.error("Skipping student model evaluation.")

    # Also evaluate the teacher (original model) if no baseline specified
    if not args.baseline:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating teacher model (original Qwen)")
        logger.info("=" * 60)

        teacher_model = FrozenQwenStudent(
            model_name=config["model"]["name"],
            start_layer=config["replacement_model"]["start_layer"],
            end_layer=config["replacement_model"]["end_layer"],
            max_steps_T=config["model"]["max_steps_T"],
            device=device,
            dtype=torch.float32,
            bypass_mode=True,  # Use full model
        )

        report = evaluate_model_on_mmlu_pro(
            model=teacher_model,
            tokenizer=tokenizer,
            questions=questions,
            num_steps=1,  # Teacher doesn't use steps
            device=device,
            model_name="teacher_original",
            is_student=True,
        )

        logger.info(f"\n{report.summary()}")
        all_results.append(report.to_dict())

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": str(args.config),
            "num_samples": args.num_samples,
            "seed": args.seed,
            "device": device,
            "results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    logger.info("\nEvaluation complete!")

    # Print final summary table
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for result in all_results:
        logger.info(
            f"{result['model_name']:20s} T={result['num_steps']:2d} | "
            f"Accuracy: {result['accuracy']:.2%} | "
            f"Latency: {result['avg_latency_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()

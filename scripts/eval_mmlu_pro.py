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

================================================================================
EXPERIMENT RESULTS - MMLU-Pro Evaluation (72 samples, 256 max_new_tokens)
================================================================================

P1-A1: One-shot Projector (T=1 only)
  Checkpoint: ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt
  Config: configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml
  Results:
    - Base model (Qwen3.5-0.8B): 18/72 correct (25.0%)
    - P1-A1 (T=1): 4/72 correct (5.6%), 25/72 (34.7%) invalid outputs
  Issue: Could not complete generation with only 32 tokens; outputs fragments like "Based", "To", "The"

P1-A2: Shared Recurrent Residual (T ∈ [1, 2, 4, 8])
  Checkpoint: ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt
  Config: configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml
  W&B Run: ze54okvs (stilted-paper-3)
  Training: Multi-step with T ∈ [2,4,6,8], Endpoint + KL loss, Mix B
  Eval T values: [1, 2, 4, 8]
  
  Evaluation Commands:
    T=1:  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt --num-steps 1 --num-samples 72 --max-new-tokens 256
    T=2:  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt --num-steps 2 --num-samples 72 --max-new-tokens 256
    T=4:  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt --num-steps 4 --num-samples 72 --max-new-tokens 256
    T=8:  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt --num-steps 8 --num-samples 72 --max-new-tokens 256
  
  Expected: Performance should improve with more steps (T=4,8 > T=1,2)
  Baseline: Teacher model (full Qwen3.5-0.8B) = 25.0% accuracy

================================================================================
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
    """Extract the answer letter from model output using robust regex patterns.

    Handles various formats including markdown bold, parenthetical prefixes, and colons.

    Args:
        text: Model-generated text
        valid_options: List of valid option letters

    Returns:
        Extracted answer letter or "INVALID"
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.strip().lower()

    # Create a set for faster lookup (lowercase for comparison)
    valid_set = set(opt.lower() for opt in valid_options)

    # Pattern 1: Markdown bold answer: **C** or **Answer: C** or **answer is C**
    # Examples: "(A)answer is **C**" -> extract C, "**Answer: I**" -> extract I
    match = re.search(r"\*\*\s*(?:answer\s*[:is]+\s*)?\(?([a-j])\)?\s*\*\*", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer

    # Pattern 2: "answer is X" or "the answer is X" (case insensitive)
    # Matches: "answer is a", "the answer is b", "answer is (c)"
    match = re.search(r"(?:the\s+)?answer\s+is\s+\(?([a-j])\)?", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer

    # Pattern 3: "answer: X" or "Answer: X" (case insensitive)
    # Matches: "answer: a", "Answer: B", "answer: (c)"
    match = re.search(r"answer:\s*\(?([a-j])\)?", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer

    # Pattern 4: Letter in markdown bold anywhere: **A**, **B**, **(C)**
    match = re.search(r"\*\*\s*\(?([a-j])\)?\s*\*\*", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer

    # Fallback 1: Try to find single letter at start (original behavior, case insensitive)
    if len(text_lower) > 0 and text_lower[0] in valid_set:
        return text_lower[0].upper()

    # Fallback 2: Try to find pattern like "A." or "(A)" or "A)" at start
    match = re.match(r"^[\(\[]?([a-j])[\)\]\.]?\s*", text_lower)
    if match and match.group(1) in valid_set:
        return match.group(1).upper()

    # Fallback 3: Try to find standalone option letters anywhere in text
    for opt in valid_options:
        opt_lower = opt.lower()
        # Pattern: word boundary + option + word boundary
        pattern = r"\b" + re.escape(opt_lower) + r"\b"
        if re.search(pattern, text_lower):
            return opt

    return "INVALID"


def generate_autoregressive(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_steps: int,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    eos_token_id: Optional[int] = None,
    is_student: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Generate tokens autoregressively using the model.

    Args:
        model: Model to generate with
        input_ids: Initial input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        num_steps: Number of steps for iterative models
        tokenizer: Tokenizer for detecting EOS
        max_new_tokens: Maximum number of new tokens to generate
        eos_token_id: Token ID that signals end of generation
        is_student: Whether this is a student model

    Returns:
        Tuple of (generated_token_ids, latency_ms)
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated_tokens = []
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            # Forward pass
            if is_student:
                logits = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    num_steps=num_steps,
                )
            else:
                logits = model(current_input_ids, num_steps=num_steps)

            # Get next token prediction (greedy)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())

            # Check for EOS
            if next_token.item() == eos_token_id:
                break

            # Append to input for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            # Extend attention mask
            new_mask = torch.ones(
                (current_attention_mask.size(0), 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)

            # Truncate if getting too long (to avoid OOM)
            if current_input_ids.size(1) > 2048:
                # Keep only the last 1024 tokens
                current_input_ids = current_input_ids[:, -1024:]
                current_attention_mask = current_attention_mask[:, -1024:]

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return torch.tensor([generated_tokens], device=input_ids.device), latency_ms


def evaluate_question(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: Dict[str, Any],
    num_steps: int,
    device: str,
    model_name: str,
    is_student: bool = True,
    max_new_tokens: int = 256,
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
        max_new_tokens: Maximum tokens to generate (default 256)

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

    # Generate autoregressively with timing
    generated_ids, latency_ms = generate_autoregressive(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_steps=num_steps,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        is_student=is_student,
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    raw_output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

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
        raw_output_token=int(generated_ids[0][0].item()) if len(generated_ids[0]) > 0 else -1,
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
    max_new_tokens: int = 256,
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
        max_new_tokens: Maximum tokens to generate per question

    Returns:
        MMLUProReport with aggregated results
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluating {model_name} with T={num_steps} on {len(questions)} questions "
        f"(max_new_tokens={max_new_tokens})..."
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
            max_new_tokens=max_new_tokens,
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

    # Log sample outputs for debugging (first 3 incorrect and first 3 correct)
    incorrect_samples = [r for r in results if not r.is_correct][:3]
    correct_samples = [r for r in results if r.is_correct][:3]

    if incorrect_samples:
        logger.info(f"\n  Sample INCORRECT predictions:")
        for r in incorrect_samples:
            logger.info(f"    Q: {r.question[:60]}...")
            logger.info(f"    Expected: {r.correct_answer}, Predicted: {r.predicted_answer}")
            logger.info(f"    Raw output: {r.raw_output_text[:100]}...")

    if correct_samples:
        logger.info(f"\n  Sample CORRECT predictions:")
        for r in correct_samples:
            logger.info(f"    Q: {r.question[:60]}...")
            logger.info(f"    Answer: {r.correct_answer}")
            logger.info(f"    Raw output: {r.raw_output_text[:100]}...")

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
    parser = argparse.ArgumentParser(
        description="Evaluate models on MMLU-Pro",
        epilog="""
Examples:
  # P1-A1: One-shot projector (T=1 only)
  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt --num-steps 1

  # P1-A2: Shared recurrent residual - evaluate all T values
  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt --num-steps 1 2 4 8 --num-samples 72 --max-new-tokens 256

  # Teacher baseline
  python scripts/eval_mmlu_pro.py --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml --num-samples 72
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per question (default: 256)",
    )

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
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Evaluating T values: {num_steps_list}")
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
                    max_new_tokens=args.max_new_tokens,
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
            max_new_tokens=args.max_new_tokens,
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
    logger.info("FINAL SUMMARY - MMLU-Pro Results")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'T':>3} | {'Accuracy':>10} | {'Correct':>8} | {'Latency':>10}")
    logger.info("-" * 60)
    for result in all_results:
        logger.info(
            f"{result['model_name']:<20} {result['num_steps']:>3} | "
            f"{result['accuracy']:>9.2%} | "
            f"{result['num_correct']:>3}/{result['num_total']:<4} | "
            f"{result['avg_latency_ms']:>9.2f}ms"
        )
    logger.info("=" * 60)

    # T-sweep comparison if multiple T values were evaluated
    student_results = [r for r in all_results if r['model_name'] == 'trained_midblock']
    if len(student_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("T-SWEEP COMPARISON (Student Model)")
        logger.info("=" * 60)
        logger.info(f"{'T':>3} | {'Accuracy':>10} | {'Correct':>8} | {'vs T=1':>10}")
        logger.info("-" * 40)
        baseline_acc = None
        for result in sorted(student_results, key=lambda x: x['num_steps']):
            acc = result['accuracy']
            if baseline_acc is None:
                baseline_acc = acc
                improvement = "baseline"
            else:
                improvement = f"+{(acc - baseline_acc):.2%}"
            logger.info(
                f"{result['num_steps']:>3} | "
                f"{acc:>9.2%} | "
                f"{result['num_correct']:>3}/{result['num_total']:<4} | "
                f"{improvement:>10}"
            )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

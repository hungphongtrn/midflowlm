#!/usr/bin/env python3
"""Side-by-side MMLU-Pro evaluation comparing base model vs trained models.

Creates a CSV with:
- input: The question text
- base_model_full_message: Chat template + special tokens from base Qwen model
- p1_a1_full_message: Chat template + special tokens from P1-A1 trained model
- (extensible to add more experiments)

Usage:
    # Standard run (requires ~16GB VRAM)
    python scripts/eval_mmlu_pro_side_by_side.py \\
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml \\
        --p1-a1-checkpoint ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt \\
        --num-samples 72 \\
        --output results/mmlu_pro_side_by_side.csv
    
    # Low memory mode (for 13GB VRAM while training is running)
    python scripts/eval_mmlu_pro_side_by_side.py \\
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml \\
        --p1-a1-checkpoint ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt \\
        --num-samples 72 \\
        --low-memory \\
        --output results/mmlu_pro_side_by_side.csv
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent


# MMLU-Pro test set has exactly 72 samples per category, but we'll sample randomly
DEFAULT_NUM_SAMPLES = 72


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


def load_mmlu_pro_test(
    split: str = "test", num_samples: int = DEFAULT_NUM_SAMPLES, seed: int = 42
) -> List[Dict[str, Any]]:
    """Load MMLU-Pro test dataset.
    
    Args:
        split: Dataset split to load (test has 72 samples per category)
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
    available = len(dataset)
    if num_samples > available:
        logger.warning(f"Requested {num_samples} samples but only {available} available")
        num_samples = available
    indices = random.sample(range(available), num_samples)
    
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
        
        questions.append({
            "question_id": idx,
            "question": item["question"],
            "options": options,
            "correct_answer": item["answer"],  # This is the letter (A, B, C, etc.)
            "category": item.get("category", "unknown"),
        })
    
    logger.info(f"Loaded {len(questions)} questions from MMLU-Pro {split} split")
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
        Formatted prompt string (with special tokens)
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
    
    Args:
        text: Model-generated text
        valid_options: List of valid option letters
    
    Returns:
        Extracted answer letter or "INVALID"
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.strip().lower()
    valid_set = set(opt.lower() for opt in valid_options)
    
    # Pattern 1: "answer is X" or "the answer is X" (case insensitive)
    # Matches: "answer is a", "the answer is b", "answer is (c)"
    match = re.search(r"(?:the\s+)?answer\s+is\s+\(?([a-j])\)?", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer
    
    # Pattern 2: "answer: X" or "Answer: X" (case insensitive)
    # Matches: "answer: a", "Answer: B", "answer: (c)"
    match = re.search(r"answer:\s*\(?([a-j])\)?", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer
    
    # Fallback: Try to find single letter at start (case insensitive)
    if len(text_lower) > 0 and text_lower[0] in valid_set:
        return text_lower[0].upper()
    
    # Fallback: Try to find pattern like "A." or "(A)" or "A)" at start
    match = re.match(r"^[\(\[]?([a-j])[\)\]\.]?\s*", text_lower)
    if match and match.group(1) in valid_set:
        return match.group(1).upper()
    
    # Fallback: Try to find standalone option letters anywhere
    for opt in valid_options:
        opt_lower = opt.lower()
        pattern = r"\b" + re.escape(opt_lower) + r"\b"
        if re.search(pattern, text_lower):
            return opt
    
    return "INVALID"


def generate_with_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    num_steps: int = 1,
    is_student: bool = True,
    max_new_tokens: int = 256,
) -> Tuple[str, str, int]:
    """Generate response from model autoregressively and return both raw and decoded text.
    
    NOTE: Processes single sample (batch size 1) to minimize VRAM usage.
    This is intentional for running evaluation alongside training.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Prompt text (with chat template)
        device: Device
        num_steps: Number of steps for student model
        is_student: Whether this is a student model
        max_new_tokens: Maximum number of new tokens to generate (default 256)
    
    Returns:
        Tuple of (decoded_text_with_special_tokens, decoded_text_clean, first_token_id)
    """
    # Tokenize (batch size 1)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    eos_token_id = tokenizer.eos_token_id
    generated_tokens = []
    first_token_id = -1
    
    model.eval()
    with torch.no_grad():
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        for i in range(max_new_tokens):
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
            token_id = int(next_token[0].item())
            
            # Track first token ID
            if i == 0:
                first_token_id = token_id
            
            generated_tokens.append(token_id)
            
            # Check for EOS
            if token_id == eos_token_id:
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
                current_input_ids = current_input_ids[:, -1024:]
                current_attention_mask = current_attention_mask[:, -1024:]
    
    # Decode all generated tokens
    generated_tensor = torch.tensor([generated_tokens], device=device)
    raw_output_with_special = tokenizer.decode(generated_tensor[0], skip_special_tokens=False)
    clean_output = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
    
    return raw_output_with_special, clean_output, first_token_id


def create_base_model(config: dict, device: str) -> FrozenQwenStudent:
    """Create the base (teacher) model - full Qwen without replacement.
    
    Args:
        config: Configuration dictionary
        device: Device
    
    Returns:
        FrozenQwenStudent in bypass mode (full teacher)
    """
    model_config = config["model"]
    replacement_config = config["replacement_model"]
    
    model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=torch.bfloat16,  # Use bf16 for consistency with training
        bypass_mode=True,  # Use full model (no replacement)
    )
    return model


def create_p1_a1_model(config: dict, device: str, checkpoint_path: str) -> FrozenQwenStudent:
    """Create the P1-A1 trained model.
    
    Args:
        config: Configuration dictionary
        device: Device
        checkpoint_path: Path to checkpoint
    
    Returns:
        FrozenQwenStudent with loaded weights
    """
    model_config = config["model"]
    replacement_config = config["replacement_model"]
    
    model = FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=torch.bfloat16,
        bypass_mode=False,  # Use replacement
    )
    
    # Load checkpoint
    logger.info(f"Loading P1-A1 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle trainer checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint (global_step={checkpoint.get('global_step', 'N/A')})")
    else:
        # Try loading as midblock state dict directly
        model.load_midblock(checkpoint_path)
    
    return model


def run_side_by_side_evaluation(
    base_model: torch.nn.Module,
    p1_a1_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    questions: List[Dict[str, Any]],
    device: str,
    low_memory: bool = False,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Run side-by-side evaluation on all questions.
    
    Processes one question at a time (batch size 1) to minimize VRAM usage.
    This allows the evaluation to run alongside training (13GB used) without OOM.
    
    Args:
        base_model: Base Qwen model
        p1_a1_model: P1-A1 trained model
        tokenizer: Tokenizer
        questions: List of questions
        device: Device
        low_memory: If True, clear CUDA cache aggressively after each question
        max_new_tokens: Maximum tokens to generate per question
    
    Returns:
        List of result dictionaries for CSV
    """
    results = []
    
    for idx, question in enumerate(questions):
        # Create prompt with chat template
        prompt = create_mmlu_pro_prompt(
            question["question"], question["options"], tokenizer
        )
        
        logger.info(f"Processing question {idx + 1}/{len(questions)} (ID: {question['question_id']})")
        
        # Generate with base model (batch size 1 to save memory)
        base_raw, base_clean, base_token_id = generate_with_model(
            base_model, tokenizer, prompt, device, num_steps=1, is_student=True,
            max_new_tokens=max_new_tokens
        )
        
        # Clear cache if using CUDA to free memory for next model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Generate with P1-A1 model (batch size 1 to save memory)
        p1_a1_raw, p1_a1_clean, p1_a1_token_id = generate_with_model(
            p1_a1_model, tokenizer, prompt, device, num_steps=1, is_student=True,
            max_new_tokens=max_new_tokens
        )
        
        # Extract answers
        valid_options = [chr(ord("A") + i) for i in range(len(question["options"]))]
        base_answer = extract_answer(base_clean, valid_options)
        p1_a1_answer = extract_answer(p1_a1_clean, valid_options)
        
        # Build full message with chat template + model response
        base_full_message = prompt + base_raw
        p1_a1_full_message = prompt + p1_a1_raw
        
        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "input": question["question"],
            "options": json.dumps(question["options"]),
            "correct_answer": question["correct_answer"],
            
            # Base model outputs
            "base_model_full_message": base_full_message,
            "base_model_raw_token": base_raw,
            "base_model_clean": base_clean,
            "base_model_token_id": base_token_id,
            "base_model_answer": base_answer,
            "base_model_correct": base_answer == question["correct_answer"],
            
            # P1-A1 model outputs
            "p1_a1_full_message": p1_a1_full_message,
            "p1_a1_raw_token": p1_a1_raw,
            "p1_a1_clean": p1_a1_clean,
            "p1_a1_token_id": p1_a1_token_id,
            "p1_a1_answer": p1_a1_answer,
            "p1_a1_correct": p1_a1_answer == question["correct_answer"],
        }
        
        results.append(result)
        
        # Clear cache after each question if in low-memory mode
        if low_memory and device == "cuda":
            torch.cuda.empty_cache()
        
        if (idx + 1) % 10 == 0:
            base_acc = sum(1 for r in results if r["base_model_correct"]) / len(results)
            p1_a1_acc = sum(1 for r in results if r["p1_a1_correct"]) / len(results)
            logger.info(f"  Progress: {idx + 1}/{len(questions)} | Base: {base_acc:.1%} | P1-A1: {p1_a1_acc:.1%}")
    
    return results


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV
    """
    if not results:
        logger.warning("No results to save")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get field names from first result
    fieldnames = list(results[0].keys())
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics.
    
    Args:
        results: List of result dictionaries
    """
    total = len(results)
    if total == 0:
        return
    
    base_correct = sum(1 for r in results if r["base_model_correct"])
    p1_a1_correct = sum(1 for r in results if r["p1_a1_correct"])
    
    base_acc = base_correct / total
    p1_a1_acc = p1_a1_correct / total
    
    logger.info("\n" + "=" * 60)
    logger.info("MMLU-PRO SIDE-BY-SIDE EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Questions: {total}")
    logger.info(f"Base Model Accuracy: {base_acc:.2%} ({base_correct}/{total})")
    logger.info(f"P1-A1 Model Accuracy: {p1_a1_acc:.2%} ({p1_a1_correct}/{total})")
    logger.info(f"Accuracy Difference: {(p1_a1_acc - base_acc):+.2%}")
    
    # Agreement stats
    both_correct = sum(1 for r in results if r["base_model_correct"] and r["p1_a1_correct"])
    both_wrong = sum(1 for r in results if not r["base_model_correct"] and not r["p1_a1_correct"])
    base_correct_p1_wrong = sum(1 for r in results if r["base_model_correct"] and not r["p1_a1_correct"])
    p1_correct_base_wrong = sum(1 for r in results if not r["base_model_correct"] and r["p1_a1_correct"])
    
    logger.info("\nAgreement Breakdown:")
    logger.info(f"  Both correct: {both_correct} ({both_correct/total:.1%})")
    logger.info(f"  Both wrong: {both_wrong} ({both_wrong/total:.1%})")
    logger.info(f"  Base correct, P1-A1 wrong: {base_correct_p1_wrong} ({base_correct_p1_wrong/total:.1%})")
    logger.info(f"  P1-A1 correct, Base wrong: {p1_correct_base_wrong} ({p1_correct_base_wrong/total:.1%})")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side MMLU-Pro evaluation: base model vs P1-A1"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to P1-A1 config YAML file"
    )
    parser.add_argument(
        "--p1-a1-checkpoint", type=str, required=True,
        help="Path to P1-A1 checkpoint file"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to evaluate on (cuda/cpu)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"Number of MMLU-Pro samples to evaluate (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--output", type=str, default="results/mmlu_pro_side_by_side.csv",
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="Enable aggressive memory optimization for limited VRAM (13GB)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum new tokens to generate per question (default: 256)"
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
    
    # Load tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load MMLU-Pro test questions
    questions = load_mmlu_pro_test(
        split="test",
        num_samples=args.num_samples,
        seed=args.seed,
    )
    
    # Create models
    logger.info("\n" + "=" * 60)
    logger.info("Loading models...")
    logger.info("=" * 60)
    
    logger.info("Creating base model (full Qwen)...")
    base_model = create_base_model(config, device)
    
    logger.info("Creating P1-A1 model...")
    p1_a1_model = create_p1_a1_model(config, device, args.p1_a1_checkpoint)
    
    # Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Running side-by-side evaluation...")
    if args.low_memory:
        logger.info("Low memory mode: Enabled (for 13GB VRAM)")
    logger.info(f"Batch size: 1 (processing single samples)")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("=" * 60)
    
    results = run_side_by_side_evaluation(
        base_model=base_model,
        p1_a1_model=p1_a1_model,
        tokenizer=tokenizer,
        questions=questions,
        device=device,
        low_memory=args.low_memory,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results_to_csv(results, args.output)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

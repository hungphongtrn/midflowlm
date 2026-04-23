#!/usr/bin/env python3
"""Side-by-side MMLU-Pro evaluation comparing base model vs trained models.

Creates a CSV with:
- input: The question text
- base_model_full_message: Chat template + special tokens from base Qwen model
- p1_a1_T1_full_message: Chat template + special tokens from P1-A1 (T=1)
- p1_a2_T1_full_message: Chat template + special tokens from P1-A2 (T=1)
- p1_a2_T2_full_message: Chat template + special tokens from P1-A2 (T=2)
- p1_a2_T4_full_message: Chat template + special tokens from P1-A2 (T=4)
- p1_a2_T8_full_message: Chat template + special tokens from P1-A2 (T=8)
- p1_a3_T1_full_message: Chat template + special tokens from P1-A3 (T=1)
- p1_a3_T2_full_message: Chat template + special tokens from P1-A3 (T=2)
- p1_a3_T4_full_message: Chat template + special tokens from P1-A3 (T=4)
- p1_a3_T8_full_message: Chat template + special tokens from P1-A3 (T=8)

Usage:
    # Initial run: Evaluate base model + P1-A1
    python scripts/eval_mmlu_pro_side_by_side.py \
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml \
        --p1-a1-checkpoint ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt \
        --num-steps 1 \
        --num-samples 72 \
        --output results/mmlu_pro_side_by_side.csv

    # Append mode: Add P1-A2 to existing CSV (skip base model, it's already there)
    python scripts/eval_mmlu_pro_side_by_side.py \
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml \
        --p1-a2-checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \
        --num-steps 1 2 4 8 \
        --num-samples 72 \
        --output results/mmlu_pro_side_by_side.csv \
        --append results/mmlu_pro_side_by_side.csv \
        --skip-base

    # Append mode: Add P1-A3 to existing CSV
    python scripts/eval_mmlu_pro_side_by_side.py \
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml \
        --p1-a3-checkpoint ./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \
        --num-steps 1 2 4 8 \
        --num-samples 72 \
        --output results/mmlu_pro_side_by_side.csv \
        --append results/mmlu_pro_side_by_side.csv \
        --skip-base

    # Full run: Compare all experiments at once
    python scripts/eval_mmlu_pro_side_by_side.py \
        --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml \
        --p1-a2-checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \
        --p1-a3-checkpoint ./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \
        --num-steps 1 2 4 8 \
        --num-samples 72 \
        --output results/mmlu_pro_side_by_side.csv

================================================================================
EXPERIMENT CONFIGURATIONS
================================================================================

P1-A1: One-shot Projector (T=1 only)
  Checkpoint: ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt
  Config: configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml
  Architecture: projector (single-step only)
  Training: T=1, Endpoint + KL loss (1.0/0.0/0.5/0.0), Mix B
  Eval T values: [1]

P1-A2: Shared Recurrent Residual (T ∈ [2,4,6,8] during training)
  Checkpoint: ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt
  Config: configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml
  W&B Run: ze54okvs (stilted-paper-3)
  Architecture: shared_recurrent_residual
  Training: Multi-step with T ∈ [2,4,6,8], Endpoint + KL loss (1.0/0.0/0.5/0.0), Mix B
  Eval T values: [1, 2, 4, 8]

P1-A3: Flow Midblock (T ∈ [2,4,6,8] during training)
  Checkpoint: ./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt
  Config: configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml
  W&B Run: 5q0mthbl (major-gorge-4)
  Architecture: flow_midblock with timestep_plus_layer_boundary conditioning
  Training: Flow Midblock, continuous time sampling, T ∈ [2,4,6,8], Endpoint + KL loss (1.0/0.0/0.5/0.0), Mix B
  Eval T values: [1, 2, 4, 8]

================================================================================
"""

import argparse
import csv
import gc
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


def aggressive_gpu_cleanup(device: str):
    """Aggressively clean up GPU memory to prevent OOM and illegal memory access.

    This function:
    1. Synchronizes CUDA to ensure all operations complete
    2. Deletes any lingering tensors on GPU
    3. Runs Python garbage collection
    4. Clears CUDA cache
    5. Synchronizes again to ensure cleanup is complete

    Args:
        device: Device string ("cuda" or "cpu")
    """
    if device != "cuda" or not torch.cuda.is_available():
        return

    # Synchronize to ensure all CUDA operations complete
    torch.cuda.synchronize()

    # Collect garbage to free any circular references
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Synchronize again to ensure cache clearing is complete
    torch.cuda.synchronize()

    # Log memory status for debugging
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.debug(f"GPU cleanup complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


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
    match = re.search(r"(?:the\s+)?answer\s+is\s+\(?([a-j])\)?", text_lower)
    if match:
        answer = match.group(1).upper()
        if answer.lower() in valid_set:
            return answer

    # Pattern 2: "answer: X" or "Answer: X" (case insensitive)
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
    try:
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

                # Synchronize to catch any CUDA errors early
                if device == "cuda":
                    torch.cuda.synchronize()

                # Get next token prediction (greedy)
                next_token_logits = logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Safely get token ID - move to CPU first to avoid illegal memory access
                token_id = int(next_token.cpu().item())

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

                # Delete intermediate tensors to free memory
                del logits, next_token_logits, next_token, new_mask
                if device == "cuda":
                    torch.cuda.synchronize()

        # Synchronize before decoding
        if device == "cuda":
            torch.cuda.synchronize()

        # Decode all generated tokens using CPU tensor to save GPU memory
        generated_tensor_cpu = torch.tensor([generated_tokens])
        raw_output_with_special = tokenizer.decode(generated_tensor_cpu[0], skip_special_tokens=False)
        clean_output = tokenizer.decode(generated_tensor_cpu[0], skip_special_tokens=True)

        # Cleanup
        del generated_tensor_cpu
        del current_input_ids, current_attention_mask, input_ids, attention_mask

    except RuntimeError as e:
        logger.error(f"CUDA error during generation: {e}")
        # Cleanup on error
        aggressive_gpu_cleanup(device)
        raise

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


def create_trained_model(
    config: dict,
    device: str,
    checkpoint_path: str,
) -> FrozenQwenStudent:
    """Create a trained student model from checkpoint.

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
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle trainer checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint (global_step={checkpoint.get('global_step', 'N/A')})")
    else:
        # Try loading as midblock state dict directly
        model.load_midblock(checkpoint_path)

    return model


def load_existing_results(csv_path: str) -> List[Dict[str, Any]]:
    """Load existing results from CSV file.

    Args:
        csv_path: Path to existing CSV file

    Returns:
        List of result dictionaries from existing CSV
    """
    results = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert boolean strings back to actual booleans
            for key in row:
                if row[key].lower() in ('true', 'false'):
                    row[key] = row[key].lower() == 'true'
            results.append(row)
    return results


def run_side_by_side_evaluation(
    base_model: torch.nn.Module,
    models: Dict[str, torch.nn.Module],  # model_name -> model
    num_steps_list: List[int],
    tokenizer: AutoTokenizer,
    questions: List[Dict[str, Any]],
    device: str,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    skip_base: bool = False,
    low_memory: bool = False,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Run side-by-side evaluation on all questions.

    Processes one question at a time (batch size 1) to minimize VRAM usage.
    This allows the evaluation to run alongside training (13GB used) without OOM.

    Args:
        base_model: Base Qwen model
        models: Dictionary of trained models (name -> model)
        num_steps_list: List of T values to evaluate for multi-step models
        tokenizer: Tokenizer
        questions: List of questions
        device: Device
        existing_results: Optional existing results to merge with
        skip_base: If True, don't re-evaluate base model (use existing results)
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

        # Initialize result with question info
        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "input": question["question"],
            "options": json.dumps(question["options"]),
            "correct_answer": question["correct_answer"],
        }

        # If we have existing results and skip_base is True, copy base model data
        if existing_results and skip_base:
            existing_row = None
            for row in existing_results:
                if int(row.get("question_id", -1)) == question["question_id"]:
                    existing_row = row
                    break

            if existing_row:
                # Copy base model columns from existing results
                base_columns = [k for k in existing_row.keys() if k.startswith("base_model_")]
                for col in base_columns:
                    result[col] = existing_row[col]
                logger.info(f"  Using existing base model results for question {question['question_id']}")
            else:
                logger.warning(f"  No existing base model results found for question {question['question_id']}, generating new...")
                skip_base = False  # Fall back to generating

        # Generate with base model if not skipping
        if not skip_base:
            base_raw, base_clean, base_token_id = generate_with_model(
                base_model, tokenizer, prompt, device, num_steps=1, is_student=True,
                max_new_tokens=max_new_tokens
            )

            # Extract base answer
            valid_options = [chr(ord("A") + i) for i in range(len(question["options"]))]
            base_answer = extract_answer(base_clean, valid_options)

            # Add base model results
            result["base_model_full_message"] = prompt + base_raw
            result["base_model_raw_token"] = base_raw
            result["base_model_clean"] = base_clean
            result["base_model_token_id"] = base_token_id
            result["base_model_answer"] = base_answer
            result["base_model_correct"] = base_answer == question["correct_answer"]

        # Clear cache if using CUDA to free memory for trained models
        aggressive_gpu_cleanup(device)

        # Generate with each trained model at each T value
        for model_name, model in models.items():
            for num_steps in num_steps_list:
                logger.info(f"  Running {model_name} with T={num_steps}...")

                model_raw, model_clean, model_token_id = generate_with_model(
                    model, tokenizer, prompt, device, num_steps=num_steps, is_student=True,
                    max_new_tokens=max_new_tokens
                )

                # Extract answer
                valid_options = [chr(ord("A") + i) for i in range(len(question["options"]))]
                model_answer = extract_answer(model_clean, valid_options)

                # Add to result with dynamic column names
                prefix = f"{model_name}_T{num_steps}"
                result[f"{prefix}_full_message"] = prompt + model_raw
                result[f"{prefix}_raw_token"] = model_raw
                result[f"{prefix}_clean"] = model_clean
                result[f"{prefix}_token_id"] = model_token_id
                result[f"{prefix}_answer"] = model_answer
                result[f"{prefix}_correct"] = model_answer == question["correct_answer"]

                # Clear cache after each T value
                aggressive_gpu_cleanup(device)

            # Clear cache after each model in low-memory mode
            if low_memory:
                aggressive_gpu_cleanup(device)

        results.append(result)

        # Clear cache after each question in low-memory mode
        if low_memory:
            aggressive_gpu_cleanup(device)

        if (idx + 1) % 10 == 0:
            # Calculate accuracies so far
            base_acc = sum(1 for r in results if r.get("base_model_correct", False)) / len(results)
            logger.info(f"  Progress: {idx + 1}/{len(questions)} | Base: {base_acc:.1%}")
            for model_name in models.keys():
                for num_steps in num_steps_list:
                    key = f"{model_name}_T{num_steps}_correct"
                    if key in results[0]:
                        model_acc = sum(1 for r in results if r.get(key, False)) / len(results)
                        logger.info(f"    {model_name} T={num_steps}: {model_acc:.1%}")

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


def print_summary(results: List[Dict[str, Any]], models: Dict[str, torch.nn.Module], num_steps_list: List[int]):
    """Print summary statistics.

    Args:
        results: List of result dictionaries
        models: Dictionary of trained models (name -> model)
        num_steps_list: List of T values evaluated
    """
    total = len(results)
    if total == 0:
        return

    # Base model accuracy
    base_correct = sum(1 for r in results if r.get("base_model_correct", False))
    base_acc = base_correct / total if total > 0 else 0.0

    logger.info("\n" + "=" * 60)
    logger.info("MMLU-PRO SIDE-BY-SIDE EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Questions: {total}")
    logger.info(f"Base Model Accuracy: {base_acc:.2%} ({base_correct}/{total})")
    logger.info("")

    # Trained models accuracy
    logger.info("Trained Models:")
    for model_name in models.keys():
        logger.info(f"\n  {model_name}:")
        for num_steps in num_steps_list:
            key = f"{model_name}_T{num_steps}_correct"
            if key in results[0]:
                correct = sum(1 for r in results if r.get(key, False))
                acc = correct / total
                diff = acc - base_acc
                logger.info(f"    T={num_steps}: {acc:.2%} ({correct}/{total}) [{diff:+.2%} vs base]")

    # Agreement stats for each model at each T
    logger.info("\n" + "=" * 60)
    logger.info("AGREEMENT BREAKDOWN")
    logger.info("=" * 60)

    for model_name in models.keys():
        for num_steps in num_steps_list:
            correct_key = f"{model_name}_T{num_steps}_correct"
            if correct_key not in results[0]:
                continue

            both_correct = sum(1 for r in results if r.get("base_model_correct", False) and r.get(correct_key, False))
            both_wrong = sum(1 for r in results if not r.get("base_model_correct", False) and not r.get(correct_key, False))
            base_correct_model_wrong = sum(1 for r in results if r.get("base_model_correct", False) and not r.get(correct_key, False))
            model_correct_base_wrong = sum(1 for r in results if not r.get("base_model_correct", False) and r.get(correct_key, False))

            logger.info(f"\n{model_name} T={num_steps}:")
            logger.info(f"  Both correct: {both_correct} ({both_correct/total:.1%})")
            logger.info(f"  Both wrong: {both_wrong} ({both_wrong/total:.1%})")
            logger.info(f"  Base correct, {model_name} wrong: {base_correct_model_wrong} ({base_correct_model_wrong/total:.1%})")
            logger.info(f"  {model_name} correct, Base wrong: {model_correct_base_wrong} ({model_correct_base_wrong/total:.1%})")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side MMLU-Pro evaluation: base model vs trained models",
        epilog="""
Examples:
  # Initial run: Evaluate base model + P1-A1
  python scripts/eval_mmlu_pro_side_by_side.py \\
    --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a1_proj_mixb_endkl.yaml \\
    --p1-a1-checkpoint ./outputs/midflow_qwen_8to11_p1_a1_proj_mixb_endkl/checkpoints/best.ckpt \\
    --num-steps 1 \\
    --num-samples 72 \\
    --output results/mmlu_pro_side_by_side.csv

  # Append mode: Add P1-A2 to existing CSV (skip base model, it's already there)
  python scripts/eval_mmlu_pro_side_by_side.py \\
    --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml \\
    --p1-a2-checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \\
    --num-steps 1 2 4 8 \\
    --num-samples 72 \\
    --output results/mmlu_pro_side_by_side.csv \\
    --append results/mmlu_pro_side_by_side.csv \\
    --skip-base

  # Append mode: Add P1-A3 to existing CSV
  python scripts/eval_mmlu_pro_side_by_side.py \\
    --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468.yaml \\
    --p1-a3-checkpoint ./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \\
    --num-steps 1 2 4 8 \\
    --num-samples 72 \\
    --output results/mmlu_pro_side_by_side.csv \\
    --append results/mmlu_pro_side_by_side.csv \\
    --skip-base

  # Full run: Compare all experiments at once (fresh start)
  python scripts/eval_mmlu_pro_side_by_side.py \\
    --config configs/v0_1_matrix/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468.yaml \\
    --p1-a2-checkpoint ./outputs/midflow_qwen_8to11_p1_a2_rrb_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \\
    --p1-a3-checkpoint ./outputs/midflow_qwen_8to11_p1_a3_flow_mixb_endkl_trainT_r2468/checkpoints/best.ckpt \\
    --num-steps 1 2 4 8 \\
    --num-samples 72 \\
    --output results/mmlu_pro_side_by_side.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML file (used for base model and architecture info)"
    )
    parser.add_argument(
        "--p1-a1-checkpoint", type=str, default=None,
        help="Path to P1-A1 checkpoint file (optional)"
    )
    parser.add_argument(
        "--p1-a2-checkpoint", type=str, default=None,
        help="Path to P1-A2 checkpoint file (optional)"
    )
    parser.add_argument(
        "--p1-a3-checkpoint", type=str, default=None,
        help="Path to P1-A3 checkpoint file (optional)"
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
        "--num-steps", type=int, nargs="+", default=[1, 8],
        help="T values to evaluate for multi-step models (default: 1 8)"
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
        "--append", type=str, default=None,
        help="Path to existing CSV to append new results to (merges columns)"
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip base model evaluation (use existing results from --append file)"
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

    # Validate that at least one checkpoint is provided
    if not args.p1_a1_checkpoint and not args.p1_a2_checkpoint and not args.p1_a3_checkpoint:
        logger.error("At least one checkpoint must be provided (--p1-a1-checkpoint, --p1-a2-checkpoint, or --p1-a3-checkpoint)")
        sys.exit(1)

    # Load existing results if --append is specified
    existing_results = None
    if args.append:
        if not Path(args.append).exists():
            logger.error(f"Append file not found: {args.append}")
            sys.exit(1)
        logger.info(f"Loading existing results from {args.append}")
        existing_results = load_existing_results(args.append)
        logger.info(f"Loaded {len(existing_results)} existing rows")

        if args.skip_base:
            # Verify that base model results exist in the existing file
            if existing_results and not any(k.startswith("base_model_") for k in existing_results[0].keys()):
                logger.error("Existing file doesn't contain base model results, cannot use --skip-base")
                sys.exit(1)
            logger.info("Will skip base model evaluation and use existing results")

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Log T values
    logger.info(f"Evaluating T values: {args.num_steps}")

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

    base_model = None
    if not args.skip_base:
        logger.info("Creating base model (full Qwen)...")
        base_model = create_base_model(config, device)
    else:
        logger.info("Skipping base model (using existing results from append file)")

    # Load trained models
    models = {}
    if args.p1_a1_checkpoint:
        logger.info("Creating P1-A1 model (One-shot Projector)...")
        models["p1_a1"] = create_trained_model(config, device, args.p1_a1_checkpoint)

    if args.p1_a2_checkpoint:
        logger.info("Creating P1-A2 model (Shared Recurrent Residual)...")
        models["p1_a2"] = create_trained_model(config, device, args.p1_a2_checkpoint)

    if args.p1_a3_checkpoint:
        logger.info("Creating P1-A3 model (Flow Midblock)...")
        models["p1_a3"] = create_trained_model(config, device, args.p1_a3_checkpoint)

    # Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Running side-by-side evaluation...")
    if args.low_memory:
        logger.info("Low memory mode: Enabled (for 13GB VRAM)")
    if args.skip_base:
        logger.info("Base model: Using existing results")
    logger.info(f"Batch size: 1 (processing single samples)")
    logger.info(f"Models: {list(models.keys())}")
    logger.info(f"T values: {args.num_steps}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("=" * 60)

    results = run_side_by_side_evaluation(
        base_model=base_model,
        models=models,
        num_steps_list=args.num_steps,
        tokenizer=tokenizer,
        questions=questions,
        device=device,
        existing_results=existing_results,
        skip_base=args.skip_base,
        low_memory=args.low_memory,
        max_new_tokens=args.max_new_tokens,
    )

    # Print summary
    print_summary(results, models, args.num_steps)

    # Save results
    save_results_to_csv(results, args.output)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

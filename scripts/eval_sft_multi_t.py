#!/usr/bin/env python3
"""Multi-T evaluation of SFTFlowMidblockModel on MMLU-Pro.

Evaluates the trained FlowMidblock at multiple thinking levels (T values)
and computes accuracy, oracle accuracy, and prediction-change rate.

Usage:
    python scripts/eval_sft_multi_t.py \
        --checkpoint outputs/issue-9/sft_flow_midblock_3060_smoke/midblock_final.pth \
        --num-steps 1 4 8 16 32 \
        --num-samples 50
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

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.sft_flow_midblock import SFTFlowMidblockModel
from src.eval.sft_metrics import (
    SFTSampleResult,
    MultiTReport,
    compute_multi_t_report,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-T SFT evaluation on MMLU-Pro")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to midblock checkpoint (.pth)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B", help="Base model name")
    parser.add_argument("--start-layer", type=int, default=8)
    parser.add_argument("--end-layer", type=int, default=11)
    parser.add_argument("--num-steps", type=int, nargs="+", default=[1, 4, 8, 16, 32], help="T values to evaluate")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of MMLU-Pro questions")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to generate per question")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/issue-9/eval", help="Output directory")
    parser.add_argument("--experiment-id", type=str, default="sft-flow-midblock-smoke", help="Experiment identifier")
    return parser.parse_args()


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_mmlu_pro(num_samples: int, seed: int):
    logger.info(f"Loading MMLU-Pro validation (n={num_samples})")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    questions = []
    for idx in indices:
        item = dataset[idx]
        options_raw = item.get("options", [])
        options = []
        for opt in options_raw:
            match = re.match(r"^([A-J])\.\s*(.+)$", opt.strip())
            if match:
                options.append(match.group(2))
            else:
                options.append(opt)
        questions.append({
            "question": item["question"],
            "options": options,
            "correct_answer": item["answer"],
            "category": item.get("category", "unknown"),
        })
    logger.info(f"Loaded {len(questions)} questions")
    return questions


def create_prompt(question: str, options: list, tokenizer) -> str:
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join(f"{l}. {o}" for l, o in zip(option_letters, options))
    user_content = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, etc.).

Question: {question}

Options:
{options_text}

Answer:"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer multiple choice questions with only the letter of the correct answer."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str, num_options: int) -> str:
    """Extract answer letter from model output."""
    valid_letters = [chr(ord("A") + i) for i in range(num_options)]
    text_upper = text.strip().upper()

    for pattern in [
        r"\*\*([A-J])\*\*",
        r"answer[:\s]+([A-J])",
        r"option[:\s]+([A-J])",
        r"\b([A-J])\b",
    ]:
        matches = re.findall(pattern, text_upper)
        for m in matches:
            if m in valid_letters:
                return valid_letters[valid_letters.index(m)]
    return "INVALID"


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load model
    logger.info("Loading SFTFlowMidblockModel...")
    model = SFTFlowMidblockModel(
        model_name=args.model_name,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        thinking_level=32,
        checkpoint_path=args.checkpoint,
    )
    model.to(device)
    model.eval()
    logger.info(f"Trainable params: {model.trainable_params:,}")
    logger.info(f"Frozen params:    {model.frozen_params:,}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load questions
    questions = load_mmlu_pro(args.num_samples, args.seed)

    # Evaluate at each T
    all_results = []
    t_values = sorted(args.num_steps)

    for t in t_values:
        logger.info(f"Evaluating T={t}...")
        model.thinking_level = t
        num_correct = 0

        for qid, q in enumerate(questions):
            prompt = create_prompt(q["question"], q["options"], tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.time()
            with torch.no_grad():
                outputs = model.qwen.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latency_ms = (time.time() - t0) * 1000

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted = extract_answer(generated_text, len(q["options"]))
            is_correct = predicted == q["correct_answer"]

            if is_correct:
                num_correct += 1

            result = SFTSampleResult(
                question_id=qid,
                question=q["question"],
                correct_answer=q["correct_answer"],
                predicted_answer=predicted,
                is_correct=is_correct,
                num_steps=t,
                latency_ms=latency_ms,
                category=q["category"],
            )
            all_results.append(result)

            if (qid + 1) % 10 == 0 and qid > 0:
                logger.info(f"  T={t}: {qid+1}/{len(questions)} — {num_correct}/{qid+1} correct")

        acc = num_correct / len(questions) if questions else 0
        logger.info(f"  T={t} final: {acc:.1%} ({num_correct}/{len(questions)})")

    # Compute report
    report = compute_multi_t_report(all_results, t_values, experiment_id=args.experiment_id)

    # Print summary
    print("\n" + report.summary())

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{args.experiment_id}_results.json"
    with open(json_path, "w") as f:
        json.dump({"report": report.to_dict(), "details": [r.to_dict() for r in all_results]}, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")

    # CSV
    csv_path = output_dir / f"{args.experiment_id}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "category", "num_steps", "predicted_answer",
            "correct_answer", "is_correct", "latency_ms",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "question_id": r.question_id,
                "category": r.category,
                "num_steps": r.num_steps,
                "predicted_answer": r.predicted_answer,
                "correct_answer": r.correct_answer,
                "is_correct": r.is_correct,
                "latency_ms": r.latency_ms,
            })
    logger.info(f"Saved CSV: {csv_path}")

    return report


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_evaluation(args)

#!/usr/bin/env python3
"""Evaluate P3-D3 on MMLU (original, 4-option)."""

import argparse
import csv
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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.student_qwen import FrozenQwenStudent


@dataclass
class BenchmarkResult:
    question: str
    options: List[str]
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    num_steps: int
    model_name: str
    prompt_text: str
    prompt_tokens: List[int]
    raw_output_token: int
    raw_output_text: str
    category: str = "unknown"
    experiment_id: str = ""

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
            "experiment_id": self.experiment_id,
        }


@dataclass
class BenchmarkReport:
    accuracy: float
    num_correct: int
    num_total: int
    model_name: str
    num_steps: int
    avg_latency_ms: float
    detailed_results: List[Dict[str, Any]]
    experiment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "model_name": self.model_name,
            "num_steps": self.num_steps,
            "avg_latency_ms": self.avg_latency_ms,
            "detailed_results": self.detailed_results,
            "experiment_id": self.experiment_id,
        }

    def summary(self) -> str:
        return (
            f"Model: {self.model_name} (T={self.num_steps})"
            f"{' [' + self.experiment_id + ']' if self.experiment_id else ''}\n"
            f"Accuracy: {self.accuracy:.2%} ({self.num_correct}/{self.num_total})\n"
            f"Avg Latency: {self.avg_latency_ms:.2f} ms"
        )


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_mmlu_samples(
    num_samples: int = 500,
    seed: int = 42,
    subjects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    dataset = load_dataset("cais/mmlu", "all", split="test")
    random.seed(seed)

    if subjects:
        filtered = []
        for i, item in enumerate(dataset):
            if item.get("subject", "") in subjects:
                filtered.append(i)
        indices = random.sample(filtered, min(num_samples, len(filtered)))
    else:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    questions = []
    for idx in indices:
        item = dataset[idx]
        choices = item["choices"]
        if len(choices) != 4:
            continue
        questions.append({
            "question": item["question"],
            "options": choices,
            "correct_answer": item["answer"],
            "category": item.get("subject", "unknown"),
        })
    return questions


def create_mmlu_prompt(question: str, options: List[str], tokenizer: AutoTokenizer) -> str:
    option_letters = ["A", "B", "C", "D"][:len(options)]
    options_text = "\n".join(
        [f"{letter}. {opt}" for letter, opt in zip(option_letters, options)]
    )
    user_content = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer.

Question: {question}

Options:
{options_text}

Answer:"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers multiple choice questions. Respond with only the letter of the correct answer.",
        },
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def extract_answer(text: str, valid_options: List[str]) -> str:
    text_lower = text.strip().lower()
    valid_set = set(opt.lower() for opt in valid_options)

    match = re.search(r"\*\*\s*(?:answer\s*[:is]+\s*)?\(?([a-d])\)?\s*\*\*", text_lower)
    if match and match.group(1).upper().lower() in valid_set:
        return match.group(1).upper()

    match = re.search(r"(?:the\s+)?answer\s+is\s+\(?([a-d])\)?", text_lower)
    if match and match.group(1).upper().lower() in valid_set:
        return match.group(1).upper()

    match = re.search(r"answer:\s*\(?([a-d])\)?", text_lower)
    if match and match.group(1).upper().lower() in valid_set:
        return match.group(1).upper()

    if len(text_lower) > 0 and text_lower[0] in valid_set:
        return text_lower[0].upper()

    match = re.match(r"^[\(\[]?([a-d])[\)\]\.]?\s*", text_lower)
    if match and match.group(1) in valid_set:
        return match.group(1).upper()

    return "INVALID"


def generate_autoregressive(
    model, input_ids, attention_mask, num_steps, tokenizer,
    max_new_tokens=256, eos_token_id=None, is_student=True,
) -> Tuple[torch.Tensor, float]:
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    generated_tokens = []
    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        for _ in range(max_new_tokens):
            if is_student:
                logits = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    num_steps=num_steps,
                )
            else:
                logits = model(current_input_ids, num_steps=num_steps)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            if next_token.item() == eos_token_id:
                break
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            new_mask = torch.ones(
                (current_attention_mask.size(0), 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
            if current_input_ids.size(1) > 2048:
                current_input_ids = current_input_ids[:, -1024:]
                current_attention_mask = current_attention_mask[:, -1024:]
    end_time = time.perf_counter()
    return torch.tensor([generated_tokens], device=input_ids.device), (end_time - start_time) * 1000


def evaluate_model_on_mmlu(
    model, tokenizer, questions, num_steps, device, model_name,
    is_student=True, max_new_tokens=256, experiment_id="",
) -> BenchmarkReport:
    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluating {model_name} with T={num_steps} on {len(questions)} MMLU questions"
    )
    results = []
    latencies = []
    for idx, q in enumerate(questions):
        prompt = create_mmlu_prompt(q["question"], q["options"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generated_ids, latency = generate_autoregressive(
            model, input_ids, attention_mask, num_steps, tokenizer,
            max_new_tokens=max_new_tokens, is_student=is_student,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        valid_opts = [chr(ord("A") + i) for i in range(len(q["options"]))]
        predicted = extract_answer(generated_text, valid_opts)

        correct_answer = q["correct_answer"]
        if isinstance(correct_answer, int):
            correct_answer = chr(ord("A") + correct_answer)

        result = BenchmarkResult(
            question=q["question"],
            options=q["options"],
            correct_answer=correct_answer,
            predicted_answer=predicted,
            is_correct=(predicted == correct_answer),
            num_steps=num_steps,
            model_name=model_name,
            prompt_text=prompt,
            prompt_tokens=input_ids[0].tolist(),
            raw_output_token=int(generated_ids[0][0]) if len(generated_ids[0]) > 0 else -1,
            raw_output_text=raw_text,
            category=q.get("category", "unknown"),
            experiment_id=experiment_id,
        )
        results.append(result)
        latencies.append(latency)

    num_correct = sum(1 for r in results if r.is_correct)
    accuracy = num_correct / len(results) if results else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return BenchmarkReport(
        accuracy=accuracy, num_correct=num_correct, num_total=len(results),
        model_name=model_name, num_steps=num_steps, avg_latency_ms=avg_latency,
        detailed_results=[r.to_dict() for r in results], experiment_id=experiment_id,
    )


def create_student_model(config, device, checkpoint_path=None):
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
        logger = logging.getLogger(__name__)
        cp = Path(checkpoint_path)
        if not cp.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using uninitialized student model")
            return model
        checkpoint = torch.load(str(cp), map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate P3-D3 on MMLU")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-steps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default="P3-D3")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = load_mmlu_samples(num_samples=args.num_samples, seed=args.seed)
    model = create_student_model(config, device, args.checkpoint)
    all_results = []

    for T in args.num_steps:
        report = evaluate_model_on_mmlu(
            model, tokenizer, questions, T, device, "trained_midblock",
            experiment_id=args.experiment_id, max_new_tokens=args.max_new_tokens,
        )
        logger.info(f"\n{report.summary()}")
        all_results.append(report.to_dict())

    if not args.skip_teacher:
        teacher = FrozenQwenStudent(
            model_name=config["model"]["name"],
            start_layer=config["replacement_model"]["start_layer"],
            end_layer=config["replacement_model"]["end_layer"],
            max_steps_T=config["model"]["max_steps_T"],
            device=device, dtype=torch.float32, bypass_mode=True,
        )
        report = evaluate_model_on_mmlu(
            teacher, tokenizer, questions, 1, device, "teacher_original",
            experiment_id="teacher", max_new_tokens=args.max_new_tokens,
        )
        logger.info(f"\n{report.summary()}")
        all_results.append(report.to_dict())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    if args.csv_output:
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment_id", "model_name", "num_steps",
                "accuracy", "num_correct", "num_total", "avg_latency_ms",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    "experiment_id": args.experiment_id,
                    "model_name": r["model_name"],
                    "num_steps": r["num_steps"],
                    "accuracy": f"{r['accuracy']:.4f}",
                    "num_correct": r["num_correct"],
                    "num_total": r["num_total"],
                    "avg_latency_ms": f"{r['avg_latency_ms']:.2f}",
                })

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

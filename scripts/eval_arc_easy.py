#!/usr/bin/env python3
"""Evaluate P3-D3 on ARC-Easy science reasoning."""

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
from typing import Dict, List, Any, Tuple

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
    raw_output_text: str
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
            "raw_output_text": self.raw_output_text,
            "experiment_id": self.experiment_id,
        }


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_arc_easy_val(num_samples: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    questions = []
    for idx in indices:
        item = dataset[idx]
        choices_texts = item["choices"]["text"]
        questions.append({
            "question": item["question"],
            "options": choices_texts,
            "correct_answer": item["answerKey"],
            "category": "arc_easy",
        })
    return questions


def create_arc_prompt(question: str, options: List[str], tokenizer: AutoTokenizer) -> str:
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join(
        [f"{letter}. {opt}" for letter, opt in zip(option_letters, options)]
    )
    user_content = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer.

Question: {question}

Options:
{options_text}

Answer:"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Respond with only the letter of the correct answer."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str, valid_options: List[str]) -> str:
    text_lower = text.strip().lower()
    valid_set = set(opt.lower() for opt in valid_options)

    for pattern in [
        r"\*\*\s*(?:answer\s*[:is]+\s*)?\(?([a-d])\)?\s*\*\*",
        r"(?:the\s+)?answer\s+is\s+\(?([a-d])\)?",
        r"answer:\s*\(?([a-d])\)?",
    ]:
        match = re.search(pattern, text_lower)
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
    max_new_tokens=128, eos_token_id=None, is_student=True,
) -> Tuple[torch.Tensor, float]:
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    generated_tokens = []
    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        cur_ids, cur_mask = input_ids, attention_mask
        for _ in range(max_new_tokens):
            if is_student:
                logits = model(input_ids=cur_ids, attention_mask=cur_mask, num_steps=num_steps)
            else:
                logits = model(cur_ids, num_steps=num_steps)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            if next_token.item() == eos_token_id:
                break
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones((cur_mask.size(0), 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
            if cur_ids.size(1) > 2048:
                cur_ids, cur_mask = cur_ids[:, -1024:], cur_mask[:, -1024:]
    return torch.tensor([generated_tokens], device=input_ids.device), (time.perf_counter() - start_time) * 1000


def evaluate_model_on_arc_easy(
    model, tokenizer, questions, num_steps, device, model_name,
    is_student=True, max_new_tokens=128, experiment_id="",
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    results = []
    latencies = []
    for q in questions:
        prompt = create_arc_prompt(q["question"], q["options"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        gen_ids, latency = generate_autoregressive(
            model, inputs["input_ids"].to(device), inputs["attention_mask"].to(device),
            num_steps, tokenizer, max_new_tokens=max_new_tokens, is_student=is_student,
        )
        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        valid_opts = [chr(ord("A") + i) for i in range(len(q["options"]))]
        predicted = extract_answer(generated_text, valid_opts)

        results.append(BenchmarkResult(
            question=q["question"], options=q["options"],
            correct_answer=q["correct_answer"], predicted_answer=predicted,
            is_correct=(predicted == q["correct_answer"]), num_steps=num_steps,
            model_name=model_name, prompt_text=prompt,
            raw_output_text=generated_text, experiment_id=experiment_id,
        ))
        latencies.append(latency)

    num_correct = sum(1 for r in results if r.is_correct)
    return {
        "accuracy": num_correct / len(results) if results else 0.0,
        "num_correct": num_correct, "num_total": len(results),
        "model_name": model_name, "num_steps": num_steps,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "detailed_results": [r.to_dict() for r in results],
        "experiment_id": experiment_id,
    }


def create_student_model(config, device, checkpoint_path=None):
    mc = config["model"]; rc = config["replacement_model"]
    model = FrozenQwenStudent(
        model_name=mc["name"], start_layer=rc["start_layer"],
        end_layer=rc["end_layer"], max_steps_T=mc["max_steps_T"],
        device=device, dtype=torch.float32, bypass_mode=False,
    )
    if checkpoint_path:
        logger = logging.getLogger(__name__)
        cp = Path(checkpoint_path)
        if not cp.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using uninitialized student model")
            return model
        ckpt = torch.load(str(cp), map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate P3-D3 on ARC-Easy")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-steps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default="P3-D3")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.log_level)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = load_arc_easy_val(num_samples=args.num_samples, seed=args.seed)
    model = create_student_model(config, device, args.checkpoint)
    all_results = []

    for T in args.num_steps:
        report = evaluate_model_on_arc_easy(
            model, tokenizer, questions, T, device, "trained_midblock",
            experiment_id=args.experiment_id, max_new_tokens=args.max_new_tokens,
        )
        logger.info(f"T={T}: {report['accuracy']:.2%} ({report['num_correct']}/{report['num_total']})")
        all_results.append(report)

    if not args.skip_teacher:
        teacher = FrozenQwenStudent(
            model_name=config["model"]["name"],
            start_layer=config["replacement_model"]["start_layer"],
            end_layer=config["replacement_model"]["end_layer"],
            max_steps_T=config["model"]["max_steps_T"],
            device=device, dtype=torch.float32, bypass_mode=True,
        )
        report = evaluate_model_on_arc_easy(
            teacher, tokenizer, questions, 1, device, "teacher_original",
            experiment_id="teacher", max_new_tokens=args.max_new_tokens,
        )
        logger.info(f"Teacher: {report['accuracy']:.2%} ({report['num_correct']}/{report['num_total']})")
        all_results.append(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    if args.csv_output:
        with open(args.csv_output, "w", newline="") as f:
            w = csv.DictWriter(f, ["experiment_id","model_name","num_steps","accuracy","num_correct","num_total","avg_latency_ms"])
            w.writeheader()
            for r in all_results:
                w.writerow({"experiment_id":args.experiment_id,"model_name":r["model_name"],"num_steps":r["num_steps"],"accuracy":f"{r['accuracy']:.4f}","num_correct":r["num_correct"],"num_total":r["num_total"],"avg_latency_ms":f"{r['avg_latency_ms']:.2f}"})

    logger.info("ARC-Easy evaluation complete!")


if __name__ == "__main__":
    main()

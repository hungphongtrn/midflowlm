"""Behavior-oriented MMLU-Pro inference utilities."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.eval.text_checkpoint_sweep import (
    create_student,
    load_student_checkpoint,
    validate_num_steps,
)


logger = logging.getLogger(__name__)


PromptBehavior = Literal["default", "stripped", "closed_think"]


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|begin_of_text|>"
    return tokenizer


def load_mmlu_pro_val(
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    full_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)
    if num_samples is not None:
        indices = indices[:num_samples]
    questions = []
    for idx in indices:
        item = full_dataset[idx]
        # Normalize options - strip trailing periods
        options = [opt.strip().rstrip(".") for opt in item["options"]]
        questions.append(
            {
                "question": item["question"],
                "options": options,
                "correct_answer": item["answer"],
                "category": item.get("category", "unknown"),
            }
        )
    logger.info("Loaded %s questions from MMLU-Pro", len(questions))
    return questions


def create_mmlu_pro_prompt(
    question: str,
    options: list[str],
    tokenizer: PreTrainedTokenizerBase,
    prompt_behavior: PromptBehavior = "default",
) -> str:
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join(
        f"{letter}. {opt}" for letter, opt in zip(option_letters, options)
    )
    user_content = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, etc.).

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
    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if prompt_behavior == "stripped":
        # Strip any existing think tags from the prompt
        prompt = re.sub(
            r"<\|im_start\|>assistant\n<think>\n.*?</think>\n",
            "",
            prompt,
            flags=re.DOTALL,
        )
        prompt = (
            "<|im_start|>user\n"
            + prompt
            + "<|im_start|>assistant\n"
            + prompt
        )
    elif prompt_behavior == "closed_think":
        prompt = (
            "<|im_start|>assistant\n<think>\n"
            + prompt
            + "<|im_start|>assistant\n"
            + prompt
        )
    else:  # "default" - same as closed_think for backward compatibility
        prompt = (
            "<|im_start|>assistant\n<think>\n"
            + prompt
            + "<|im_start|>assistant\n"
            + prompt
        )
    return prompt


def extract_first_valid_answer(text: str, valid_options: list[str]) -> Optional[str]:
    normalized = text.strip().upper()
    valid_set = set(valid_options)

    if normalized and normalized[0] in valid_set:
        return normalized[0]

    start_match = re.match(r"^[\s\(\[]*([A-J])[\)\]:\.\s-]", normalized)
    if start_match and start_match.group(1) in valid_set:
        return start_match.group(1)

    for opt in valid_options:
        pattern = r"(?:(?<=^)|(?<=\s))" + re.escape(opt) + r"(?=\s|$|[.,!?;:()\[\]])";
        if re.search(pattern, normalized):
            return opt

    return None


def generate_behavior_completion(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    num_steps: int,
    max_new_tokens: int,
    is_student: bool,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_on_eos: bool = True,
    solver_method: str = "euler",
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    completion_token_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    stopped_on_eos = False

    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if is_student:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=num_steps,
                    solver_method=solver_method,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=num_steps,
                    solver_method=solver_method,
                )

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token_id = int(next_token.item())
            completion_token_ids.append(next_token_id)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(next_token, device=attention_mask.device),
                ],
                dim=1,
            )

            if (
                stop_on_eos
                and eos_token_id is not None
                and next_token_id == eos_token_id
            ):
                stopped_on_eos = True
                break

    latency_ms = (time.perf_counter() - start_time) * 1000
    generated_text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
    first_generated_text = (
        tokenizer.decode([completion_token_ids[0]], skip_special_tokens=False)
        if completion_token_ids
        else ""
    )

    return {
        "prompt_token_ids": encoded["input_ids"][0].tolist(),
        "generated_token_ids": completion_token_ids,
        "generated_text": generated_text,
        "first_generated_text": first_generated_text,
        "stopped_on_eos": stopped_on_eos,
        "latency_ms": latency_ms,
    }


def build_behavior_record(
    *,
    sample_index: int,
    question: dict[str, Any],
    prompt_text: str,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_text: str,
    first_generated_text: str,
    model_name: str,
    checkpoint_path: Optional[str],
    num_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stopped_on_eos: bool,
) -> dict[str, Any]:
    valid_options = [chr(ord("A") + i) for i in range(len(question["options"]))]
    first_answer_letter = extract_first_valid_answer(generated_text, valid_options)
    return {
        "sample_index": sample_index,
        "category": question.get("category", "unknown"),
        "question": question["question"],
        "options": question["options"],
        "correct_answer": question["correct_answer"],
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "num_steps": num_steps,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "completion_length": len(generated_token_ids),
        "first_generated_text": first_generated_text,
        "first_answer_letter": first_answer_letter,
        "found_valid_answer": first_answer_letter is not None,
        "stopped_on_eos": stopped_on_eos,
    }


def summarize_behavior_records(
    records: list[dict[str, Any]], example_count: int = 3
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_name"], int(record["num_steps"]))].append(record)

    summaries = []
    for (model_name, num_steps), group in sorted(grouped.items()):
        first_text_counts = Counter(record["first_generated_text"] for record in group)
        answer_counts = Counter(
            record["first_answer_letter"]
            for record in group
            if record["first_answer_letter"] is not None
        )
        hit_count = sum(1 for record in group if record["found_valid_answer"])
        examples = [record["generated_text"] for record in group[:example_count]]
        summaries.append(
            {
                "model_name": model_name,
                "num_steps": num_steps,
                "sample_count": len(group),
                "answer_hit_rate": hit_count / len(group) if group else 0.0,
                "top_first_generated_texts": [
                    list(item) for item in first_text_counts.most_common(5)
                ],
                "top_answer_letters": [
                    list(item) for item in answer_counts.most_common(5)
                ],
                "example_completions": examples,
            }
        )
    return summaries


def format_behavior_summary(summaries: list[dict[str, Any]]) -> str:
    lines = []
    for summary in summaries:
        lines.append(
            f"{summary['model_name']:20s} T={summary['num_steps']:2d} | "
            f"samples={summary['sample_count']:3d} | "
            f"answer-hit={summary['answer_hit_rate']:.2%}"
        )
        if summary["top_first_generated_texts"]:
            lines.append(f"  first texts: {summary['top_first_generated_texts'][:3]}")
        if summary["top_answer_letters"]:
            lines.append(f"  answer letters: {summary['top_answer_letters'][:3]}")
        for example in summary["example_completions"]:
            lines.append(f"  example: {example[:160]}")
    return "\n".join(lines)


def run_mmlu_pro_behavior_observation(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
    output_path: str | Path,
    num_samples: int = 20,
    seed: int = 42,
    num_steps: Optional[list[int]] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    include_teacher: bool = True,
    solver_method: str = "euler",
    stop_on_eos: bool = True,
    prompt_behavior: PromptBehavior = "default",
) -> dict[str, Any]:
    config = load_config(config_path)
    max_steps_t = int(config["model"]["max_steps_T"])
    num_steps = num_steps or [1, max_steps_t]
    warnings = validate_num_steps(num_steps, max_steps_t)
    tokenizer = create_tokenizer(config["model"]["name"])
    questions = load_mmlu_pro_val(num_samples=num_samples, seed=seed)

    trained_model = create_student(config, device=device, bypass_mode=False)
    checkpoint_info = load_student_checkpoint(trained_model, checkpoint_path)

    teacher_model = None
    if include_teacher:
        teacher_model = create_student(config, device=device, bypass_mode=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with output_file.open("w", encoding="utf-8") as handle:
        for sample_index, question in enumerate(questions):
            prompt_text = create_mmlu_pro_prompt(
                question["question"],
                question["options"],
                tokenizer,
                prompt_behavior=prompt_behavior,
            )

            for step_count in num_steps:
                generation = generate_behavior_completion(
                    trained_model,
                    tokenizer,
                    prompt_text,
                    num_steps=step_count,
                    max_new_tokens=max_new_tokens,
                    is_student=True,
                    temperature=temperature,
                    top_p=top_p,
                    stop_on_eos=stop_on_eos,
                    solver_method=solver_method,
                )
                record = build_behavior_record(
                    sample_index=sample_index,
                    question=question,
                    prompt_text=prompt_text,
                    prompt_token_ids=generation["prompt_token_ids"],
                    generated_token_ids=generation["generated_token_ids"],
                    generated_text=generation["generated_text"],
                    first_generated_text=generation["first_generated_text"],
                    model_name="trained_midblock",
                    checkpoint_path=str(checkpoint_path),
                    num_steps=step_count,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stopped_on_eos=generation["stopped_on_eos"],
                )
                record["latency_ms"] = generation["latency_ms"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

            if teacher_model is not None:
                generation = generate_behavior_completion(
                    teacher_model,
                    tokenizer,
                    prompt_text,
                    num_steps=1,
                    max_new_tokens=max_new_tokens,
                    is_student=False,
                    temperature=temperature,
                    top_p=top_p,
                    stop_on_eos=stop_on_eos,
                    solver_method=solver_method,
                )
                record = build_behavior_record(
                    sample_index=sample_index,
                    question=question,
                    prompt_text=prompt_text,
                    prompt_token_ids=generation["prompt_token_ids"],
                    generated_token_ids=generation["generated_token_ids"],
                    generated_text=generation["generated_text"],
                    first_generated_text=generation["first_generated_text"],
                    model_name="teacher_original",
                    checkpoint_path=None,
                    num_steps=1,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stopped_on_eos=generation["stopped_on_eos"],
                )
                record["latency_ms"] = generation["latency_ms"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

    summaries = summarize_behavior_records(records)
    return {
        "config_path": str(config_path),
        "checkpoint": checkpoint_info,
        "output_path": str(output_file),
        "num_samples": num_samples,
        "seed": seed,
        "device": device,
        "num_steps": num_steps,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "solver_method": solver_method,
        "warnings": warnings,
        "summaries": summaries,
    }

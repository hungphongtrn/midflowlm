"""Utilities for side-by-side text generation checkpoint sweeps."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.model.student_qwen import FrozenQwenStudent


DEFAULT_TEXTS = [
    "Once upon a time in a small village,",
    "The robot looked at the night sky and wondered",
    "Write a short story about a brave cat who",
]


@dataclass
class SweepResult:
    prompt: str
    num_steps: int
    generated_text: str
    full_text: str
    completion_token_ids: list[int]
    prompt_length: int
    completion_length: int
    stopped_on_eos: bool


def compute_repetition_metrics(
    text: str, n_values: tuple[int, ...] = (2, 3, 4)
) -> dict[str, float]:
    """Compute n-gram repetition ratios for text.

    Args:
        text: Input text to analyze
        n_values: Tuple of n-gram sizes to analyze

    Returns:
        Dict mapping 'repeat_{n}gram_ratio' to ratio of repeated n-grams
    """
    words = text.split()
    if len(words) < 2:
        return {f"repeat_{n}gram_ratio": 0.0 for n in n_values}

    metrics = {}
    for n in n_values:
        if len(words) < n:
            metrics[f"repeat_{n}gram_ratio"] = 0.0
            continue

        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            metrics[f"repeat_{n}gram_ratio"] = 0.0
            continue

        counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(count - 1 for count in counts.values() if count > 1)

        ratio = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        metrics[f"repeat_{n}gram_ratio"] = ratio

    return metrics


def aggregate_repetition_metrics(
    comparisons: list[dict[str, Any]], n_values: tuple[int, ...] = (2, 3, 4)
) -> dict[str, float]:
    """Compute aggregate repetition metrics across comparisons.

    Args:
        comparisons: List of comparison dicts with 'generated_text' keys
        n_values: Tuple of n-gram sizes to analyze

    Returns:
        Dict mapping 'mean_repeat_{n}gram_ratio' to mean ratio
    """
    if not comparisons:
        return {f"mean_repeat_{n}gram_ratio": 0.0 for n in n_values}

    all_metrics = []
    for comparison in comparisons:
        text = comparison.get("generated_text", "")
        metrics = compute_repetition_metrics(text, n_values)
        all_metrics.append(metrics)

    aggregated = {}
    for n in n_values:
        key = f"repeat_{n}gram_ratio"
        mean_value = sum(m[key] for m in all_metrics) / len(all_metrics)
        aggregated[f"mean_repeat_{n}gram_ratio"] = mean_value

    return aggregated


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_texts(
    cli_texts: Optional[list[str]] = None, text_file: Optional[str] = None
) -> list[str]:
    texts = [text.strip() for text in (cli_texts or []) if text.strip()]

    if text_file:
        with open(text_file, "r", encoding="utf-8") as handle:
            texts.extend(line.strip() for line in handle if line.strip())

    return texts or list(DEFAULT_TEXTS)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_student(
    config: dict[str, Any], device: str, bypass_mode: bool = False
) -> FrozenQwenStudent:
    model_config = config["model"]
    replacement_config = config["replacement_model"]
    return FrozenQwenStudent(
        model_name=model_config["name"],
        start_layer=replacement_config["start_layer"],
        end_layer=replacement_config["end_layer"],
        max_steps_T=model_config["max_steps_T"],
        device=device,
        dtype=torch.float32,
        bypass_mode=bypass_mode,
    )


def load_student_checkpoint(
    model: FrozenQwenStudent, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=model.device, weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return {
            "checkpoint_format": "trainer_checkpoint",
            "global_step": checkpoint.get("global_step"),
            "current_epoch": checkpoint.get("current_epoch"),
            "path": str(checkpoint_path),
        }

    if isinstance(checkpoint, dict) and any(
        key.startswith("midblock.") for key in checkpoint
    ):
        model.load_state_dict(checkpoint, strict=False)
        return {
            "checkpoint_format": "model_state_dict",
            "path": str(checkpoint_path),
        }

    if model.midblock is None:
        raise ValueError("Model has no midblock to load")

    model.midblock.load_state_dict(checkpoint)
    return {
        "checkpoint_format": "midblock_state_dict",
        "path": str(checkpoint_path),
    }


def greedy_generate(
    model: FrozenQwenStudent,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    num_steps: int,
    max_new_tokens: int,
    stop_on_eos: bool = True,
    solver_method: str = "euler",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> SweepResult:
    """Generate text with optional temperature sampling.

    Args:
        model: The student model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_steps: Number of ODE solver steps
        max_new_tokens: Maximum tokens to generate
        stop_on_eos: Whether to stop at EOS token
        solver_method: ODE solver method
        temperature: Sampling temperature (0 = greedy, >0 = sample)
        top_p: Nucleus sampling threshold (1.0 = disabled)
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    completion_token_ids: list[int] = []
    stopped_on_eos = False
    eos_token_id = tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=num_steps,
                solver_method=solver_method,
            )
            next_token_logits = logits[:, -1, :]

            # Apply temperature sampling if temperature > 0
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
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

    completion_text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return SweepResult(
        prompt=prompt,
        num_steps=num_steps,
        generated_text=completion_text,
        full_text=full_text,
        completion_token_ids=completion_token_ids,
        prompt_length=int(encoded["input_ids"].shape[-1]),
        completion_length=len(completion_token_ids),
        stopped_on_eos=stopped_on_eos,
    )


def validate_num_steps(num_steps: list[int], max_steps_T: int) -> list[str]:
    warnings = []
    for step_count in num_steps:
        if step_count <= 0:
            raise ValueError("All num_steps values must be positive integers")
        if step_count > max_steps_T:
            warnings.append(
                f"Requested num_steps={step_count} exceeds max_steps_T={max_steps_T}. "
                "This is allowed, but discrete step embeddings clamp to the final trained step, "
                "so results are extrapolative and may degrade."
            )
    return warnings


def build_comparison_rows(
    original_model: FrozenQwenStudent,
    trained_model: FrozenQwenStudent,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    num_steps: list[int],
    max_new_tokens: int,
    solver_method: str = "euler",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list[dict[str, Any]]:
    rows = []
    for text in texts:
        original_result = asdict(
            greedy_generate(
                original_model,
                tokenizer,
                text,
                num_steps=1,
                max_new_tokens=max_new_tokens,
                solver_method="euler",
                temperature=temperature,
                top_p=top_p,
            )
        )
        trained_results = {
            str(step_count): asdict(
                greedy_generate(
                    trained_model,
                    tokenizer,
                    text,
                    num_steps=step_count,
                    max_new_tokens=max_new_tokens,
                    solver_method=solver_method,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            for step_count in num_steps
        }
        rows.append(
            {
                "input": text,
                "original_output": original_result["generated_text"],
                "original_details": original_result,
                "trained_outputs": {
                    step: result["generated_text"]
                    for step, result in trained_results.items()
                },
                "trained_details": trained_results,
            }
        )
    return rows


def build_text_table(rows: list[dict[str, Any]], num_steps: list[int]) -> str:
    headers = [
        "input",
        "original output",
        *[f"num_steps = {step_count}" for step_count in num_steps],
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["input"],
                row["original_output"],
                *[
                    row["trained_outputs"].get(str(step_count), "")
                    for step_count in num_steps
                ],
            ]
        )

    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for idx, value in enumerate(table_row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in table_rows)
    return "\n".join(lines)


def run_text_sweep(
    config_path: str | Path,
    checkpoint_path: str | Path,
    texts: list[str],
    num_steps: list[int],
    max_new_tokens: int,
    device: str,
    output_path: Optional[str | Path] = None,
    solver_method: str = "euler",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    config = load_config(config_path)
    max_steps_T = int(config["model"]["max_steps_T"])
    warnings = validate_num_steps(num_steps, max_steps_T)

    original_model = create_student(config, device=device, bypass_mode=True)
    trained_model = create_student(config, device=device, bypass_mode=False)
    tokenizer = create_tokenizer(config["model"]["name"])
    checkpoint_info = load_student_checkpoint(trained_model, checkpoint_path)

    comparisons = build_comparison_rows(
        original_model=original_model,
        trained_model=trained_model,
        tokenizer=tokenizer,
        texts=texts,
        num_steps=num_steps,
        max_new_tokens=max_new_tokens,
        solver_method=solver_method,
        temperature=temperature,
        top_p=top_p,
    )
    table = build_text_table(comparisons, num_steps)

    payload = {
        "config_path": str(config_path),
        "checkpoint": checkpoint_info,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "num_steps": num_steps,
        "max_steps_T": max_steps_T,
        "solver_method": solver_method,
        "temperature": temperature,
        "top_p": top_p,
        "warnings": warnings,
        "comparisons": comparisons,
        "repetition_metrics": aggregate_repetition_metrics(comparisons),
        "table": table,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload

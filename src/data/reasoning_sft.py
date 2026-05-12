from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.utils.dataset_processing import (
    create_assistant_only_labels,
    get_chat_template_separators,
    pack_tokenized_dataset,
)


def filter_by_token_budget(
    dataset: Dataset, max_total_tokens: int = 8192, **kwargs: Any
) -> Dataset:
    def _within_budget(example: dict[str, Any]) -> bool:
        meta = example.get("meta")
        if not isinstance(meta, dict):
            return True
        input_tokens = meta.get("input_tokens")
        output_tokens = meta.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return True
        return int(input_tokens) + int(output_tokens) <= max_total_tokens

    return dataset.filter(_within_budget, **kwargs)


def tokenize_reasoning_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    num_proc: int = 4,
    conversations_field: str = "conversations",
) -> Dataset:
    def _map_glm_roles(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
        mapped = []
        for turn in conversations:
            src_role = (turn.get("from") or "").lower()
            if src_role == "human":
                role = "user"
            elif src_role in {"gpt", "assistant"}:
                role = "assistant"
            else:
                role = src_role if src_role else "user"
            mapped.append({"role": role, "content": turn.get("value", "")})
        return mapped

    def _tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        messages = _map_glm_roles(example.get(conversations_field, []))
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = tokenizer(text, truncation=True, max_length=max_length).input_ids
        return {"input_ids": input_ids}

    tokenized = dataset.map(
        _tokenize,
        num_proc=num_proc,
        desc="Tokenizing GLM reasoning conversations",
    )

    instruction_part, response_part = get_chat_template_separators(tokenizer)
    return create_assistant_only_labels(
        tokenized,
        tokenizer,
        instruction_part=instruction_part,
        response_part=response_part,
        num_proc=num_proc,
    )


def create_reasoning_sft_datasets(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    num_proc: int = 4,
    val_split: float = 0.02,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 1337,
) -> tuple[Dataset, Dataset]:
    filtered = filter_by_token_budget(dataset, max_total_tokens=max_length)
    shuffled = filtered.shuffle(seed=seed)

    split = shuffled.train_test_split(test_size=val_split, seed=seed)
    train_raw = split["train"]
    eval_raw = split["test"]

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    train_tok = tokenize_reasoning_dataset(
        train_raw, tokenizer, max_length=max_length, num_proc=num_proc
    )
    eval_tok = tokenize_reasoning_dataset(
        eval_raw, tokenizer, max_length=max_length, num_proc=num_proc
    )

    train_packed = pack_tokenized_dataset(train_tok, max_seq_length=max_length)

    def _with_attention_mask(example: dict[str, Any]) -> dict[str, list[int]]:
        input_ids = example["input_ids"]
        return {"attention_mask": [1] * len(input_ids)}

    train_final = train_packed.map(_with_attention_mask, num_proc=num_proc)
    eval_final = eval_tok.map(_with_attention_mask, num_proc=num_proc)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_drop = [c for c in train_final.column_names if c not in keep_cols]
    eval_drop = [c for c in eval_final.column_names if c not in keep_cols]
    if train_drop:
        train_final = train_final.remove_columns(train_drop)
    if eval_drop:
        eval_final = eval_final.remove_columns(eval_drop)

    return train_final, eval_final

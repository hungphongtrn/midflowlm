from __future__ import annotations

import hashlib
import json
import logging
import os as _os
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

from src.utils.dataset_processing import (
    create_assistant_only_labels,
    get_chat_template_separators,
    pack_tokenized_dataset,
)


def _compute_cache_hash(data_cfg: dict) -> str:
    """Compute a deterministic hash of preprocessing parameters."""
    processing = data_cfg.get("processing", {})
    dataset_cfg = data_cfg.get("dataset", {})
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "dataset_name": dataset_cfg.get("name"),
                "dataset_split": dataset_cfg.get("split"),
                "num_proc": processing.get("num_proc"),
                "val_split": processing.get("val_split"),
                "max_train_samples": processing.get("max_train_samples"),
                "max_eval_samples": processing.get("max_eval_samples"),
                "packing_strategy": processing.get("packing_strategy"),
                "seed": processing.get("seed"),
                "max_seq_length": data_cfg.get("max_seq_length"),
            },
            sort_keys=True,
        ).encode()
    )
    return hasher.hexdigest()


def filter_by_token_budget(
    dataset: Dataset, max_total_tokens: int = 8192, **kwargs: Any
) -> Dataset:
    """Filter samples by input+output token budget.

    Additional keyword arguments are forwarded to ``Dataset.filter``.
    """

    def _within_budget(example: dict[str, Any]) -> bool:
        meta = example.get("meta")
        if not isinstance(meta, dict):
            return True
        input_tokens = meta.get("input_tokens")
        output_tokens = meta.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return True
        try:
            total_tokens = int(input_tokens) + int(output_tokens)
        except (TypeError, ValueError):
            # Keep malformed meta values to avoid dropping potentially useful
            # samples due to metadata issues.
            return True
        return total_tokens <= max_total_tokens

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
                role = "user"
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
    packing_strategy: str = "bfd",
) -> tuple[Dataset, Dataset]:
    if not 0 < val_split < 1:
        raise ValueError(
            f"val_split must be in the open interval (0, 1), got {val_split}."
        )

    filtered = filter_by_token_budget(dataset, max_total_tokens=max_length)
    if len(filtered) == 0:
        raise ValueError(
            "No samples available after token-budget filtering. "
            "Check max_length/max_total_tokens or input metadata."
        )

    shuffled = filtered.shuffle(seed=seed)

    split = shuffled.train_test_split(test_size=val_split, seed=seed)
    train_raw = split["train"]
    eval_raw = split["test"]

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    if len(train_raw) == 0 or len(eval_raw) == 0:
        raise ValueError(
            "Train/eval split produced an empty dataset after subsampling. "
            "Increase sample limits or adjust val_split."
        )

    train_tok = tokenize_reasoning_dataset(
        train_raw, tokenizer, max_length=max_length, num_proc=num_proc
    )
    eval_tok = tokenize_reasoning_dataset(
        eval_raw, tokenizer, max_length=max_length, num_proc=num_proc
    )

    pack_cols = [c for c in ("input_ids", "labels") if c in train_tok.column_names]
    train_for_pack = train_tok.select_columns(pack_cols)
    train_packed = pack_tokenized_dataset(
        train_for_pack,
        max_seq_length=max_length,
        packing_strategy=packing_strategy,
    )

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


def create_reasoning_sft_datasets_from_config(
    data_cfg: dict,
    tokenizer: PreTrainedTokenizer,
    force_reprocess: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load and preprocess reasoning SFT dataset from a data config section."""
    dataset_cfg = data_cfg["dataset"]
    processing = data_cfg["processing"]
    cache_dir = data_cfg["cache_dir"]
    max_seq_length = data_cfg["max_seq_length"]

    expected_hash = _compute_cache_hash(data_cfg)
    cache_info_path = _os.path.join(cache_dir, "cache_info.json")

    logger = logging.getLogger("src.data.reasoning_sft")

    if not force_reprocess and _os.path.exists(cache_info_path):
        try:
            with open(cache_info_path) as f:
                cached_info = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read cache info from %s: %s", cache_info_path, exc)
        else:
            if cached_info.get("config_hash") == expected_hash:
                train_path = _os.path.join(cache_dir, "train")
                eval_path = _os.path.join(cache_dir, "eval")
                if _os.path.exists(train_path) and _os.path.exists(eval_path):
                    try:
                        logger.info("Cache hit - loading preprocessed datasets from %s", cache_dir)
                        return load_from_disk(train_path), load_from_disk(eval_path)
                    except Exception as exc:
                        logger.warning(
                            "Failed to load cached datasets from %s: %s. Reprocessing.",
                            cache_dir,
                            exc,
                        )

    logger.info(
        "Loading dataset %s (split=%s) from HuggingFace Hub",
        dataset_cfg["name"],
        dataset_cfg["split"],
    )
    ds = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    logger.info("Loaded %s raw samples", f"{len(ds):,}")

    train_ds, eval_ds = create_reasoning_sft_datasets(
        ds,
        tokenizer,
        max_length=max_seq_length,
        num_proc=processing.get("num_proc", 4),
        val_split=processing.get("val_split", 0.02),
        max_train_samples=processing.get("max_train_samples"),
        max_eval_samples=processing.get("max_eval_samples"),
        seed=processing.get("seed", 1337),
        packing_strategy=processing.get("packing_strategy", "bfd"),
    )

    _os.makedirs(cache_dir, exist_ok=True)
    train_ds.save_to_disk(_os.path.join(cache_dir, "train"))
    eval_ds.save_to_disk(_os.path.join(cache_dir, "eval"))
    with open(cache_info_path, "w") as f:
        json.dump({"config_hash": expected_hash}, f)

    logger.info("Datasets cached to %s", cache_dir)
    return train_ds, eval_ds

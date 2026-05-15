"""
Dataset preprocessing utilities for CPT, SFT, and MIX training modes.

Handles tokenization, label masking (assistant-only loss for SFT cloned from
unsloth's train_on_responses_only), and packing with full deterministic control
-- all done before the data reaches Hugging Face Trainer.

Modes:
  CPT  -- plain text field, labels = input_ids (standard causal LM)
  SFT  -- messages format, chat template → tokenize → mask non-assistant tokens
  MIX  -- both pipelines → concatenate → pack
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def compute_fingerprint(**config_kwargs: Any) -> str:
    """Compute a deterministic short fingerprint from JSON-serializable kwargs."""
    import hashlib
    import json

    json_str = json.dumps(config_kwargs, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()[:16]


def apply_chat_template(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    chat_template: str | None = None,
    enable_thinking: bool = True,
    thinking_block_delimiters: List[str] | None = None,
) -> Dict[str, List[str]]:
    """Render batched ``messages`` examples into text using tokenizer chat template.

    This is a local fallback replacement for the previously external helper.
    """
    del chat_template, enable_thinking, thinking_block_delimiters

    messages_batch = examples.get("messages")
    if messages_batch is None:
        raise ValueError("Expected 'messages' column for SFT tokenization.")

    texts: List[str] = []
    for messages in messages_batch:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(rendered)
    return {"text": texts}


def load_and_format_dataset(
    ds_config: Dict[str, Any],
    max_samples: int | None = None,
    fingerprint: str | None = None,
) -> Dataset:
    """Load a dataset and normalize it to include a ``messages`` column.

    Supports datasets from Hugging Face Hub or local disk (Dataset/DatasetDict).
    """
    import os as _os
    from datasets import DatasetDict, load_dataset, load_from_disk

    del fingerprint

    path = ds_config["path"]
    split = ds_config.get("split", "train")
    disk = ds_config.get("disk", False)

    if disk or (isinstance(path, str) and _os.path.isdir(path)):
        ds_or_dict = load_from_disk(path)
        if isinstance(ds_or_dict, DatasetDict):
            if split not in ds_or_dict:
                split = next(iter(ds_or_dict.keys()))
            dataset = ds_or_dict[split]
        else:
            dataset = ds_or_dict
    else:
        dataset = load_dataset(path, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if "messages" in dataset.column_names:
        return dataset

    if "conversations" in dataset.column_names:
        def _map_conversations(example: Dict[str, Any]) -> Dict[str, Any]:
            mapped = []
            for turn in example.get("conversations", []) or []:
                src_role = str(turn.get("from", "")).lower()
                if src_role == "human":
                    role = "user"
                elif src_role in {"gpt", "assistant"}:
                    role = "assistant"
                else:
                    role = "user"
                mapped.append({"role": role, "content": turn.get("value", "")})
            return {"messages": mapped}

        return dataset.map(_map_conversations)

    raise ValueError(
        "Could not format dataset to messages format. "
        f"Available columns: {list(dataset.column_names)}"
    )


# ============================================================================
# Longest Common Sublist (cloned from unsloth)
# ============================================================================


def _longest_common_sublist(lists: List[List[int]]) -> List[int]:
    """Find the longest common sublist among multiple lists via binary search."""
    if not lists:
        return []

    min_len = min(len(lst) for lst in lists)
    if min_len == 0:
        return []

    def _has_common_sublist(length: int) -> Tuple[bool, List[int]]:
        common = set()
        first = lists[0]
        for i in range(len(first) - length + 1):
            common.add(tuple(first[i : i + length]))
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i : i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        return True, list(common.pop())

    left, right = 1, min_len
    result: List[int] = []
    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = _has_common_sublist(mid)
        if exists:
            result = sublist
            left = mid + 1
        else:
            right = mid - 1
    return result


# ============================================================================
# Common Token ID Finder (cloned from unsloth)
# ============================================================================


def _find_common_token_ids(
    component: str,
    tokenizer: PreTrainedTokenizer,
    force_match: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """Convert a separator string to common token IDs with whitespace handling.

    Tokenizers can tokenize newlines/spaces inconsistently.  This function
    tokenizes the ``component`` with different whitespace permutations and
    finds the longest token ID subsequence common to all of them.

    Returns:
        (substring_ids, optional_left_ids, optional_right_ids)
        - substring_ids:   core token IDs for the separator
        - optional_left:   tokens that *may* precede the separator
        - optional_right:  tokens that *may* follow the separator
    """
    if component is None:
        return [], [], []
    right_text = ""
    if component.endswith(" "):
        right_text = " "
    elif component.endswith("\n"):
        right_text = "\n"
    left_text = ""
    if component.startswith(" "):
        left_text = " "
    elif component.startswith("\n"):
        left_text = "\n"
    stripped = component.strip()

    all_input_ids: List[List[int]] = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left * left_text + stripped + right * right_text
                all_input_ids.append(tokenizer(x, add_special_tokens=False).input_ids)
                x = left * "\n" + stripped + right * "\n"
                all_input_ids.append(tokenizer(x, add_special_tokens=False).input_ids)
    else:
        x = tokenizer(component, add_special_tokens=False).input_ids
        all_input_ids.append(x)

    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # Fix: if substring is [0] and original is a single token, use that token
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]

    # If all tokenizations are identical, use the original unchanged
    if (
        len(set(str(x) for x in all_input_ids)) == 1
        and len(all_input_ids[0]) + 1 == len(substring)
        and all_input_ids[0] == substring[:-1]
    ):
        substring = all_input_ids[0]

    # Get optional left / right tokens
    original = tokenizer(component, add_special_tokens=False).input_ids
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring:
            break
    optional_left = original[:j]
    optional_right = original[j + len(substring) :]
    return substring, optional_left, optional_right


def _find_sublist(haystack: List[int], needle: List[int]) -> int | None:
    """Return the start index of needle in haystack, or None."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


# ============================================================================
# Chat Template Separator Detection
# ============================================================================


KNOWN_SEPARATORS: Dict[str, Tuple[str, str]] = {
    # (instruction_part, response_part) -- turn-start text markers
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "phi": ("<|user|>\n", "<|assistant|>\n"),
    "deepseek": ("User: ", "Assistant: "),
}


def _auto_detect_separators(
    tokenizer: PreTrainedTokenizer,
) -> Tuple[str | None, str | None]:
    """Auto-detect instruction/response separators from chat template.

    Renders minimal single-message conversations and extracts the text
    preceding the message content for each role.
    """
    if tokenizer.chat_template is None:
        return None, None

    SENTINEL = "ZXSMARKERXZ"

    sentinel_ids = tokenizer(SENTINEL, add_special_tokens=False).input_ids

    # --- instruction_part (user) ---
    try:
        convo_user = [{"role": "user", "content": SENTINEL}]
        text_user = tokenizer.apply_chat_template(
            convo_user, tokenize=False, add_generation_prompt=False
        )
        ids_user = tokenizer(text_user, add_special_tokens=False).input_ids
        pos = _find_sublist(ids_user, sentinel_ids)
        instruction_part = tokenizer.decode(ids_user[:pos]) if pos is not None else None
    except Exception:
        instruction_part = None

    # --- response_part (assistant) ---
    try:
        convo_asst = [{"role": "assistant", "content": SENTINEL}]
        text_asst = tokenizer.apply_chat_template(
            convo_asst, tokenize=False, add_generation_prompt=False
        )
        ids_asst = tokenizer(text_asst, add_special_tokens=False).input_ids
        pos = _find_sublist(ids_asst, sentinel_ids)
        response_part = tokenizer.decode(ids_asst[:pos]) if pos is not None else None
    except Exception:
        response_part = None

    return instruction_part, response_part


def get_chat_template_separators(
    tokenizer: PreTrainedTokenizer,
    instruction_part: str | None = None,
    response_part: str | None = None,
) -> Tuple[str, str]:
    """Determine the instruction/response separator strings for SFT masking.

    Priority:
    1. Explicit user overrides
    2. Known model families (by model name)
    3. Auto-detect from chat template
    4. Fail with descriptive error

    Returns:
        (instruction_part, response_part) -- the text markers that precede
        user and assistant message content in the rendered chat template.
        e.g., for Qwen: ``('<|im_start|>user\\n', '<|im_start|>assistant\\n')``
    """
    if instruction_part and response_part:
        return instruction_part, response_part

    model_name = (getattr(tokenizer, "name_or_path", "") or "").lower()

    for prefix, (inst, resp) in KNOWN_SEPARATORS.items():
        if prefix in model_name:
            return inst, resp

    inst, resp = _auto_detect_separators(tokenizer)
    if inst and resp:
        print(f"  Auto-detected separators: {inst!r} / {resp!r}")
        return inst, resp

    raise ValueError(
        "Could not determine chat template separators for assistant-only loss.\n"
        f"Model: {model_name!r}\n"
        "Please set sft_response_separators in the training config:\n"
        "  training:\n"
        "    sft_response_separators:\n"
        '      instruction_part: "<|im_start|>user\\n"\n'
        '      response_part: "<|im_start|>assistant\\n"\n'
        f"Known model families: {list(KNOWN_SEPARATORS.keys())}"
    )


# ============================================================================
# Tokenization
# ============================================================================


def tokenize_sft_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_proc: int = 4,
    chat_template: str | None = None,
    enable_thinking: bool = True,
    thinking_block_delimiters: List[str] | None = None,
    fingerprint: str | None = None,
) -> Dataset:
    """Tokenize a messages-format dataset for SFT.

    Applies the chat template to render messages to text, then tokenizes.
    Returns a dataset with an ``input_ids`` column (no labels yet).
    """
    def _tokenize_fn(examples: Dict) -> Dict:
        result = apply_chat_template(
            examples,
            tokenizer,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
            thinking_block_delimiters=thinking_block_delimiters,
        )
        texts = result["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {"input_ids": tokenized["input_ids"]}

    map_kwargs = dict(
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing (SFT)",
    )
    if fingerprint:
        map_kwargs["new_fingerprint"] = fingerprint
        print(f"  [cache] fingerprint={fingerprint}")

    return dataset.map(_tokenize_fn, **map_kwargs)


def tokenize_cpt_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_proc: int = 4,
    text_field: str = "text",
    fingerprint: str | None = None,
) -> Dataset:
    """Tokenize a plain-text dataset for CPT.

    Labels = input_ids (standard causal language modeling on all tokens).
    Returns a dataset with ``input_ids`` and ``labels`` columns.
    """
    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not in dataset columns: "
            f"{list(dataset.column_names)}"
        )

    def _tokenize_fn(examples: Dict) -> Dict:
        tokenized = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length - 1,
            padding=False,
        )
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is None or endoftext_id == tokenizer.unk_token_id:
            endoftext_id = tokenizer.eos_token_id
        input_ids = [ids + [endoftext_id] for ids in tokenized["input_ids"]]
        # For CPT, all tokens get loss
        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }

    map_kwargs = dict(
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing (CPT)",
    )
    if fingerprint:
        map_kwargs["new_fingerprint"] = fingerprint
        print(f"  [cache] fingerprint={fingerprint}")

    return dataset.map(_tokenize_fn, **map_kwargs)


# ============================================================================
# Label Masking -- Assistant-Only Loss (cloned from unsloth)
# ============================================================================


def mask_instruction_tokens(
    examples: Dict[str, Any],
    A_first: int,
    A_must: List[int],
    A_left_reversed: List[int],
    A_right_forward: List[int],
    Q_first: int,
    Q_must: List[int],
    Q_left_reversed: List[int],
    Q_right_forward: List[int],
) -> Dict[str, Any]:
    """Create labels that mask out non-assistant (instruction) tokens.

    Cloned from unsloth's ``_train_on_responses_only`` inner function.
    This is designed to be called via ``dataset.map(..., batched=True)``.

    For each sample, finds all ``(assistant_end, next_user_start)`` spans and
    copies ``input_ids`` to ``labels`` for those spans; everything else → -100.

    Multi-turn friendly: trains on *all* assistant turns (not just the last).
    """
    input_ids_list = examples["input_ids"]
    use_tensors = isinstance(input_ids_list, torch.Tensor)
    if use_tensors:
        input_ids_list = input_ids_list.tolist()

    len_A_must = len(A_must)
    len_Q_must = len(Q_must)

    all_labels: List[List[int]] = []
    for input_ids in input_ids_list:
        n = len(input_ids)
        labels = [-100] * n
        n_minus_1 = n - 1
        j = 0

        spans: List[Tuple[int, int]] = []  # (assistant_end, next_user_start)

        while j < n:
            # Find <assistant> marker
            k = j + len_A_must
            if input_ids[j] == A_first and k <= n and input_ids[j:k] == A_must:
                # Backtrack for optional left tokens
                for opt_left in A_left_reversed:
                    if j < 1:
                        break
                    if opt_left == input_ids[j - 1]:
                        j -= 1
                    else:
                        break
                # Forward for optional right tokens
                for opt_right in A_right_forward:
                    if k >= n_minus_1:
                        break
                    if opt_right == input_ids[k + 1]:
                        k += 1
                    else:
                        break
                assistant_end = k
                j = assistant_end

                # Find next <user> marker (or end of sequence)
                while j < n:
                    _k = j + len_Q_must
                    if (j == n_minus_1) or (
                        input_ids[j] == Q_first
                        and _k <= n
                        and input_ids[j:_k] == Q_must
                    ):
                        # Backtrack optional left tokens
                        for opt_left in Q_left_reversed:
                            if j < 1:
                                break
                            if opt_left == input_ids[j - 1]:
                                j -= 1
                            else:
                                break
                        # Forward optional right tokens
                        for opt_right in Q_right_forward:
                            if _k >= n_minus_1:
                                break
                            if opt_right == input_ids[_k + 1]:
                                _k += 1
                            else:
                                break

                        if j != n_minus_1:
                            user_start = j
                            j = _k  # skip past Q_must
                        else:
                            user_start = n
                            _k = n

                        spans.append((assistant_end, user_start))
                        break
                    j += 1
            j += 1

        # Copy input_ids → labels for assistant spans
        for a_end, u_start in spans:
            labels[a_end:u_start] = input_ids[a_end:u_start]

        all_labels.append(labels)

    return {"labels": all_labels}


def create_assistant_only_labels(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    instruction_part: str | None,
    response_part: str | None,
    num_proc: int = 4,
    fingerprint: str | None = None,
) -> Dataset:
    """Apply assistant-only label masking to a tokenized SFT dataset.

    Requires the dataset to already have an ``input_ids`` column.
    Adds a ``labels`` column where non-assistant tokens are set to -100.

    If instruction_part or response_part is None, returns the dataset with
    labels copied from input_ids (no masking applied).
    """
    # Skip masking if separators are not available
    if instruction_part is None or response_part is None:
        print(
            "  Warning: instruction_part or response_part is None, skipping assistant-only masking"
        )
        print("  All tokens will be used for training (no -100 masking)")

        def _copy_input_ids_to_labels(example: Dict) -> Dict:
            input_ids = example.get("input_ids")
            if input_ids is None:
                return example
            if isinstance(input_ids, torch.Tensor):
                return {"labels": input_ids.clone()}
            return {"labels": list(input_ids)}

        map_kwargs = dict(
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Copying input_ids to labels (no masking)",
        )
        if fingerprint:
            map_kwargs["new_fingerprint"] = f"{fingerprint}_nomask"
            print(f"  [cache] fingerprint={fingerprint}_nomask")

        return dataset.map(_copy_input_ids_to_labels, **map_kwargs)

    # Find common token ID patterns for the separators
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer)
    A_must, A_left, A_right = _find_common_token_ids(response_part, tokenizer)

    A_first = A_must[0]
    A_left_rev = A_left[::-1]
    Q_first = Q_must[0]
    Q_left_rev = Q_left[::-1]

    fn_kwargs = dict(
        A_first=A_first,
        A_must=A_must,
        A_left_reversed=A_left_rev,
        A_right_forward=A_right,
        Q_first=Q_first,
        Q_must=Q_must,
        Q_left_reversed=Q_left_rev,
        Q_right_forward=Q_right,
    )

    # Apply masking via .map()
    map_kwargs = dict(
        batched=True,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Masking instruction tokens",
    )
    if fingerprint:
        map_kwargs["new_fingerprint"] = f"{fingerprint}_mask"
        print(f"  [cache] fingerprint={fingerprint}_mask")

    dataset = dataset.map(mask_instruction_tokens, **map_kwargs)

    # Filter out samples where all labels are -100 (no valid training signal)
    def _has_valid_labels(example: Dict) -> bool:
        labels = example.get("labels")
        if labels is None:
            return True
        if isinstance(labels, torch.Tensor):
            return bool((labels != -100).any().item())
        return any(lbl != -100 for lbl in labels)

    n_before = len(dataset)
    filter_kwargs = dict(
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Filtering fully masked samples",
    )
    if fingerprint:
        filter_kwargs["new_fingerprint"] = f"{fingerprint}_filter"
        print(f"  [cache] fingerprint={fingerprint}_filter")

    dataset = dataset.filter(_has_valid_labels, **filter_kwargs)
    n_removed = n_before - len(dataset)
    if n_removed > 0:
        print(
            f"  Removed {n_removed} samples where all labels were -100 "
            f"(response truncated or not found)"
        )

    return dataset


# ============================================================================
# Packing
# ============================================================================


def pack_tokenized_dataset(
    dataset: Dataset,
    max_seq_length: int,
    packing_strategy: str = "bfd",
    map_kwargs: Dict[str, Any] | None = None,
) -> Dataset:
    """Pack multiple tokenized samples into fixed-length sequences.

    Wraps TRL's ``pack_dataset``.  All columns present in ``dataset``
    (typically ``input_ids``, ``labels``, and optionally ``attention_mask``)
    are packed coherently.

    Args:
        dataset: Dataset with ``input_ids`` and ``labels`` columns.
        max_seq_length: Target sequence length.
        packing_strategy: One of ``"bfd"``, ``"bfd_split"``, ``"wrapped"``.
        map_kwargs: Extra kwargs for ``dataset.map()``.

    Returns:
        Packed dataset with sequences of length ``max_seq_length``.
    """
    from trl import pack_dataset

    if map_kwargs is None:
        map_kwargs = {}

    # Ensure required columns are present
    columns_to_keep = ["input_ids", "labels"]
    for col in columns_to_keep:
        if col not in dataset.column_names:
            raise ValueError(
                f"Column '{col}' not found in dataset for packing. "
                f"Available: {list(dataset.column_names)}"
            )

    return pack_dataset(
        dataset,
        seq_length=max_seq_length,
        strategy=packing_strategy,
        map_kwargs=map_kwargs,
    )


# ============================================================================
# Raw Data Loading (no tokenization)
# ============================================================================


def load_cpt_raw(
    df_config: Dict[str, Any],
    fingerprint: str | None = None,
) -> Dataset:
    """Load a raw CPT text dataset (no tokenization)."""
    import os as _os
    from datasets import load_dataset, load_from_disk, DatasetDict

    path = df_config["path"]
    split = df_config.get("split", "train")
    text_field = df_config.get("text_field", "text")
    disk = df_config.get("disk", False)

    if disk or (isinstance(path, str) and _os.path.isdir(path)):
        print(f"  Loading CPT dataset from disk: {path} (split={split})")
        ds_or_dict = load_from_disk(path)
        if isinstance(ds_or_dict, DatasetDict):
            if split not in ds_or_dict:
                available = list(ds_or_dict.keys())
                print(f"    Split '{split}' not found, available: {available}")
                split = available[0]
                print(f"    Using split: {split}")
            dataset = ds_or_dict[split]
        else:
            dataset = ds_or_dict
    else:
        print(f"  Loading CPT dataset: {path} (split={split})")
        dataset = load_dataset(path, split=split)

    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found. "
            f"Available: {list(dataset.column_names)}"
        )

    print(f"    Loaded {len(dataset):,} raw samples")
    return dataset


# ============================================================================
# Main Orchestrator
# ============================================================================


def prepare_preprocessed_dataset(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    verify_only: bool = False,
) -> Tuple[Dataset | None, Dataset | None]:
    """Load, split, subset, tokenize, mask labels, and optionally pack datasets.

    Pipeline order: Load raw → Split train/val → Subset via max_train/max_eval
    → Tokenize → (Pack train only)

    Routes to CPT, SFT, or MIX pipeline based on ``config.training_mode``.

    Returns:
        (train_dataset, eval_dataset) with ``input_ids`` + ``labels`` columns.
        Both are ``None`` if ``verify_only=True``.
    """
    training_mode = config.get("training_mode", "sft")
    dataset_config = config["dataset"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    max_length = model_cfg["max_seq_length"]
    num_proc = training_cfg.get("dataset_num_proc", 4)
    packing_enabled = training_cfg.get("packing", False)
    packing_strategy = training_cfg.get("packing_strategy", "bfd")
    preprocessing_num_workers = dataset_config.get("preprocessing_num_workers", 4)

    print("\n" + "=" * 80)
    print(f"PREPROCESSING DATASETS (mode={training_mode.upper()})")
    print("=" * 80)

    sep_override = training_cfg.get("sft_response_separators")
    instruction_part = None
    response_part = None
    if sep_override:
        instruction_part = sep_override.get("instruction_part")
        response_part = sep_override.get("response_part")

    # Auto-detect separators if not provided in config
    if instruction_part is None or response_part is None:
        try:
            instruction_part, response_part = get_chat_template_separators(
                tokenizer,
                instruction_part=instruction_part,
                response_part=response_part,
            )
        except ValueError as e:
            print(f"  Warning: Could not auto-detect separators: {e}")
            print("  SFT assistant-only masking will be skipped.")
            instruction_part = None
            response_part = None

    chat_template_name = model_cfg.get("chat_template")

    # Get thinking block delimiters from config (for models with thinking mode)
    thinking_block_delimiters = training_cfg.get("thinking_block_delimiters")

    val_ratio = dataset_config.get("validation_split_ratio", 0.02)
    max_train_samples = dataset_config.get("max_train_samples")
    max_eval_samples = dataset_config.get("max_eval_samples")

    # Budget tracking — subset BEFORE tokenization
    train_budget = max_train_samples if max_train_samples else 10**12
    eval_budget = max_eval_samples if max_eval_samples else 10**12

    tokenized_train_parts: List[Dataset] = []
    tokenized_val_parts: List[Dataset] = []

    # =====================================================================
    # CPT — Load raw → split → subset → tokenize
    # =====================================================================
    cpt_datasets = dataset_config.get("cpt_datasets", [])
    if training_mode in ("cpt", "mix") and cpt_datasets:
        print(f"\n--- CPT Datasets ({len(cpt_datasets)} source(s)) ---")
        raw_cpt_trains: List[Dataset] = []
        raw_cpt_vals: List[Dataset] = []

        for cpt_ds in cpt_datasets:
            try:
                raw = load_cpt_raw(cpt_ds)
                if len(raw) == 0:
                    continue

                val_size = max(1, int(len(raw) * val_ratio))
                if len(raw) > val_size:
                    raw_cpt_vals.append(raw.select(range(val_size)))
                    raw_cpt_trains.append(raw.select(range(val_size, len(raw))))
                else:
                    raw_cpt_vals.append(raw)
                    raw_cpt_trains.append(raw)
            except Exception as e:
                print(f"    Error loading CPT {cpt_ds.get('name', '?'):} {e}")
                import traceback

                traceback.print_exc()

        if raw_cpt_trains:
            cpt_train_raw = concatenate_datasets(raw_cpt_trains)
            cpt_train_raw = cpt_train_raw.shuffle(
                seed=dataset_config.get("interleave_seed", 3407)
            )
            cpt_val_raw = concatenate_datasets(raw_cpt_vals)
            cpt_val_raw = cpt_val_raw.shuffle(seed=3407)

            n_train = min(len(cpt_train_raw), train_budget)
            n_val = min(len(cpt_val_raw), eval_budget)

            if n_train > 0:
                cpt_fp = compute_fingerprint(
                    op="tokenize_cpt",
                    dataset="cpt_pool",
                    max_length=max_length,
                    tokenizer=getattr(tokenizer, "name_or_path", ""),
                    n_samples=n_train,
                )
                tok_cpt_train = tokenize_cpt_dataset(
                    cpt_train_raw.select(range(n_train)),
                    tokenizer,
                    max_length=max_length,
                    num_proc=num_proc,
                    fingerprint=cpt_fp,
                )
                tokenized_train_parts.append(tok_cpt_train)
                train_budget -= n_train
                print(f"    CPT train: {len(tok_cpt_train):,} tokenized samples")

            if n_val > 0:
                cpt_val_fp = compute_fingerprint(
                    op="tokenize_cpt",
                    dataset="cpt_pool_val",
                    max_length=max_length,
                    tokenizer=getattr(tokenizer, "name_or_path", ""),
                    n_samples=n_val,
                )
                tok_cpt_val = tokenize_cpt_dataset(
                    cpt_val_raw.select(range(n_val)),
                    tokenizer,
                    max_length=max_length,
                    num_proc=num_proc,
                    fingerprint=cpt_val_fp,
                )
                tokenized_val_parts.append(tok_cpt_val)
                eval_budget -= n_val
                print(f"    CPT val:   {len(tok_cpt_val):,} tokenized samples")

    # =====================================================================
    # SFT — Load raw → format → split → subset → tokenize + mask
    # =====================================================================
    sft_datasets = dataset_config.get("sft_datasets") or dataset_config.get(
        "train_datasets", []
    )
    if training_mode in ("sft", "mix") and sft_datasets:
        print(f"\n--- SFT Datasets ({len(sft_datasets)} source(s)) ---")
        raw_sft_trains: List[Dataset] = []
        raw_sft_vals: List[Dataset] = []

        for sft_ds in sft_datasets:
            try:
                sft_fp = compute_fingerprint(
                    op="load_format_sft",
                    dataset=sft_ds.get("path", ""),
                    formatter=sft_ds.get("formatter", "chat"),
                )
                formatted = load_and_format_dataset(
                    sft_ds, max_samples=None, fingerprint=sft_fp
                )
                if formatted is None or len(formatted) == 0:
                    continue

                def _has_valid_messages(example):
                    msgs = example.get("messages")
                    if not msgs or not isinstance(msgs, list):
                        return False
                    return any(
                        m.get("content") and str(m.get("content", "")).strip()
                        for m in msgs
                    )

                filter_empty_kwargs = dict(
                    num_proc=num_proc,
                    load_from_cache_file=True,
                    desc=f"Filtering empty messages {sft_ds.get('name', sft_ds['path'])}",
                )
                formatted = formatted.filter(_has_valid_messages, **filter_empty_kwargs)

                if len(formatted) == 0:
                    continue

                val_size = max(1, int(len(formatted) * val_ratio))
                if len(formatted) > val_size:
                    raw_sft_vals.append(formatted.select(range(val_size)))
                    raw_sft_trains.append(
                        formatted.select(range(val_size, len(formatted)))
                    )
                else:
                    raw_sft_vals.append(formatted)
                    raw_sft_trains.append(formatted)
            except Exception as e:
                print(f"    Error loading SFT {sft_ds.get('name', '?'):} {e}")
                import traceback

                traceback.print_exc()

        if raw_sft_trains:
            sft_train_raw = concatenate_datasets(raw_sft_trains)
            sft_train_raw = sft_train_raw.shuffle(
                seed=dataset_config.get("interleave_seed", 3407)
            )
            sft_val_raw = concatenate_datasets(raw_sft_vals)
            sft_val_raw = sft_val_raw.shuffle(seed=3407)

            n_train = min(len(sft_train_raw), train_budget)
            n_val = min(len(sft_val_raw), eval_budget)

            if n_train > 0:
                sft_tok_fp = compute_fingerprint(
                    op="tokenize_sft",
                    dataset="sft_pool",
                    formatter="mixed",
                    max_length=max_length,
                    tokenizer=getattr(tokenizer, "name_or_path", ""),
                    chat_template=chat_template_name,
                    instruction_part=instruction_part,
                    response_part=response_part,
                    n_samples=n_train,
                )
                tok_sft_train = tokenize_sft_dataset(
                    sft_train_raw.select(range(n_train)),
                    tokenizer,
                    max_length=max_length,
                    num_proc=num_proc,
                    chat_template=chat_template_name,
                    thinking_block_delimiters=thinking_block_delimiters,
                    fingerprint=sft_tok_fp,
                )
                tok_sft_train = create_assistant_only_labels(
                    tok_sft_train,
                    tokenizer,
                    instruction_part=instruction_part,
                    response_part=response_part,
                    num_proc=num_proc,
                    fingerprint=sft_tok_fp,
                )
                tokenized_train_parts.append(tok_sft_train)
                train_budget -= n_train
                print(f"    SFT train: {len(tok_sft_train):,} tokenized samples")

            if n_val > 0:
                sft_val_fp = compute_fingerprint(
                    op="tokenize_sft",
                    dataset="sft_pool_val",
                    formatter="mixed",
                    max_length=max_length,
                    tokenizer=getattr(tokenizer, "name_or_path", ""),
                    chat_template=chat_template_name,
                    instruction_part=instruction_part,
                    response_part=response_part,
                    n_samples=n_val,
                )
                tok_sft_val = tokenize_sft_dataset(
                    sft_val_raw.select(range(n_val)),
                    tokenizer,
                    max_length=max_length,
                    num_proc=num_proc,
                    chat_template=chat_template_name,
                    thinking_block_delimiters=thinking_block_delimiters,
                    fingerprint=sft_val_fp,
                )
                tok_sft_val = create_assistant_only_labels(
                    tok_sft_val,
                    tokenizer,
                    instruction_part=instruction_part,
                    response_part=response_part,
                    num_proc=num_proc,
                    fingerprint=sft_val_fp,
                )
                tokenized_val_parts.append(tok_sft_val)
                eval_budget -= n_val
                print(f"    SFT val:   {len(tok_sft_val):,} tokenized samples")

    if not tokenized_train_parts:
        raise ValueError(
            f"No datasets could be loaded for mode='{training_mode}'! "
            "Check dataset configuration."
        )

    # =====================================================================
    # Concatenate all tokenized parts
    # =====================================================================
    print(f"\n  Concatenating {len(tokenized_train_parts)} tokenized parts...")
    train_dataset = concatenate_datasets(tokenized_train_parts)
    val_dataset = concatenate_datasets(tokenized_val_parts)

    # Strip non-training columns
    keep_cols = ["input_ids", "labels"]
    train_dataset = train_dataset.select_columns(
        [c for c in keep_cols if c in train_dataset.column_names]
    )
    val_dataset = val_dataset.select_columns(
        [c for c in keep_cols if c in val_dataset.column_names]
    )

    # Packing (train only)
    if packing_enabled:
        print(f"\n  Packing training dataset (strategy={packing_strategy})...")
        print(f"  Before packing: {len(train_dataset):,} samples")
        n_before = len(train_dataset)

        map_kwargs = {
            "num_proc": num_proc,
            "desc": f"Packing train ({packing_strategy})",
        }
        train_dataset = pack_tokenized_dataset(
            train_dataset,
            max_seq_length=max_length,
            packing_strategy=packing_strategy,
            map_kwargs=map_kwargs,
        )
        n_after = len(train_dataset)
        print(
            f"  After packing:  {n_after:,} sequences "
            f"({n_before / max(n_after, 1):.1f}x compression)"
        )

    print(
        f"\n  Final train: {len(train_dataset):,} "
        f"(columns: {list(train_dataset.column_names)})"
    )
    print(
        f"  Final eval:  {len(val_dataset):,} "
        f"(columns: {list(val_dataset.column_names)})"
    )

    if verify_only:
        print("\n  Verification complete! Check the logged samples above.")
        return None, None

    return train_dataset, val_dataset

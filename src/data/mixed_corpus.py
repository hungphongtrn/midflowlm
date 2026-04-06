"""Mixed-corpus dataset loading and tokenization utilities."""

from typing import Dict, Optional, Any, List, Tuple
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer


def render_mcq_example(
    example: Dict,
    component_cfg: Dict,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    use_chat_template: bool = False,
) -> str:
    """Render a multiple-choice question example as a formatted string.

    Args:
        example: Dictionary containing question, choices, and answer
        component_cfg: Component configuration with field names
        tokenizer: Optional tokenizer for chat template rendering
        use_chat_template: Whether to format as chat messages with EOS

    Returns:
        Formatted MCQ string with question, options, and answer
    """
    question = example[component_cfg["question_field"]]
    choices = example[component_cfg["choices_field"]]
    labels = choices["label"]
    texts = choices["text"]
    answer = example[component_cfg["answer_field"]]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))

    if (
        use_chat_template
        and tokenizer is not None
        and getattr(tokenizer, "chat_template", None)
    ):
        # Format as chat with explicit instruction to return single letter
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer multiple choice questions by providing only the letter of the correct answer immediately.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nOptions:\n{options}\n\nProvide only the answer letter (e.g., A, B, C, or D):",
            },
            {"role": "assistant", "content": answer},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    # Fallback to plain format
    return f"Question: {question}\n\nOptions:\n{options}\n\nAnswer: {answer}"


def format_example_text(
    example: Dict,
    component_cfg: Dict,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> str:
    """Format an example based on its format_type.

    Args:
        example: The raw example dictionary
        component_cfg: Component configuration with format_type and field mappings
        tokenizer: Optional tokenizer for chat template rendering

    Returns:
        Formatted text string
    """
    format_type = component_cfg["format_type"]

    if format_type == "plain_text":
        return example[component_cfg["text_field"]]

    if format_type == "chat_messages":
        messages = example[component_cfg["messages_field"]]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    if format_type == "mcq_choices":
        # Check if this component should use chat template for MCQ
        use_chat_template = component_cfg.get("use_chat_template", False)
        return render_mcq_example(example, component_cfg, tokenizer, use_chat_template)

    raise ValueError(f"Unsupported format_type: {format_type}")


def load_component_dataset(
    component: Dict,
    split: str,
) -> Any:
    """Load a single component dataset from HuggingFace.

    Args:
        component: Component configuration dict
        split: Which split to load ('train' or 'val')

    Returns:
        Loaded dataset for the specified split
    """
    split_field = f"{split}_split"
    split_name = component.get(split_field, "train")

    dataset = load_dataset(
        component["dataset_name"],
        component.get("dataset_config", None),
        split=split_name,
        trust_remote_code=True,
    )

    return dataset


def build_mixture_split(
    config: Any,
    split: str,
    tokenizer: PreTrainedTokenizer,
) -> Any:
    """Build a single split of the mixed corpus by concatenating components.

    Args:
        config: Configuration object with data.mixture_components
        split: Which split to build ('train' or 'val')
        tokenizer: Tokenizer for text formatting

    Returns:
        Concatenated dataset for the split
    """
    datasets = []

    for component in config.data.mixture_components:
        ds = load_component_dataset(component, split)
        ds = ds.shuffle(seed=config.data.shuffle_seed)
        limit = component[f"{split}_samples"]
        ds = ds.select(range(min(limit, len(ds))))

        def format_with_tokenizer(ex, comp=component):
            return {"text": format_example_text(ex, comp, tokenizer)}

        ds = ds.map(format_with_tokenizer)
        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
        datasets.append(ds)

    return concatenate_datasets(datasets)


def build_mixture_split_with_stats(
    config: Any,
    split: str,
    tokenizer: PreTrainedTokenizer,
    seq_len: int = None,
) -> Tuple[Any, Dict]:
    """Build mixture split with truncation statistics.

    Args:
        config: Configuration object with data.mixture_components
        split: Which split to build ('train' or 'val')
        tokenizer: Tokenizer for text formatting
        seq_len: Sequence length for truncation detection (defaults to config.data.seq_len)

    Returns:
        tuple: (dataset, stats_dict) where stats_dict contains per-component
               truncation statistics
    """
    # Build dataset
    dataset = build_mixture_split(config, split, tokenizer)

    # Get sequence length from config if not provided
    if seq_len is None:
        seq_len = config.data.seq_len

    # Collect stats per component
    stats = {"by_component": {}}

    for component in config.data.mixture_components:
        name = component["name"]

        # Load component dataset to track truncation stats
        ds = load_component_dataset(component, split)
        ds = ds.shuffle(seed=config.data.shuffle_seed)
        limit = component[f"{split}_samples"]
        ds = ds.select(range(min(limit, len(ds))))

        # Track truncation statistics
        total_sequences = len(ds)
        truncated_sequences = 0
        original_lengths = []

        for i in range(total_sequences):
            text = ds[i].get("text", "")
            # Get original length by tokenizing without truncation
            full_tokens = tokenizer(text, truncation=False, add_special_tokens=True)
            original_len = len(full_tokens["input_ids"])
            original_lengths.append(original_len)

            # Check if truncation would occur
            if original_len > seq_len:
                truncated_sequences += 1

        # Calculate statistics
        mean_original_length = (
            sum(original_lengths) / len(original_lengths) if original_lengths else 0
        )
        truncation_rate = (
            truncated_sequences / total_sequences if total_sequences > 0 else 0.0
        )

        stats["by_component"][name] = {
            "total_sequences": total_sequences,
            "truncated_sequences": truncated_sequences,
            "truncation_rate": truncation_rate,
            "mean_original_length": mean_original_length,
        }

    return dataset, stats


def get_truncation_stats(stats: Dict) -> Dict:
    """Get formatted truncation statistics summary.

    Args:
        stats: Stats dict from build_mixture_split_with_stats

    Returns:
        Dict with formatted summary including overall and per-component stats
    """
    summary = {
        "by_component": {},
        "overall_truncation_rate": 0.0,
        "per_component": {},
    }

    if "by_component" not in stats:
        return summary

    total_sequences = 0
    total_truncated = 0

    for name, component_stats in stats["by_component"].items():
        # Copy per-component stats
        summary["by_component"][name] = component_stats.copy()

        # Add to summary format
        summary["per_component"][name] = {
            "truncation_rate": component_stats.get("truncation_rate", 0.0),
            "mean_original_length": component_stats.get("mean_original_length", 0),
        }

        # Accumulate totals for overall rate
        total_sequences += component_stats.get("total_sequences", 0)
        total_truncated += component_stats.get("truncated_sequences", 0)

    # Compute overall truncation rate
    if total_sequences > 0:
        summary["overall_truncation_rate"] = total_truncated / total_sequences

    return summary


def tokenize_function(
    examples: Dict[str, list],
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
) -> Dict[str, list]:
    """Tokenize text examples to fixed sequence length.

    Args:
        examples: Dictionary with text field containing list of strings
        tokenizer: HuggingFace tokenizer
        seq_len: Target sequence length

    Returns:
        Dictionary with input_ids and attention_mask
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors=None,
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def get_mixed_corpus_dataloaders(
    config: Any,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 8,
) -> Dict[str, DataLoader]:
    """Create deterministic mixed-corpus dataloaders for train/val.

    Args:
        config: Configuration object with data section
        tokenizer: Pre-loaded tokenizer (if None, will load from config)
        batch_size: Batch size for dataloaders

    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    data_config = config.data
    model_config = config.model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name,
            revision=model_config.revision,
            trust_remote_code=True,
        )
        # Note: Qwen3.5 already has proper PAD token (<|endoftext|>) distinct from EOS (<|im_end|>)
        # Only set pad_token if model truly lacks one
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_mixture_split(config, "train", tokenizer)
    val_dataset = build_mixture_split(config, "val", tokenizer)

    tokenize_fn = lambda examples: tokenize_function(
        examples,
        tokenizer=tokenizer,
        seq_len=data_config.seq_len,
    )

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloaders = {}

    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
        and data_config.num_workers > 0,
        prefetch_factor=getattr(data_config, "prefetch_factor", 2)
        if data_config.num_workers > 0
        else None,
    )

    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
        and data_config.num_workers > 0,
        prefetch_factor=getattr(data_config, "prefetch_factor", 2)
        if data_config.num_workers > 0
        else None,
    )

    return dataloaders


def get_sample_batch(
    dataloaders: Dict[str, DataLoader],
    split: str = "train",
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Get a single sample batch from a dataloader.

    Args:
        dataloaders: Dictionary of dataloaders
        split: Which split to sample from
        device: Device to move tensors to

    Returns:
        Dictionary with input_ids and attention_mask tensors
    """
    if split not in dataloaders:
        raise ValueError(
            f"Split '{split}' not found in dataloaders. Available: {list(dataloaders.keys())}"
        )

    dataloader = dataloaders[split]
    batch = next(iter(dataloader))

    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    return batch

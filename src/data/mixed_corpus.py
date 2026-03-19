"""Mixed-corpus dataset loading and tokenization utilities."""

from typing import Dict, Optional, Any, List
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer


def render_mcq_example(example: Dict, component_cfg: Dict) -> str:
    """Render a multiple-choice question example as a formatted string.

    Args:
        example: Dictionary containing question, choices, and answer
        component_cfg: Component configuration with field names

    Returns:
        Formatted MCQ string with question, options, and answer
    """
    question = example[component_cfg["question_field"]]
    choices = example[component_cfg["choices_field"]]
    labels = choices["label"]
    texts = choices["text"]
    answer = example[component_cfg["answer_field"]]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
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
        return render_mcq_example(example, component_cfg)

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
    )

    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
        and data_config.num_workers > 0,
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

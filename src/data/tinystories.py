"""TinyStories dataset loading and tokenization utilities."""

from typing import Dict, Optional, Any
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer


def tokenize_function(
    examples: Dict[str, list],
    tokenizer: PreTrainedTokenizer,
    text_field: str,
    seq_len: int,
) -> Dict[str, list]:
    """Tokenize text examples to fixed sequence length.

    Args:
        examples: Dictionary with text field containing list of strings
        tokenizer: HuggingFace tokenizer
        text_field: Name of the field containing text
        seq_len: Target sequence length

    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Tokenize with padding and truncation to fixed length
    tokenized = tokenizer(
        examples[text_field],
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors=None,  # Return lists, not tensors (datasets handles batching)
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def get_tinystories_dataloaders(
    config: Any,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 8,
) -> Dict[str, DataLoader]:
    """Create deterministic TinyStories dataloaders for train/val/test.

    Args:
        config: Configuration object with data section
        tokenizer: Pre-loaded tokenizer (if None, will load from config)
        batch_size: Batch size for dataloaders

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    data_config = config.data
    model_config = config.model

    # Load tokenizer if not provided
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

    # Load dataset
    dataset = load_dataset(
        data_config.dataset_name,
        revision=data_config.dataset_revision,
        trust_remote_code=True,
    )

    # Get splits
    train_dataset = dataset.get("train")
    val_dataset = dataset.get("validation")
    test_dataset = dataset.get("test")

    # Handle case where validation doesn't exist (use split from train)
    if val_dataset is None and train_dataset is not None:
        # Split train into train/val if no validation set exists
        split_dataset = train_dataset.train_test_split(
            test_size=data_config.val_samples,
            seed=data_config.shuffle_seed,
            shuffle=True,
        )
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    # Shuffle and limit samples with deterministic seed
    shuffle_seed = data_config.shuffle_seed

    if train_dataset is not None and data_config.train_samples > 0:
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)
        train_indices = list(range(min(data_config.train_samples, len(train_dataset))))
        train_dataset = train_dataset.select(train_indices)

    if val_dataset is not None and data_config.val_samples > 0:
        val_dataset = val_dataset.shuffle(seed=shuffle_seed)
        val_indices = list(range(min(data_config.val_samples, len(val_dataset))))
        val_dataset = val_dataset.select(val_indices)

    if test_dataset is not None and data_config.test_samples > 0:
        test_dataset = test_dataset.shuffle(seed=shuffle_seed)
        test_indices = list(range(min(data_config.test_samples, len(test_dataset))))
        test_dataset = test_dataset.select(test_indices)

    # Tokenize datasets
    tokenize_fn = lambda examples: tokenize_function(
        examples,
        tokenizer=tokenizer,
        text_field=data_config.text_field,
        seq_len=data_config.seq_len,
    )

    if train_dataset is not None:
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if val_dataset is not None:
        val_dataset = val_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset",
        )
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if test_dataset is not None:
        test_dataset = test_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset",
        )
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create dataloaders
    dataloaders = {}

    if train_dataset is not None:
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers
            and data_config.num_workers > 0,
        )

    if val_dataset is not None:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers
            and data_config.num_workers > 0,
        )

    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
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

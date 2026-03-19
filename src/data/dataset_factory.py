"""Dataset factory for dispatching between different dataset loaders."""

import copy
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.data.mixed_corpus import get_mixed_corpus_dataloaders
from src.data.tinystories import get_tinystories_dataloaders


def normalize_data_config(data_config: Any) -> Any:
    """Normalize data config to handle both legacy and mixture schemas.

    This ensures that nested mixture_components survive config parsing
    when configs are loaded as dicts and converted to namespace objects.

    Args:
        data_config: The data configuration object

    Returns:
        Normalized data config with mixture_components preserved
    """
    if hasattr(data_config, "mixture_components"):
        return data_config

    normalized = copy.copy(data_config)

    if hasattr(data_config, "__dict__"):
        raw_components = data_config.__dict__.get("mixture_components")
        if raw_components is not None:
            normalized.mixture_components = raw_components

    return normalized


def get_experiment_dataloaders(
    config: Any,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 8,
) -> Dict[str, DataLoader]:
    """Factory function that dispatches to the appropriate dataset loader.

    This function inspects the config's data.loader field to determine
    which dataset loader to use. It supports both legacy TinyStories
    configs and new mixture configs.

    Args:
        config: Configuration object with data section
        tokenizer: Pre-loaded tokenizer (if None, will load from config)
        batch_size: Batch size for dataloaders

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders

    Raises:
        ValueError: If data.loader is not recognized
    """
    data_config = normalize_data_config(config.data)
    loader_name = getattr(data_config, "loader", "tinystories")

    if loader_name == "tinystories":
        config.data = data_config
        return get_tinystories_dataloaders(
            config=config,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    if loader_name == "mixture":
        config.data = data_config
        return get_mixed_corpus_dataloaders(
            config=config,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    raise ValueError(
        f"Unsupported data.loader: {loader_name}. Expected 'tinystories' or 'mixture'."
    )

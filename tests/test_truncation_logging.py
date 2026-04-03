import pytest
from unittest.mock import MagicMock, patch
from src.data.mixed_corpus import build_mixture_split_with_stats, get_truncation_stats


def test_build_mixture_split_with_stats_returns_dataset_and_stats():
    """Test that truncation statistics are collected during dataset building."""

    config = MagicMock()
    config.data.seq_len = 1024
    config.data.mixture_components = [
        {
            "name": "test_dataset",
            "dataset_name": "test/dataset",
            "train_split": "train",
            "val_split": "validation",
            "train_samples": 100,
            "val_samples": 10,
            "format_type": "plain_text",
            "text_field": "text",
        }
    ]
    config.data.shuffle_seed = 42

    tokenizer = MagicMock()

    # Mock tokenizer behavior for truncation detection
    def mock_tokenize(text, truncation=False, add_special_tokens=True):
        # Return different lengths based on text length to simulate truncation
        if len(text) > 50:
            return {"input_ids": list(range(1500))}  # Would be truncated
        else:
            return {"input_ids": list(range(50))}  # Would not be truncated

    tokenizer.side_effect = mock_tokenize

    # Mock dataset with actual examples
    mock_examples = [{"text": "x" * 100}] * 10  # 10 examples that would be truncated

    with (
        patch("src.data.mixed_corpus.load_dataset") as mock_load,
        patch("src.data.mixed_corpus.concatenate_datasets") as mock_concat,
    ):
        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda i: mock_examples[i])
        mock_load.return_value = mock_dataset

        # Return a mock dataset from concatenate_datasets
        mock_concat.return_value = MagicMock()

        dataset, stats = build_mixture_split_with_stats(
            config=config,
            split="train",
            tokenizer=tokenizer,
        )

    assert dataset is not None
    assert isinstance(stats, dict)
    assert "by_component" in stats
    assert "test_dataset" in stats["by_component"]
    component_stats = stats["by_component"]["test_dataset"]
    assert "truncation_rate" in component_stats
    assert "mean_original_length" in component_stats


def test_get_truncation_stats_returns_per_component_stats():
    """Test get_truncation_stats produces summary metrics."""
    stats = {
        "by_component": {
            "fineweb": {"truncation_rate": 0.05, "mean_original_length": 1500},
            "ultrachat": {"truncation_rate": 0.12, "mean_original_length": 1800},
        }
    }

    summary = get_truncation_stats(stats)

    assert "overall_truncation_rate" in summary
    assert "per_component" in summary
    assert summary["by_component"]["fineweb"]["truncation_rate"] == 0.05

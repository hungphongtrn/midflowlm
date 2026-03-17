"""Tests for TinyStories dataset loading and tokenization."""

import pytest
import torch
from unittest.mock import MagicMock, patch


def test_deterministic_shuffled_splits():
    """Test that dataset splits are deterministic with the same shuffle seed."""
    from src.data.tinystories import get_tinystories_dataloaders
    
    config = MagicMock()
    config.data.dataset_name = "roneneldan/TinyStories"
    config.data.dataset_revision = None
    config.data.text_field = "text"
    config.data.seq_len = 128
    config.data.train_samples = 100
    config.data.val_samples = 20
    config.data.test_samples = 20
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False
    config.data.shuffle_seed = 1337
    config.model.name = "Qwen/Qwen3.5-0.8B"
    config.model.revision = None
    
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    
    with patch("src.data.tinystories.AutoTokenizer") as mock_tokenizer_cls, \
         patch("src.data.tinystories.load_dataset") as mock_load_dataset:
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.__len__ = MagicMock(return_value=10000)
        mock_dataset.__getitem__ = MagicMock(return_value={"text": "Once upon a time..."})
        
        mock_load_dataset.return_value = {
            "train": mock_dataset,
            "validation": mock_dataset,
        }
        
        # First call with seed 1337
        loaders1 = get_tinystories_dataloaders(config, tokenizer=mock_tokenizer)
        
        # Verify shuffle was called with the correct seed
        mock_dataset.shuffle.assert_called()
        call_kwargs = mock_dataset.shuffle.call_args[1]
        assert call_kwargs.get("seed") == 1337 or 1337 in str(call_kwargs)


def test_tokenization_fixed_seq_len():
    """Test that tokenization produces fixed sequence length outputs."""
    from src.data.tinystories import tokenize_function
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5] + [0] * 123],  # 128 tokens
        "attention_mask": [[1, 1, 1, 1, 1] + [0] * 123],
    }
    
    examples = {"text": ["Once upon a time..."]}
    result = tokenize_function(
        examples, 
        tokenizer=mock_tokenizer, 
        text_field="text",
        seq_len=128
    )
    
    assert "input_ids" in result
    assert "attention_mask" in result
    mock_tokenizer.assert_called_once()
    call_kwargs = mock_tokenizer.call_args[1]
    assert call_kwargs.get("max_length") == 128 or call_kwargs.get("truncation") == True


def test_expected_keys_in_batch():
    """Test that batches contain expected keys: input_ids, attention_mask."""
    from src.data.tinystories import get_tinystories_dataloaders
    
    config = MagicMock()
    config.data.dataset_name = "roneneldan/TinyStories"
    config.data.dataset_revision = None
    config.data.text_field = "text"
    config.data.seq_len = 128
    config.data.train_samples = 10
    config.data.val_samples = 5
    config.data.test_samples = 5
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False
    config.data.shuffle_seed = 1337
    config.model.name = "Qwen/Qwen3.5-0.8B"
    config.model.revision = None
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    
    # Create mock tokenized dataset that returns proper tensors
    mock_item = {
        "input_ids": torch.randint(0, 1000, (128,)),
        "attention_mask": torch.ones(128, dtype=torch.int64),
    }
    
    mock_dataset = MagicMock()
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.select.return_value = mock_dataset
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.__len__ = MagicMock(return_value=100)
    mock_dataset.__getitem__ = MagicMock(return_value=mock_item)
    mock_dataset.set_format = MagicMock()
    
    with patch("src.data.tinystories.AutoTokenizer") as mock_tokenizer_cls, \
         patch("src.data.tinystories.load_dataset") as mock_load_dataset:
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_load_dataset.return_value = {
            "train": mock_dataset,
            "validation": mock_dataset,
        }
        
        loaders = get_tinystories_dataloaders(config, tokenizer=mock_tokenizer)
        
        # Check that set_format was called with the expected keys
        mock_dataset.set_format.assert_called()
        call_kwargs = mock_dataset.set_format.call_args[1]
        assert "input_ids" in call_kwargs.get("columns", [])
        assert "attention_mask" in call_kwargs.get("columns", [])


def test_sample_count_limits():
    """Test that sample limits are honored for train/val/test splits."""
    from src.data.tinystories import get_tinystories_dataloaders
    
    config = MagicMock()
    config.data.dataset_name = "roneneldan/TinyStories"
    config.data.dataset_revision = None
    config.data.text_field = "text"
    config.data.seq_len = 128
    config.data.train_samples = 500
    config.data.val_samples = 100
    config.data.test_samples = 50
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False
    config.data.shuffle_seed = 1337
    config.model.name = "Qwen/Qwen3.5-0.8B"
    config.model.revision = None
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    
    # Track shuffle and select calls
    shuffle_calls = []
    select_calls = []
    
    def create_mock_dataset(size=10000):
        """Create a properly chained mock dataset."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=size)
        ds.set_format = MagicMock()
        
        def capture_shuffle(**kwargs):
            shuffle_calls.append(kwargs)
            return ds
        
        def capture_select(indices):
            select_calls.append(indices)
            # Return new mock with selected size
            return create_mock_dataset(len(indices))
        
        ds.shuffle = capture_shuffle
        ds.select = capture_select
        ds.map = MagicMock(return_value=ds)
        return ds
    
    mock_dataset = create_mock_dataset(10000)
    
    with patch("src.data.tinystories.AutoTokenizer") as mock_tokenizer_cls, \
         patch("src.data.tinystories.load_dataset") as mock_load_dataset:
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_load_dataset.return_value = {
            "train": mock_dataset,
            "validation": mock_dataset,
        }
        
        loaders = get_tinystories_dataloaders(config, tokenizer=mock_tokenizer)
        
        # Check that shuffle was called with seed
        assert len(shuffle_calls) >= 2
        for call in shuffle_calls:
            assert call.get("seed") == 1337
        
        # Check that select was called with limited indices
        assert len(select_calls) >= 2
        
        # Verify train selection has train_samples limit
        train_indices = select_calls[0]
        assert len(train_indices) == 500  # Should match train_samples exactly


def test_dataloader_batch_structure():
    """Test that dataloader returns properly structured batches."""
    from src.data.tinystories import get_tinystories_dataloaders
    
    config = MagicMock()
    config.data.dataset_name = "roneneldan/TinyStories"
    config.data.dataset_revision = None
    config.data.text_field = "text"
    config.data.seq_len = 128
    config.data.train_samples = 10
    config.data.val_samples = 5
    config.data.test_samples = 5
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False
    config.data.shuffle_seed = 1337
    config.model.name = "Qwen/Qwen3.5-0.8B"
    config.model.revision = None
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    
    # Create a mock dataset that returns tensor data
    mock_item = {
        "input_ids": torch.randint(0, 1000, (128,)),
        "attention_mask": torch.ones(128, dtype=torch.int64),
    }
    
    mock_dataset = MagicMock()
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.select.return_value = mock_dataset
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.__len__ = MagicMock(return_value=10)
    mock_dataset.__getitem__ = MagicMock(return_value=mock_item)
    mock_dataset.set_format = MagicMock()
    
    with patch("src.data.tinystories.AutoTokenizer") as mock_tokenizer_cls, \
         patch("src.data.tinystories.load_dataset") as mock_load_dataset, \
         patch("src.data.tinystories.DataLoader") as mock_dataloader_cls:
        
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_load_dataset.return_value = {
            "train": mock_dataset,
            "validation": mock_dataset,
        }
        
        # Mock dataloader to return a simple iterator
        mock_dataloader = MagicMock()
        mock_dataloader_cls.return_value = mock_dataloader
        
        loaders = get_tinystories_dataloaders(config, tokenizer=mock_tokenizer)
        
        # Verify DataLoader was called
        assert mock_dataloader_cls.called
        
        # Verify loaders dict has expected keys
        assert "train" in loaders
        assert "val" in loaders

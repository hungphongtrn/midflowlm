"""Data loading utilities for training from teacher cache.

This module provides:
- Dataset wrapper for cached teacher data
- Deterministic dataloader creation
- Variable T sampling support
"""

import json
import random
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

from src.data.teacher_cache import load_metadata, load_shard


class CacheDataset(Dataset):
    """Dataset that loads samples from teacher cache.
    
    This dataset loads cached teacher outputs including:
    - input_ids: Token IDs
    - attention_mask: Attention mask
    - h_start: Hidden state before replacement span
    - trajectory_targets: Hidden states within the span
    - h_target: Hidden state after replacement span
    - teacher_logits: Optional teacher logits
    
    Args:
        cache_dir: Directory containing cache files
        split: Which split to use (train/val/test)
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        split: str = "train",
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load metadata
        self.metadata = load_metadata(self.cache_dir)
        self.num_samples = self.metadata.num_samples
        self.seq_len = self.metadata.seq_len
        self.span_depth = self.metadata.span_depth
        
        # Find all shards
        self.shards = self._find_shards()
        
        if not self.shards:
            raise ValueError(f"No cache shards found in {cache_dir}")
        
        # Build index mapping: sample_idx -> (shard_idx, local_idx)
        self.sample_map = self._build_sample_map()
    
    def _find_shards(self) -> List[Path]:
        """Find all cache shard files."""
        # Look for .safetensors or .pt files
        safetensor_shards = sorted(self.cache_dir.glob("shard_*_of_*.safetensors"))
        pt_shards = sorted(self.cache_dir.glob("shard_*_of_*.pt"))
        
        if safetensor_shards:
            return safetensor_shards
        return pt_shards
    
    def _build_sample_map(self) -> List[tuple]:
        """Build mapping from global sample index to (shard_idx, local_idx)."""
        sample_map = []
        
        for shard_idx, shard_path in enumerate(self.shards):
            # Load shard to determine number of samples
            # We load the first tensor to get batch size
            shard_data = torch.load(shard_path, map_location="cpu")
            if "input_ids" in shard_data:
                num_samples_in_shard = shard_data["input_ids"].shape[0]
            elif "h_start" in shard_data:
                num_samples_in_shard = shard_data["h_start"].shape[0]
            else:
                # Fallback: assume single sample per shard
                num_samples_in_shard = 1
            
            for local_idx in range(num_samples_in_shard):
                sample_map.append((shard_idx, local_idx))
        
        return sample_map
    
    def __len__(self) -> int:
        return len(self.sample_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the cache.
        
        Args:
            idx: Global sample index
            
        Returns:
            Dictionary with sample data
        """
        shard_idx, local_idx = self.sample_map[idx]
        
        # Load shard (cache this for efficiency if needed)
        shard_data = torch.load(self.shards[shard_idx], map_location="cpu")
        
        # Extract single sample from batch
        sample = {}
        
        if "input_ids" in shard_data:
            sample["input_ids"] = shard_data["input_ids"][local_idx]
        
        if "attention_mask" in shard_data:
            sample["attention_mask"] = shard_data["attention_mask"][local_idx]
        
        if "h_start" in shard_data:
            sample["h_start"] = shard_data["h_start"][local_idx]
        
        if "h_target" in shard_data:
            sample["h_target"] = shard_data["h_target"][local_idx]
        
        # Reconstruct trajectory_targets list into tensor [seq, depth, hidden]
        if "num_trajectory_targets" in shard_data:
            num_targets = int(shard_data["num_trajectory_targets"].item())
            trajectory_list = []
            for i in range(num_targets):
                key = f"trajectory_target_{i}"
                if key in shard_data:
                    trajectory_list.append(shard_data[key][local_idx])
            if trajectory_list:
                # Stack to [depth, seq, hidden] then transpose to [seq, depth, hidden]
                sample["trajectory_targets"] = torch.stack(trajectory_list, dim=0).transpose(0, 1)
        
        if "teacher_logits" in shard_data:
            sample["teacher_logits"] = shard_data["teacher_logits"][local_idx]
        
        return sample


class DeterministicSampler(Sampler):
    """Deterministic sampler that ensures reproducibility.
    
    Args:
        data_source: Dataset to sample from
        num_samples: Number of samples to draw
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the indices
    """
    
    def __init__(
        self,
        data_source,
        num_samples: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.data_source = data_source
        self.num_samples = num_samples or len(data_source)
        self.seed = seed
        self.shuffle = shuffle
        
        # Generate deterministic indices
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        if shuffle:
            self.indices = torch.randperm(len(data_source), generator=generator).tolist()
        else:
            self.indices = list(range(len(data_source)))
        
        # Limit to num_samples
        self.indices = self.indices[:self.num_samples]
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


def create_cache_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    split: str = "train",
) -> DataLoader:
    """Create a deterministic dataloader from teacher cache.
    
    Args:
        cache_dir: Directory containing cache files
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        split: Which split to use
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = CacheDataset(cache_dir=cache_dir, split=split)
    
    # Create deterministic sampler
    sampler = DeterministicSampler(
        data_source=dataset,
        seed=seed,
        shuffle=shuffle,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_cache_batch,
    )
    
    return dataloader


def _collate_cache_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for cache batches.
    
    Stacks individual samples into batched tensors.
    """
    collated = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        if key == "trajectory_targets":
            # Stack trajectory targets: list of [seq, depth, hidden] -> [batch, seq, depth, hidden]
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = torch.stack(values, dim=0)
    
    return collated


def get_train_val_dataloaders(
    cache_dir: Union[str, Path],
    config: Dict,
    train_seed: int = 42,
    val_seed: int = 1337,
) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders from cache.
    
    Args:
        cache_dir: Directory containing cache files
        config: Configuration dictionary with data settings
        train_seed: Seed for train dataloader shuffle
        val_seed: Seed for validation dataloader (no shuffle, but for consistency)
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    data_config = config.get("data", {})
    batch_size = data_config.get("batch_size", 8)
    num_workers = data_config.get("num_workers", 0)
    pin_memory = data_config.get("pin_memory", False)
    
    dataloaders = {}
    
    # Train dataloader with shuffling
    dataloaders["train"] = create_cache_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        shuffle=True,
        seed=train_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        split="train",
    )
    
    # Val dataloader without shuffling
    dataloaders["val"] = create_cache_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        shuffle=False,
        seed=val_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        split="val",
    )
    
    return dataloaders


def get_cache_info(cache_dir: Union[str, Path]) -> Dict:
    """Get information about the cache.
    
    Args:
        cache_dir: Directory containing cache files
        
    Returns:
        Dictionary with cache metadata and statistics
    """
    metadata = load_metadata(cache_dir)
    
    # Count shards
    safetensor_shards = list(Path(cache_dir).glob("shard_*_of_*.safetensors"))
    pt_shards = list(Path(cache_dir).glob("shard_*_of_*.pt"))
    num_shards = len(safetensor_shards) or len(pt_shards)
    
    return {
        "model_name": metadata.model_name,
        "model_revision": metadata.model_revision,
        "start_layer": metadata.start_layer,
        "end_layer": metadata.end_layer,
        "span_depth": metadata.span_depth,
        "seq_len": metadata.seq_len,
        "store_logits": metadata.store_logits,
        "num_samples": metadata.num_samples,
        "num_shards": num_shards,
    }

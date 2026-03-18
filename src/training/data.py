"""Data loading utilities for training from teacher cache.

This module provides:
- Dataset wrapper for cached teacher data
- Deterministic dataloader creation
- Variable T sampling support
"""

import re
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
from torch.utils.data import Dataset, DataLoader, Sampler

from src.data.teacher_cache import load_metadata, resolve_split_cache_dir


class CacheDataset(Dataset):
    """Dataset that loads samples from teacher cache.

    This dataset loads cached teacher outputs including:
    - input_ids: Token IDs
    - attention_mask: Attention mask
    - h_start: Hidden state before replacement span
    - velocity_target: v_target = h_target - h_start for continuous-time training
    - h_target: Hidden state after replacement span (optional, for verification)
    - teacher_logits: Optional teacher logits

    Training State Reconstruction:
        The cache stores velocity targets that enable reconstruction of training states
        at any timestep t in [0, 1] using the straight-line interpolation rule:

            h_t = h_start + t[:, None, None] * velocity_target

        This is the continuous-time formulation where:
        - At t=0: h_0 = h_start (initial state)
        - At t=1: h_1 = h_start + velocity_target = h_target (final state)
        - At intermediate t: interpolated state along the straight-line trajectory

    Args:
        cache_dir: Directory containing cache files
        split: Which split to use (train/val/test)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        split: str = "train",
    ):
        self.cache_root = Path(cache_dir)
        self.split = split

        # Resolve split-specific cache directory
        self.cache_dir = resolve_split_cache_dir(self.cache_root, split)

        # Fallback to root if split subdirectory doesn't exist (flat cache structure)
        if not self.cache_dir.exists():
            self.cache_dir = self.cache_root

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
        """Find all cache shard files and sort by actual shard index."""
        # Look for .safetensors or .pt files
        safetensor_shards = list(self.cache_dir.glob("shard_*_of_*.safetensors"))
        pt_shards = list(self.cache_dir.glob("shard_*_of_*.pt"))

        all_shards = safetensor_shards + pt_shards

        if not all_shards:
            return []

        # Parse shard index from each filename and sort by it
        # Pattern: shard_{idx}_of_{total}.ext
        shard_with_index = []
        for shard_path in all_shards:
            match = re.search(r"shard_(\d+)_of_\d+\.", shard_path.name)
            if match:
                shard_idx = int(match.group(1))
                shard_with_index.append((shard_idx, shard_path))

        # Sort by shard index
        shard_with_index.sort(key=lambda x: x[0])

        # Extract sorted paths
        shards = [path for _, path in shard_with_index]

        return shards

    def _build_sample_map(self) -> List[tuple]:
        """Build mapping from global sample index to (shard_idx, local_idx)."""
        sample_map = []
        samples_per_shard = self._get_samples_per_shard()

        for shard_idx, num_samples_in_shard in enumerate(samples_per_shard):
            for local_idx in range(num_samples_in_shard):
                sample_map.append((shard_idx, local_idx))

        return sample_map

    def _get_samples_per_shard(self) -> List[int]:
        """Infer shard sample counts from metadata instead of shard payloads."""
        num_shards = len(self.shards)
        metadata_counts = getattr(self.metadata, "samples_per_shard", None)

        if metadata_counts is not None:
            if len(metadata_counts) != num_shards:
                raise ValueError(
                    "Metadata samples_per_shard length does not match shard count"
                )
            if sum(metadata_counts) != self.num_samples:
                raise ValueError(
                    "Metadata samples_per_shard total does not match num_samples"
                )
            return metadata_counts

        # Distribute samples evenly across shards
        base_samples, remainder = divmod(self.num_samples, num_shards)
        return [
            base_samples + (1 if shard_idx < remainder else 0)
            for shard_idx in range(num_shards)
        ]

    def _load_shard_by_path(
        self, shard_path: Path, device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load a shard directly from its file path.

        Loads cached data including velocity_target for continuous-time training.
        For backward compatibility, also reconstructs trajectory_targets if present
        in old format.

        Args:
            shard_path: Path to the shard file
            device: Device to load tensors to

        Returns:
            Dictionary with loaded tensors including velocity_target
        """
        from src.data.teacher_cache import HAS_SAFETENSORS

        # Try safetensors first, then .pt fallback
        if shard_path.suffix == ".safetensors" and HAS_SAFETENSORS:
            from safetensors.torch import load_file

            loaded = cast(Dict[str, Any], load_file(str(shard_path)))
        else:
            loaded = cast(Dict[str, Any], torch.load(shard_path, map_location=device))

        # For backward compatibility: reconstruct trajectory_targets if present in old format
        if "num_trajectory_targets" in loaded:
            num_targets = int(loaded["num_trajectory_targets"].item())
            trajectory_targets = []
            for i in range(num_targets):
                key = f"trajectory_target_{i}"
                if key in loaded:
                    trajectory_targets.append(loaded[key])
            loaded["trajectory_targets"] = trajectory_targets

        return loaded

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

        # Load shard by filename directly (handles mixed naming patterns)
        shard_path = self.shards[shard_idx]
        shard_data = self._load_shard_by_path(shard_path, device="cpu")

        # Extract single sample from batch (or use full shard if no batch dim)
        sample = {}

        if "input_ids" in shard_data:
            input_ids = shard_data["input_ids"]
            if input_ids.dim() == 1:
                # No batch dimension, use full tensor
                sample["input_ids"] = input_ids
            else:
                # Has batch dimension, index it
                sample["input_ids"] = input_ids[local_idx]

        if "attention_mask" in shard_data:
            attention_mask = shard_data["attention_mask"]
            if attention_mask.dim() == 1:
                sample["attention_mask"] = attention_mask
            else:
                sample["attention_mask"] = attention_mask[local_idx]

        if "h_start" in shard_data:
            h_start = shard_data["h_start"]
            if h_start.dim() == 2:
                # [seq_len, hidden]: no batch dim
                sample["h_start"] = h_start
            else:
                # [batch, seq_len, hidden]: index it
                sample["h_start"] = h_start[local_idx]

        if "h_target" in shard_data:
            h_target = shard_data["h_target"]
            if h_target.dim() == 2:
                sample["h_target"] = h_target
            else:
                sample["h_target"] = h_target[local_idx]

        # velocity_target for continuous-time training
        if "velocity_target" in shard_data:
            velocity_target = shard_data["velocity_target"]
            if velocity_target.dim() == 2:
                # [seq_len, hidden]: no batch dim
                sample["velocity_target"] = velocity_target
            else:
                # [batch, seq_len, hidden]: index it
                sample["velocity_target"] = velocity_target[local_idx]

        # teacher_logits is optional - only include if present
        if "teacher_logits" in shard_data:
            teacher_logits = shard_data["teacher_logits"]
            if teacher_logits.dim() == 2:
                # [seq_len, vocab]: no batch dim
                sample["teacher_logits"] = teacher_logits
            else:
                # [batch, seq_len, vocab]: index it
                sample["teacher_logits"] = teacher_logits[local_idx]

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
            self.indices = torch.randperm(
                len(data_source), generator=generator
            ).tolist()
        else:
            self.indices = list(range(len(data_source)))

        # Limit to num_samples
        self.indices = self.indices[: self.num_samples]

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


def _collate_cache_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function for cache batches.

    Stacks individual samples into batched tensors.
    All tensors including velocity_target are stacked directly.
    """
    collated = {}

    # Get all keys from first sample
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
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

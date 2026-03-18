"""Teacher cache generation with velocity targets for continuous-time training.

This module provides utilities for caching teacher model outputs including:
- h_start: Hidden state before the replacement span
- velocity_target: v_target = h_target - h_start for continuous-time training
- h_target: Hidden state after the replacement span (optional, for verification)
- teacher_logits: Optional final logits

The cache is stored in safetensors format for efficient loading during training.

Training State Reconstruction:
    The cache stores velocity targets that enable reconstruction of training states
    at any timestep t in [0, 1] using the straight-line interpolation rule:

        h_t = h_start + t[:, None, None] * velocity_target

    This is the continuous-time formulation where:
    - At t=0: h_0 = h_start (initial state)
    - At t=1: h_1 = h_start + velocity_target = h_target (final state)
    - At intermediate t: interpolated state along the straight-line trajectory
"""

import json
import torch
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

# Optional safetensors import with fallback
try:
    from safetensors.torch import save_file, load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logging.warning("safetensors not available, falling back to torch.save")

from src.model.qwen_parity import QwenInspector


logger = logging.getLogger(__name__)


def resolve_split_cache_dir(cache_dir: Union[str, Path], split: str) -> Path:
    """Resolve cache directory for a specific split.

    Args:
        cache_dir: Root cache directory
        split: Dataset split (train/val/test)

    Returns:
        Path to split-specific cache directory
    """
    cache_root = Path(cache_dir)
    return cache_root / split


@dataclass
class CacheMetadata:
    """Metadata for teacher cache.

    Attributes:
        model_name: Name of the teacher model
        model_revision: Revision/commit hash of the model
        start_layer: First layer of replacement span (inclusive)
        end_layer: Last layer of replacement span (inclusive)
        span_depth: Number of layers in the span (end_layer - start_layer + 1)
        seq_len: Sequence length for cached samples
        store_logits: Whether logits are stored in the cache (default: False)
        num_samples: Total number of samples cached
        target_type: Type of target stored ("velocity" or "trajectory")
        time_domain: Time domain for continuous training [start, end]
        training_state_rule: Documentation of h_t reconstruction formula
    """

    model_name: str
    model_revision: Optional[str]
    start_layer: int
    end_layer: int
    span_depth: int
    seq_len: int
    store_logits: bool = False
    num_samples: int = 0
    samples_per_shard: Optional[List[int]] = None
    target_type: str = "velocity"
    time_domain: List[float] = field(default_factory=lambda: [0.0, 1.0])
    training_state_rule: str = "h_t = h_start + t[:, None, None] * velocity_target"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        """Create CacheMetadata from dictionary."""
        return cls(**data)


class TeacherCacheWriter:
    """Writer for teacher cache with velocity targets.

    This class handles:
    - Creating cache directory structure
    - Writing metadata including target_type and training state rule
    - Writing sample shards in safetensors format with velocity targets
    - Supporting resumable/idempotent operations

    The cache stores velocity_target = h_target - h_start for continuous-time training.
    Training states are reconstructed using:
        h_t = h_start + t[:, None, None] * velocity_target
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        model_name: str,
        start_layer: int = 8,
        end_layer: int = 11,
        seq_len: int = 128,
        store_logits: bool = False,
        model_revision: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Initialize TeacherCacheWriter.

        Args:
            cache_dir: Directory to store cache files
            model_name: Name of the teacher model
            start_layer: First layer of replacement span (inclusive)
            end_layer: Last layer of replacement span (inclusive)
            seq_len: Sequence length for cached samples
            store_logits: Whether to store teacher logits
            model_revision: Model revision/commit hash
            overwrite: Whether to overwrite existing cache
        """
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.span_depth = end_layer - start_layer + 1
        self.seq_len = seq_len
        self.store_logits = store_logits
        self.model_revision = model_revision
        self.overwrite = overwrite

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TeacherCacheWriter at {self.cache_dir}")
        logger.info(f"  Model: {model_name}")
        logger.info(
            f"  Span: layers {start_layer}-{end_layer} (depth={self.span_depth})"
        )
        logger.info(f"  Store logits: {store_logits}")

    def write_metadata(self, num_samples: int) -> None:
        """Write metadata to cache directory.

        Args:
            num_samples: Total number of samples in the cache
        """
        metadata = CacheMetadata(
            model_name=self.model_name,
            model_revision=self.model_revision,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            span_depth=self.span_depth,
            seq_len=self.seq_len,
            store_logits=self.store_logits,
            num_samples=num_samples,
            target_type="velocity",
            time_domain=[0.0, 1.0],
            training_state_rule="h_t = h_start + t[:, None, None] * velocity_target",
        )

        metadata_path = self.cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Wrote metadata to {metadata_path}")
        logger.info(f"  Target type: velocity")
        logger.info(f"  Training state rule: h_t = h_start + t * velocity_target")

    def _get_shard_path(self, shard_idx: int, num_shards: int) -> Path:
        """Get the path for a shard file.

        Args:
            shard_idx: Index of the shard
            num_shards: Total number of shards

        Returns:
            Path to the shard file
        """
        return self.cache_dir / f"shard_{shard_idx:04d}_of_{num_shards:04d}.safetensors"

    def shard_exists(self, shard_idx: int, num_shards: int) -> bool:
        """Check if a shard already exists.

        Args:
            shard_idx: Index of the shard
            num_shards: Total number of shards

        Returns:
            True if the shard exists, False otherwise
        """
        shard_path = self._get_shard_path(shard_idx, num_shards)
        return shard_path.exists()

    def write_shard(
        self,
        sample_data: Dict[str, Any],
        shard_idx: int,
        num_shards: int,
    ) -> None:
        """Write a sample shard to cache.

        Args:
            sample_data: Dictionary with sample data including:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - h_start: Hidden state before span
                - velocity_target: v_target = h_target - h_start for continuous-time training
                - h_target: Hidden state after span (optional, for verification)
                - teacher_logits: Optional logits
            shard_idx: Index of the shard
            num_shards: Total number of shards
        """
        shard_path = self._get_shard_path(shard_idx, num_shards)

        # Check if shard already exists and we're not overwriting
        if shard_path.exists() and not self.overwrite:
            logger.info(f"Shard {shard_idx} already exists, skipping (overwrite=False)")
            return

        # Prepare tensors for saving (clone to avoid shared memory issues)
        tensors_to_save = {}

        # Add input_ids and attention_mask
        if "input_ids" in sample_data:
            tensors_to_save["input_ids"] = sample_data["input_ids"].clone()
        if "attention_mask" in sample_data:
            tensors_to_save["attention_mask"] = sample_data["attention_mask"].clone()

        # Add h_start
        if "h_start" in sample_data:
            tensors_to_save["h_start"] = sample_data["h_start"].clone()

        # Add velocity_target for continuous-time training
        if "velocity_target" in sample_data:
            tensors_to_save["velocity_target"] = sample_data["velocity_target"].clone()

        # Add h_target (optional, for verification/debugging)
        if "h_target" in sample_data:
            tensors_to_save["h_target"] = sample_data["h_target"].clone()

        # Add teacher_logits if enabled
        if self.store_logits and "teacher_logits" in sample_data:
            tensors_to_save["teacher_logits"] = sample_data["teacher_logits"].clone()

        # Save using safetensors if available, otherwise torch.save
        if HAS_SAFETENSORS:
            save_file(tensors_to_save, str(shard_path))
        else:
            # Use .pt extension for torch format
            pt_path = str(shard_path).replace(".safetensors", ".pt")
            torch.save(tensors_to_save, pt_path)

        logger.info(f"Wrote shard {shard_idx}/{num_shards} to {shard_path}")


def generate_sample_cache(
    sample: Dict[str, torch.Tensor],
    inspector: QwenInspector,
    device: str = "cpu",
    store_logits: bool = False,
) -> Dict[str, Any]:
    """Generate cache data for a single sample.

    This function extracts teacher outputs and computes the velocity target
    for continuous-time training: v_target = h_target - h_start.

    Training state reconstruction:
        h_t = h_start + t[:, None, None] * velocity_target

    Args:
        sample: Dictionary with input_ids and attention_mask
        inspector: QwenInspector instance for extracting teacher outputs
        device: Device to run inference on
        store_logits: Whether to store teacher logits

    Returns:
        Dictionary with cached data:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - h_start: Hidden state before replacement span
            - velocity_target: v_target = h_target - h_start
            - h_target: Hidden state after replacement span (optional)
            - teacher_logits: Optional final logits
    """
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)

    # Handle batch dimension - add if needed
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    # Extract all teacher outputs
    outputs = inspector.extract_all(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Compute velocity target: v_target = h_target - h_start
    h_start = outputs["h_start"].cpu().squeeze(0)  # Remove batch dim
    h_target = outputs["h_target"].cpu().squeeze(0)  # Remove batch dim
    velocity_target = h_target - h_start

    # Prepare cache data with velocity target
    cache_data = {
        "input_ids": sample["input_ids"].cpu(),
        "attention_mask": sample["attention_mask"].cpu(),
        "h_start": h_start,
        "velocity_target": velocity_target,
        "h_target": h_target,  # Keep for verification/debugging
    }

    if store_logits:
        cache_data["teacher_logits"] = (
            outputs["logits"].cpu().squeeze(0)
        )  # Remove batch dim

    return cache_data


def generate_batch_cache(
    batch: Dict[str, torch.Tensor],
    inspector: QwenInspector,
    device: str = "cpu",
    store_logits: bool = False,
) -> List[Dict[str, Any]]:
    """Generate cache data for a batch of samples efficiently.

    This function processes the entire batch at once to leverage GPU parallelism,
    then splits the results into individual sample caches.

    Computes velocity targets: v_target = h_target - h_start for continuous-time training.

    Training state reconstruction:
        h_t = h_start + t[:, None, None] * velocity_target

    Args:
        batch: Dictionary with input_ids and attention_mask (batched)
        inspector: QwenInspector instance for extracting teacher outputs
        device: Device to run inference on
        store_logits: Whether to store teacher logits

    Returns:
        List of cache dictionaries, one per sample in the batch:
            Each dictionary contains:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - h_start: Hidden state before replacement span
            - velocity_target: v_target = h_target - h_start
            - h_target: Hidden state after replacement span (optional)
            - teacher_logits: Optional logits
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Ensure batch dimension exists
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    batch_size = input_ids.shape[0]

    # Extract all teacher outputs for the entire batch at once
    outputs = inspector.extract_all(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Split results into individual samples with velocity targets
    cache_list = []
    for i in range(batch_size):
        h_start = outputs["h_start"][i].cpu()
        h_target = outputs["h_target"][i].cpu()
        velocity_target = h_target - h_start

        cache_data = {
            "input_ids": batch["input_ids"][i].cpu(),
            "attention_mask": batch["attention_mask"][i].cpu(),
            "h_start": h_start,
            "velocity_target": velocity_target,
            "h_target": h_target,  # Keep for verification/debugging
        }

        if store_logits:
            cache_data["teacher_logits"] = outputs["logits"][i].cpu()

        cache_list.append(cache_data)

    return cache_list


def load_shard(
    cache_dir: Union[str, Path],
    shard_idx: int,
    num_shards: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load a cached shard.

    Args:
        cache_dir: Directory containing cache files
        shard_idx: Index of the shard to load
        num_shards: Total number of shards
        device: Device to load tensors to

    Returns:
        Dictionary with loaded tensors including velocity_target
    """
    cache_dir = Path(cache_dir)
    shard_path = cache_dir / f"shard_{shard_idx:04d}_of_{num_shards:04d}.safetensors"
    pt_path = cache_dir / f"shard_{shard_idx:04d}_of_{num_shards:04d}.pt"

    # Try safetensors first, then .pt fallback
    if shard_path.exists():
        load_path = shard_path
        use_safetensors = True
    elif pt_path.exists():
        load_path = pt_path
        use_safetensors = False
    else:
        raise FileNotFoundError(f"Shard not found: {shard_path} or {pt_path}")

    # Load using appropriate format
    if use_safetensors and HAS_SAFETENSORS:
        loaded = load_file(str(load_path))
    else:
        loaded = torch.load(load_path, map_location=device)

    # Note: velocity_target is stored directly, no reconstruction needed
    # For backward compatibility, reconstruct trajectory_targets if present in old format
    if "num_trajectory_targets" in loaded:
        num_targets = int(loaded["num_trajectory_targets"].item())
        trajectory_targets = []
        for i in range(num_targets):
            key = f"trajectory_target_{i}"
            if key in loaded:
                trajectory_targets.append(loaded[key])
        loaded["trajectory_targets"] = trajectory_targets

    return loaded


def load_metadata(cache_dir: Union[str, Path]) -> CacheMetadata:
    """Load cache metadata.

    Args:
        cache_dir: Directory containing cache files

    Returns:
        CacheMetadata object
    """
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        data = json.load(f)

    return CacheMetadata.from_dict(data)


def build_teacher_cache(
    cache_dir: Union[str, Path],
    model_name: str,
    start_layer: int = 8,
    end_layer: int = 11,
    seq_len: int = 128,
    store_logits: bool = False,
    device: str = "cpu",
    overwrite: bool = False,
) -> TeacherCacheWriter:
    """Create a TeacherCacheWriter with the specified configuration.

    This is a convenience factory function for creating a cache writer.

    Args:
        cache_dir: Directory to store cache files
        model_name: Name of the teacher model
        start_layer: First layer of replacement span
        end_layer: Last layer of replacement span
        seq_len: Sequence length for cached samples
        store_logits: Whether to store teacher logits
        device: Device for model inference
        overwrite: Whether to overwrite existing cache

    Returns:
        Configured TeacherCacheWriter instance
    """
    return TeacherCacheWriter(
        cache_dir=cache_dir,
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        seq_len=seq_len,
        store_logits=store_logits,
        overwrite=overwrite,
    )

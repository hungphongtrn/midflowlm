"""Tests for teacher cache generation with full trajectory targets.

These tests verify that we can:
1. Generate and cache teacher outputs (h_start, trajectory_targets, h_target, logits)
2. Write metadata about the cache
3. Support resumable/idempotent shard writing
4. Store span metadata (start_layer, end_layer, span_depth)
5. Optionally store logits
"""

import pytest
import torch
import yaml
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np


@pytest.fixture
def config():
    """Load v0 config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "v0_onemotif.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (2, 128), device=device),
        "attention_mask": torch.ones(2, 128, device=device),
    }


class TestTeacherCacheImports:
    """Test that teacher_cache module can be imported."""

    def test_import_teacher_cache(self):
        """Test that src.data.teacher_cache exists and can be imported."""
        from src.data import teacher_cache

        assert teacher_cache is not None

    def test_import_teacher_cache_writer(self):
        """Test that TeacherCacheWriter class exists."""
        from src.data.teacher_cache import TeacherCacheWriter

        assert TeacherCacheWriter is not None

    def test_import_cache_metadata(self):
        """Test that CacheMetadata class exists."""
        from src.data.teacher_cache import CacheMetadata

        assert CacheMetadata is not None


class TestCacheMetadata:
    """Test cache metadata structure and serialization."""

    def test_metadata_creation(self):
        """Test that CacheMetadata can be created with required fields."""
        from src.data.teacher_cache import CacheMetadata

        metadata = CacheMetadata(
            model_name="Qwen/Qwen3.5-0.8B",
            model_revision=None,
            start_layer=8,
            end_layer=11,
            span_depth=4,
            seq_len=128,
            store_logits=True,
            num_samples=100,
        )

        assert metadata.model_name == "Qwen/Qwen3.5-0.8B"
        assert metadata.start_layer == 8
        assert metadata.end_layer == 11
        assert metadata.span_depth == 4
        assert metadata.store_logits is True

    def test_metadata_to_dict(self):
        """Test that CacheMetadata can be converted to dictionary."""
        from src.data.teacher_cache import CacheMetadata

        metadata = CacheMetadata(
            model_name="Qwen/Qwen3.5-0.8B",
            model_revision="abc123",
            start_layer=8,
            end_layer=11,
            span_depth=4,
            seq_len=128,
            store_logits=True,
            num_samples=100,
        )

        data = metadata.to_dict()
        assert data["model_name"] == "Qwen/Qwen3.5-0.8B"
        assert data["model_revision"] == "abc123"
        assert data["start_layer"] == 8
        assert data["end_layer"] == 11
        assert data["store_logits"] is True

    def test_metadata_from_dict(self):
        """Test that CacheMetadata can be loaded from dictionary."""
        from src.data.teacher_cache import CacheMetadata

        data = {
            "model_name": "Qwen/Qwen3.5-0.8B",
            "model_revision": "abc123",
            "start_layer": 8,
            "end_layer": 11,
            "span_depth": 4,
            "seq_len": 128,
            "store_logits": True,
            "num_samples": 100,
        }

        metadata = CacheMetadata.from_dict(data)
        assert metadata.model_name == "Qwen/Qwen3.5-0.8B"
        assert metadata.start_layer == 8
        assert metadata.end_layer == 11


class TestTeacherCacheWriter:
    """Test TeacherCacheWriter functionality."""

    def test_cache_writer_initialization(self, config, temp_cache_dir):
        """Test that TeacherCacheWriter initializes correctly."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=config["replacement_model"]["start_layer"],
            end_layer=config["replacement_model"]["end_layer"],
            seq_len=config["data"]["seq_len"],
            store_logits=config["teacher_cache"]["store_logits"],
        )

        assert writer.cache_dir == Path(temp_cache_dir)
        assert writer.model_name == config["model"]["name"]
        assert writer.start_layer == 8
        assert writer.end_layer == 11
        assert writer.span_depth == 4

    def test_cache_directory_creation(self, config, temp_cache_dir):
        """Test that cache directory is created on initialization."""
        from src.data.teacher_cache import TeacherCacheWriter

        cache_path = Path(temp_cache_dir) / "subdir"
        assert not cache_path.exists()

        writer = TeacherCacheWriter(
            cache_dir=cache_path,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_metadata_writing(self, config, temp_cache_dir):
        """Test that metadata is written to cache directory."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        writer.write_metadata(num_samples=100)

        metadata_path = Path(temp_cache_dir) / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["model_name"] == config["model"]["name"]
        assert metadata["start_layer"] == 8
        assert metadata["end_layer"] == 11
        assert metadata["span_depth"] == 4
        assert metadata["num_samples"] == 100
        assert metadata["store_logits"] is True

    def test_span_metadata_storage(self, config, temp_cache_dir):
        """Test that span metadata (start_layer, end_layer, span_depth) is stored."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=4,
            end_layer=7,
            seq_len=128,
            store_logits=True,
        )

        writer.write_metadata(num_samples=50)

        metadata_path = Path(temp_cache_dir) / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["start_layer"] == 4
        assert metadata["end_layer"] == 7
        assert metadata["span_depth"] == 4

    def test_resumable_shard_writing(self, config, temp_cache_dir):
        """Test that shard writing is resumable/idempotent."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        # Create a mock sample data
        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        # Write shard 0
        writer.write_shard(sample_data, shard_idx=0, num_shards=2)

        shard_path = Path(temp_cache_dir) / "shard_0000_of_0002.safetensors"
        assert shard_path.exists()

        # Writing same shard again should overwrite (idempotent)
        writer.write_shard(sample_data, shard_idx=0, num_shards=2)
        assert shard_path.exists()

    def test_shard_presence_check(self, config, temp_cache_dir):
        """Test that we can check if a shard already exists."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        # Initially shard should not exist
        assert not writer.shard_exists(0, 2)

        # Create a dummy shard file
        shard_path = Path(temp_cache_dir) / "shard_0000_of_0002.safetensors"
        shard_path.touch()

        # Now it should exist
        assert writer.shard_exists(0, 2)

    def test_complete_shard_overwrite(self, config, temp_cache_dir):
        """Test that overwrite flag controls whether to overwrite existing shards."""
        from src.data.teacher_cache import TeacherCacheWriter

        # Writer with overwrite=False
        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
            overwrite=False,
        )

        # Create existing shard
        shard_path = Path(temp_cache_dir) / "shard_0000_of_0002.safetensors"
        shard_path.touch()

        assert writer.shard_exists(0, 2)


class TestCacheContents:
    """Test that cached data contains expected keys and structure."""

    def test_cache_has_h_start(self, config, temp_cache_dir, device):
        """Test that cache contains h_start."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        # Verify h_start is in the cached data
        assert "h_start" in sample_data
        assert sample_data["h_start"].shape == (128, 896)

    def test_cache_has_trajectory_targets(self, config, temp_cache_dir):
        """Test that cache contains trajectory_targets list."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        trajectory_targets = [torch.randn(128, 896) for _ in range(4)]
        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": trajectory_targets,
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        assert "trajectory_targets" in sample_data
        assert len(sample_data["trajectory_targets"]) == 4

    def test_cache_has_h_target(self, config, temp_cache_dir):
        """Test that cache contains h_target."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        assert "h_target" in sample_data
        assert sample_data["h_target"].shape == (128, 896)

    def test_cache_has_optional_logits(self, config, temp_cache_dir):
        """Test that cache can store optional teacher_logits."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        assert "teacher_logits" in sample_data
        assert sample_data["teacher_logits"].shape == (128, 1000)

    def test_cache_skips_logits_when_disabled(self, config, temp_cache_dir):
        """Test that logits are not stored when store_logits=False."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=False,
        )

        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            # teacher_logits not included
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        # Should succeed without logits
        metadata_path = Path(temp_cache_dir) / "metadata.json"
        writer.write_metadata(num_samples=1)

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["store_logits"] is False


class TestCacheGeneration:
    """Test the full cache generation process."""

    def test_generate_sample_cache(self, config, temp_cache_dir, device):
        """Test generating cache for a single sample."""
        from src.data.teacher_cache import TeacherCacheWriter, generate_sample_cache

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        sample = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
        }

        # Mock the QwenInspector to avoid loading actual model
        mock_inspector = MagicMock()
        mock_inspector.extract_all.return_value = {
            "embeddings": torch.randn(1, 128, 896),
            "h_start": torch.randn(1, 128, 896),
            "span_states": [torch.randn(1, 128, 896) for _ in range(4)],
            "h_target": torch.randn(1, 128, 896),
            "logits": torch.randn(1, 128, 1000),
        }

        with patch("src.data.teacher_cache.QwenInspector", return_value=mock_inspector):
            cache_data = generate_sample_cache(
                sample=sample,
                inspector=mock_inspector,
                device=device,
                store_logits=True,
            )

        assert "input_ids" in cache_data
        assert "attention_mask" in cache_data
        assert "h_start" in cache_data
        assert "velocity_target" in cache_data


class TestVelocityTargetContract:
    """Test velocity target generation and storage contract."""

    def test_cache_writer_stores_velocity_target(self, config, temp_cache_dir):
        """Test that cache writer stores velocity_target as h_end - h_start."""
        from src.data.teacher_cache import TeacherCacheWriter, load_shard

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        h_start = torch.randn(128, 896)
        h_target = torch.randn(128, 896)

        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": h_start,
            "velocity_target": h_target - h_start,
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        # Load and verify velocity_target is present
        loaded = load_shard(temp_cache_dir, shard_idx=0, num_shards=1)
        assert "velocity_target" in loaded
        assert loaded["velocity_target"].shape == (128, 896)

    def test_build_teacher_cache_uses_h_end_minus_h_start(self, temp_cache_dir):
        """Test that generate_sample_cache computes velocity_target = h_end - h_start."""
        from src.data.teacher_cache import generate_sample_cache

        h_start = torch.randn(1, 16, 32)
        h_target = torch.randn(1, 16, 32)
        span_states = [torch.randn(1, 16, 32) for _ in range(4)]

        mock_inspector = MagicMock()
        mock_inspector.extract_all.return_value = {
            "embeddings": torch.randn(1, 16, 32),
            "h_start": h_start,
            "span_states": span_states,
            "h_target": h_target,
            "logits": torch.randn(1, 16, 1000),
        }

        sample = {
            "input_ids": torch.randint(0, 1000, (16,)),
            "attention_mask": torch.ones(16),
        }

        cache_data = generate_sample_cache(
            sample=sample,
            inspector=mock_inspector,
            device="cpu",
            store_logits=True,
        )

        assert "velocity_target" in cache_data
        expected_velocity = h_target.squeeze(0) - h_start.squeeze(0)
        assert torch.allclose(cache_data["velocity_target"], expected_velocity)

    def test_batch_cache_computes_velocity_target(self):
        """Test that generate_batch_cache computes velocity_target for each sample."""
        from src.data.teacher_cache import generate_batch_cache

        batch_size = 4
        h_start = torch.randn(batch_size, 16, 32)
        h_target = torch.randn(batch_size, 16, 32)
        span_states = [torch.randn(batch_size, 16, 32) for _ in range(4)]

        mock_inspector = MagicMock()
        mock_inspector.extract_all.return_value = {
            "embeddings": torch.randn(batch_size, 16, 32),
            "h_start": h_start,
            "span_states": span_states,
            "h_target": h_target,
            "logits": torch.randn(batch_size, 16, 1000),
        }

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, 16)),
            "attention_mask": torch.ones(batch_size, 16),
        }

        cache_list = generate_batch_cache(
            batch=batch,
            inspector=mock_inspector,
            device="cpu",
            store_logits=True,
        )

        assert len(cache_list) == batch_size
        for i, cache_data in enumerate(cache_list):
            assert "velocity_target" in cache_data
            expected_velocity = h_target[i] - h_start[i]
            assert torch.allclose(cache_data["velocity_target"], expected_velocity)


class TestCacheLoading:
    """Test loading cached data."""

    def test_load_shard(self, config, temp_cache_dir):
        """Test loading a cached shard."""
        from src.data.teacher_cache import TeacherCacheWriter, load_shard

        writer = TeacherCacheWriter(
            cache_dir=temp_cache_dir,
            model_name=config["model"]["name"],
            start_layer=8,
            end_layer=11,
            seq_len=128,
            store_logits=True,
        )

        # Create sample data and write shard
        sample_data = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "h_start": torch.randn(128, 896),
            "trajectory_targets": [torch.randn(128, 896) for _ in range(4)],
            "h_target": torch.randn(128, 896),
            "teacher_logits": torch.randn(128, 1000),
        }

        writer.write_shard(sample_data, shard_idx=0, num_shards=1)

        # Load the shard
        loaded = load_shard(temp_cache_dir, shard_idx=0, num_shards=1)

        assert "input_ids" in loaded
        assert "h_start" in loaded
        assert "h_target" in loaded


class TestDatasetFactory:
    """Test dataset factory dispatching between loaders."""

    def test_dataset_factory_dispatches_to_tinystories_loader(self):
        """Test that factory dispatches to tinystories loader for legacy configs."""
        from src.data.dataset_factory import get_experiment_dataloaders

        config = MagicMock()
        config.data.dataset_name = "roneneldan/TinyStories"
        config.data.dataset_revision = None
        config.data.text_field = "text"
        config.data.seq_len = 128
        config.data.train_samples = 100
        config.data.val_samples = 20
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.persistent_workers = False
        config.data.shuffle_seed = 1337
        config.model.name = "Qwen/Qwen3.5-0.8B"
        config.model.revision = None

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        with patch("src.data.dataset_factory.get_tinystories_dataloaders") as mock_ts:
            mock_ts.return_value = {"train": MagicMock(), "val": MagicMock()}
            result = get_experiment_dataloaders(config, tokenizer=mock_tokenizer)
            mock_ts.assert_called_once()

    def test_dataset_factory_dispatches_to_mixture_loader(self):
        """Test that factory dispatches to mixture loader for mixed corpus configs."""
        from src.data.dataset_factory import get_experiment_dataloaders

        config = MagicMock()
        config.data.loader = "mixture"
        config.data.seq_len = 128
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.persistent_workers = False
        config.data.shuffle_seed = 1337
        config.data.mixture_components = []
        config.model.name = "Qwen/Qwen3.5-0.8B"
        config.model.revision = None

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        with patch(
            "src.data.dataset_factory.get_mixed_corpus_dataloaders"
        ) as mock_mixed:
            mock_mixed.return_value = {"train": MagicMock(), "val": MagicMock()}
            result = get_experiment_dataloaders(config, tokenizer=mock_tokenizer)
            mock_mixed.assert_called_once()

    def test_dataset_factory_raises_on_unknown_loader(self):
        """Test that factory raises ValueError for unknown loader type."""
        from src.data.dataset_factory import get_experiment_dataloaders

        config = MagicMock()
        config.data.loader = "unknown_loader"
        config.model.name = "Qwen/Qwen3.5-0.8B"
        config.model.revision = None

        mock_tokenizer = MagicMock()

        with pytest.raises(ValueError, match="Unsupported data.loader"):
            get_experiment_dataloaders(config, tokenizer=mock_tokenizer)

    def test_dataset_factory_normalizes_config(self):
        """Test that factory normalizes data config to preserve mixture_components."""
        from src.data.dataset_factory import normalize_data_config

        raw_config = MagicMock()
        raw_config.__dict__ = {"mixture_components": [{"name": "test"}]}
        raw_config.mixture_components = None

        normalized = normalize_data_config(raw_config)
        assert hasattr(normalized, "mixture_components")
        assert normalized.mixture_components == [{"name": "test"}]

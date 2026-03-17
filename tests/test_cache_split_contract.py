"""Contract tests for cache split paths and hidden-state-only defaults.

These tests verify the cache contract:
1. Split-specific cache directories (train/val/test)
2. Hidden-state-only default (no logits)
3. Legacy logit-bearing cache compatibility
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCacheSplitContract:
    """Test split-specific cache directory contract."""

    def test_resolve_split_cache_dir_appends_split_subdir(self, tmp_path):
        """Test that resolve_split_cache_dir appends split as subdirectory."""
        from src.data.teacher_cache import resolve_split_cache_dir

        cache_root = tmp_path / "teacher_cache"
        assert resolve_split_cache_dir(cache_root, "train") == cache_root / "train"
        assert resolve_split_cache_dir(cache_root, "val") == cache_root / "val"
        assert resolve_split_cache_dir(cache_root, "test") == cache_root / "test"

    def test_resolve_split_cache_dir_with_string_path(self, tmp_path):
        """Test that resolve_split_cache_dir works with string paths."""
        from src.data.teacher_cache import resolve_split_cache_dir

        cache_root = str(tmp_path / "teacher_cache")
        result = resolve_split_cache_dir(cache_root, "train")
        assert result == Path(cache_root) / "train"


class TestHiddenStateOnlyContract:
    """Test hidden-state-only cache contract (no logits by default)."""

    def test_load_shard_reconstructs_hidden_state_only_targets(self, tmp_path):
        """Test loading shards without teacher_logits when store_logits=False."""
        from src.data.teacher_cache import TeacherCacheWriter, load_shard

        writer = TeacherCacheWriter(
            cache_dir=tmp_path,
            model_name="test-model",
            store_logits=False,
        )
        writer.write_shard(
            {
                "input_ids": torch.randint(0, 10, (16,)),
                "attention_mask": torch.ones(16),
                "h_start": torch.randn(16, 32),
                "trajectory_targets": [torch.randn(16, 32) for _ in range(4)],
                "h_target": torch.randn(16, 32),
            },
            shard_idx=0,
            num_shards=1,
        )

        loaded = load_shard(tmp_path, shard_idx=0, num_shards=1)
        assert "teacher_logits" not in loaded
        assert "trajectory_targets" in loaded
        assert len(loaded["trajectory_targets"]) == 4
        assert "h_start" in loaded
        assert "h_target" in loaded

    def test_load_shard_keeps_legacy_teacher_logits_when_present(self, tmp_path):
        """Test that legacy shards with logits still load successfully."""
        from src.data.teacher_cache import TeacherCacheWriter, load_shard

        writer = TeacherCacheWriter(
            cache_dir=tmp_path,
            model_name="test-model",
            store_logits=True,
        )
        writer.write_shard(
            {
                "input_ids": torch.randint(0, 10, (16,)),
                "attention_mask": torch.ones(16),
                "h_start": torch.randn(16, 32),
                "trajectory_targets": [torch.randn(16, 32) for _ in range(4)],
                "h_target": torch.randn(16, 32),
                "teacher_logits": torch.randn(16, 64),
            },
            shard_idx=0,
            num_shards=1,
        )

        loaded = load_shard(tmp_path, shard_idx=0, num_shards=1)
        assert "teacher_logits" in loaded
        assert "trajectory_targets" in loaded


class TestCacheMetadataDefaults:
    """Test cache metadata defaults to hidden-state-only."""

    def test_cache_metadata_defaults_to_hidden_state_only(self):
        """Test that CacheMetadata defaults to store_logits=False."""
        from src.data.teacher_cache import CacheMetadata

        metadata = CacheMetadata(
            model_name="Qwen/Qwen3.5-0.8B",
            model_revision=None,
            start_layer=8,
            end_layer=11,
            span_depth=4,
            seq_len=128,
            num_samples=8,
        )

        assert metadata.store_logits is False

    def test_teacher_cache_writer_defaults_to_hidden_state_only(self, tmp_path):
        """Test that TeacherCacheWriter defaults to store_logits=False."""
        from src.data.teacher_cache import TeacherCacheWriter

        writer = TeacherCacheWriter(
            cache_dir=tmp_path,
            model_name="test-model",
        )

        assert writer.store_logits is False

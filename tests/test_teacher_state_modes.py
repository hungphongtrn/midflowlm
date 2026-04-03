"""Tests for teacher_state mode contract.

These tests verify the control-plane contract for teacher-state modes:
- offline_cache: Uses pre-built teacher cache (existing path)
- online_no_cache: Uses live teacher extraction, no cache writing
- online_write_through_cache: Uses live teacher extraction with cache writing

Task 4 contract tests:
1. Existing configs without teacher_state.mode default to offline_cache
2. online_no_cache requires a teacher model / live-teacher prerequisites
3. Mode helper exposes booleans: requires_cache, requires_live_teacher, allow_cache_write
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


class TestTeacherStateModeResolution:
    """Tests for teacher_state mode resolution and defaults."""

    def test_existing_config_without_mode_defaults_to_offline_cache(self):
        """Configs without explicit teacher_state.mode should resolve to offline_cache."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config = {
            "teacher_cache": {"enabled": True, "cache_dir": "./cache/test"},
            "model": {"name": "Qwen/Qwen3.5-0.8B"},
            "replacement_model": {"start_layer": 8, "end_layer": 11},
            "data": {"seq_len": 128},
        }

        mode = resolve_teacher_state_mode(config)
        assert mode == "offline_cache"

    def test_explicit_offline_cache_mode(self):
        """Explicit teacher_state.mode=offline_cache should resolve correctly."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config = {
            "teacher_state": {"mode": "offline_cache"},
            "teacher_cache": {"enabled": True, "cache_dir": "./cache/test"},
        }

        mode = resolve_teacher_state_mode(config)
        assert mode == "offline_cache"

    def test_explicit_online_no_cache_mode(self):
        """Explicit teacher_state.mode=online_no_cache should resolve correctly."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config = {
            "teacher_state": {"mode": "online_no_cache"},
        }

        mode = resolve_teacher_state_mode(config)
        assert mode == "online_no_cache"

    def test_explicit_online_write_through_cache_mode(self):
        """Explicit teacher_state.mode=online_write_through_cache should resolve correctly."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config = {
            "teacher_state": {"mode": "online_write_through_cache"},
        }

        mode = resolve_teacher_state_mode(config)
        assert mode == "online_write_through_cache"


class TestTeacherStateModeBooleans:
    """Tests for per-mode boolean helpers."""

    def test_offline_cache_booleans(self):
        """offline_cache should require_cache=True, requires_live_teacher=False, allow_cache_write=False."""
        from src.training.teacher_state import TeacherStateMode

        mode = TeacherStateMode.OFFLINE_CACHE
        assert mode.requires_cache() is True
        assert mode.requires_live_teacher() is False
        assert mode.allow_cache_write() is False

    def test_online_no_cache_booleans(self):
        """online_no_cache should require_cache=False, requires_live_teacher=True, allow_cache_write=False."""
        from src.training.teacher_state import TeacherStateMode

        mode = TeacherStateMode.ONLINE_NO_CACHE
        assert mode.requires_cache() is False
        assert mode.requires_live_teacher() is True
        assert mode.allow_cache_write() is False

    def test_online_write_through_cache_booleans(self):
        """online_write_through_cache should require_cache=False, requires_live_teacher=True, allow_cache_write=True."""
        from src.training.teacher_state import TeacherStateMode

        mode = TeacherStateMode.ONLINE_WRITE_THROUGH_CACHE
        assert mode.requires_cache() is False
        assert mode.requires_live_teacher() is True
        assert mode.allow_cache_write() is True


class TestTeacherStateValidation:
    """Tests for teacher_state config validation."""

    def test_offline_cache_requires_cache_dir(self):
        """offline_cache mode should validate cache_dir existence."""
        from src.training.teacher_state import validate_teacher_state_config

        config = {
            "teacher_state": {"mode": "offline_cache"},
            "teacher_cache": {"enabled": True, "cache_dir": "./cache/nonexistent"},
        }

        with pytest.raises(ValueError, match="cache_dir.*exist|not found"):
            validate_teacher_state_config(config)

    def test_online_no_cache_requires_teacher_model(self):
        """online_no_cache mode should require model config for live extraction."""
        from src.training.teacher_state import validate_teacher_state_config

        config = {
            "teacher_state": {"mode": "online_no_cache"},
            "model": {"name": "Qwen/Qwen3.5-0.8B"},
        }

        result = validate_teacher_state_config(config)
        assert result is None

    def test_online_no_cache_without_model_raises(self):
        """online_no_cache without model config should raise ValueError."""
        from src.training.teacher_state import validate_teacher_state_config

        config = {
            "teacher_state": {"mode": "online_no_cache"},
        }

        with pytest.raises(ValueError, match="model\\.name|teacher extraction"):
            validate_teacher_state_config(config)

    def test_write_through_cache_requires_cache_config(self):
        """online_write_through_cache should require teacher_cache.enabled=True."""
        from src.training.teacher_state import validate_teacher_state_config

        config = {
            "teacher_state": {"mode": "online_write_through_cache"},
            "teacher_cache": {"enabled": False},
        }

        with pytest.raises(ValueError, match="teacher_cache.*enabled|cache.*enabled"):
            validate_teacher_state_config(config)


class TestTeacherStateContractIntegration:
    """Integration tests for the teacher_state contract with configs."""

    def test_long_context_config_has_offline_cache_mode(self):
        """The long-context config should have explicit offline_cache mode."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "v0_mixed_corpus_plus_kl_loss_long_context.yaml"
        )
        if not config_path.exists():
            pytest.skip("Long context config not yet created")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        mode = resolve_teacher_state_mode(config)
        assert mode == "offline_cache"

    def test_base_kl_config_resolves_to_offline_cache(self):
        """The base KL config should declare and resolve to offline_cache."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "v0_mixed_corpus_plus_kl_loss.yaml"
        )
        if not config_path.exists():
            pytest.skip("Base KL config not yet in worktree")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["teacher_state"]["mode"] == "offline_cache"
        mode = resolve_teacher_state_mode(config)
        assert mode == "offline_cache"


class TestTeacherStateSmokeConfigs:
    """Tests for the smoke config files."""

    def test_online_no_cache_smoke_config_exists(self):
        """The online_no_cache smoke config should exist and have correct mode."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "v0_teacher_state_online_no_cache_smoke.yaml"
        )
        if not config_path.exists():
            pytest.skip("online_no_cache smoke config not yet created")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        mode = resolve_teacher_state_mode(config)
        assert mode == "online_no_cache"

    def test_write_through_cache_smoke_config_exists(self):
        """The write_through_cache smoke config should exist and have correct mode."""
        from src.training.teacher_state import resolve_teacher_state_mode

        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "v0_teacher_state_write_through_cache_smoke.yaml"
        )
        if not config_path.exists():
            pytest.skip("write_through_cache smoke config not yet created")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        mode = resolve_teacher_state_mode(config)
        assert mode == "online_write_through_cache"

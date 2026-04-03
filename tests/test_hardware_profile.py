"""Tests for hardware profile loader module."""

import json
import pytest
import tempfile
from pathlib import Path

from src.utils.hardware_profile import (
    load_hardware_profile,
    apply_hardware_profile_to_config,
    get_default_profile_path,
)


def test_load_hardware_profile_valid():
    """Test loading a valid hardware profile."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        profile = {
            "hardware": "NVIDIA RTX 3090 24GB",
            "seq_len": 1024,
            "microbatch_size": 2,
            "gradient_accumulation": 8,
            "effective_batch_size": 16,
            "precision": "bf16-mixed",
            "gradient_checkpointing": True,
            "peak_vram_gb": 21.5,
            "tokens_per_sec": 420.0,
            "calibrated_on": "FlowMidblock_EndTrajKLCe_MixC",
        }
        json.dump(profile, f)
        f.flush()

        loaded = load_hardware_profile(f.name)
        assert loaded["hardware"] == "NVIDIA RTX 3090 24GB"
        assert loaded["microbatch_size"] == 2
        assert loaded["gradient_accumulation"] == 8

    Path(f.name).unlink()


def test_load_hardware_profile_missing_file():
    """Test loading a non-existent profile raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_hardware_profile("nonexistent_profile.json")


def test_load_hardware_profile_missing_fields():
    """Test loading a profile with missing required fields raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Missing several required fields
        profile = {"hardware": "NVIDIA RTX 3090 24GB"}
        json.dump(profile, f)
        f.flush()

        with pytest.raises(ValueError) as exc_info:
            load_hardware_profile(f.name)

        assert "missing required fields" in str(exc_info.value)

    Path(f.name).unlink()


def test_apply_hardware_profile_to_config():
    """Test applying hardware profile to training config."""
    config = {
        "experiment_name": "test_exp",
        "model": {"name": "Qwen/Qwen3.5-0.8B"},
        "data": {"batch_size": 999, "seq_len": 999},  # Will be overridden
        "train_loop": {"accumulate_grad_batches": 999},  # Will be overridden
    }

    profile = {
        "hardware": "NVIDIA RTX 3090 24GB",
        "seq_len": 1024,
        "microbatch_size": 2,
        "gradient_accumulation": 8,
        "effective_batch_size": 16,
        "precision": "bf16-mixed",
        "gradient_checkpointing": True,
        "peak_vram_gb": 21.5,
        "tokens_per_sec": 420.0,
        "calibrated_on": "FlowMidblock_EndTrajKLCe_MixC",
    }

    updated = apply_hardware_profile_to_config(config, profile)

    # Check that profile values were applied
    assert updated["data"]["batch_size"] == 2  # microbatch_size
    assert updated["data"]["seq_len"] == 1024
    assert updated["train_loop"]["accumulate_grad_batches"] == 8
    assert updated["train_loop"]["precision"] == "bf16-mixed"
    assert updated["train_loop"]["gradient_checkpointing"] is True

    # Check that original values were preserved where not overridden
    assert updated["experiment_name"] == "test_exp"
    assert updated["model"]["name"] == "Qwen/Qwen3.5-0.8B"

    # Check metadata was added
    assert "_hardware_profile" in updated
    assert updated["_hardware_profile"]["source"] == "FlowMidblock_EndTrajKLCe_MixC"


def test_apply_hardware_profile_creates_missing_sections():
    """Test applying profile to config with missing sections."""
    config = {"experiment_name": "test"}  # No data or train_loop

    profile = {
        "hardware": "NVIDIA RTX 3090 24GB",
        "seq_len": 1024,
        "microbatch_size": 2,
        "gradient_accumulation": 8,
        "effective_batch_size": 16,
        "precision": "bf16-mixed",
        "gradient_checkpointing": True,
        "calibrated_on": "test",
    }

    updated = apply_hardware_profile_to_config(config, profile)

    assert updated["data"]["batch_size"] == 2
    assert updated["train_loop"]["accumulate_grad_batches"] == 8


def test_default_profile_path():
    """Test that default profile path is valid."""
    path = get_default_profile_path()
    assert "profiles" in path
    assert "v0_1_3090_profile.json" in path

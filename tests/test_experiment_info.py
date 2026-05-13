"""Tests for _generate_experiment_info helper."""

import datetime
import unittest.mock

from scripts.train_sft import _generate_experiment_info


def test_generates_all_expected_keys():
    """Every key in the spec should be present."""
    config = {"model": {"name": "Qwen/Qwen3.5-0.8B", "start_layer": 8, "end_layer": 11, "thinking_level": 32}}
    train_result = unittest.mock.MagicMock(global_step=100, training_loss=0.5, total_flos=12345.0)
    model = unittest.mock.MagicMock(trainable_params=1000, frozen_params=500000)

    info = _generate_experiment_info(config, train_result, model)

    expected_keys = {
        "experiment_key", "name", "architecture", "base_model",
        "start_layer", "end_layer", "thinking_level",
        "trainable_params", "frozen_params",
        "global_step", "training_loss", "total_flos",
        "training_completed_at",
    }
    assert expected_keys <= set(info.keys())


def test_uses_defaults_when_config_keys_missing():
    """Missing optional config keys should fall back to defaults."""
    config = {"model": {"name": "TestModel"}}
    train_result = unittest.mock.MagicMock(global_step=0, training_loss=None, total_flos=0)
    model = unittest.mock.MagicMock(spec=[])

    info = _generate_experiment_info(config, train_result, model)

    assert info["start_layer"] == 8
    assert info["end_layer"] == 11
    assert info["thinking_level"] == 32


def test_uses_none_when_model_attrs_missing():
    """Missing model attributes should default to None."""
    config = {"model": {"name": "TestModel"}}
    train_result = unittest.mock.MagicMock(global_step=0, training_loss=None, total_flos=0)
    model = unittest.mock.MagicMock(spec=[])

    info = _generate_experiment_info(config, train_result, model)

    assert info["trainable_params"] is None
    assert info["frozen_params"] is None


def test_training_completed_at_is_utc_iso_format():
    """Timestamp should be an ISO-formatted UTC datetime string."""
    config = {"model": {"name": "TestModel"}}
    train_result = unittest.mock.MagicMock(global_step=0, training_loss=0.0, total_flos=0)
    model = unittest.mock.MagicMock(spec=[])

    info = _generate_experiment_info(config, train_result, model)

    ts = info["training_completed_at"]
    parsed = datetime.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None, "timestamp should be timezone-aware"
    # Should be within the last few seconds
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = now - parsed
    assert delta.total_seconds() < 5, f"timestamp too old: {delta.total_seconds()}s ago"

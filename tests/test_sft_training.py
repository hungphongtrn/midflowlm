"""Tests for SFT training utilities."""

import pytest
import torch

from src.training.sft_utils import (
    MidblockMetricsCallback,
    estimate_training_budget,
    validate_model_for_training,
)


class TestMidblockMetricsCallback:
    """Verify callback initialization and basic behavior."""

    def test_callback_initializes(self):
        cb = MidblockMetricsCallback()
        assert cb is not None


class TestEstimateTrainingBudget:
    """Verify budget estimation math."""

    def test_budget_math(self):
        budget = estimate_training_budget(
            num_sequences=1000,
            seq_length=8192,
            thinking_level=32,
            batch_size=1,
            grad_accum=4,
            num_epochs=1,
            steps_per_second_estimate=0.5,
        )
        assert budget["total_steps"] == 250
        assert budget["effective_batch_size"] == 4
        assert budget["steps_per_epoch"] == 250
        assert budget["total_tokens"] > 0
        assert budget["estimated_hours"] > 0

    def test_zero_sequences(self):
        budget = estimate_training_budget(
            num_sequences=0,
            seq_length=8192,
            batch_size=1,
            grad_accum=1,
        )
        assert budget["total_steps"] == 0
        assert budget["total_tokens"] == 0


class TestValidateModelForTraining:
    """Verify validation checks - use a simple mock."""

    class MockModel:
        def __init__(self, thinking_level=32):
            self.thinking_level = thinking_level
            self._params = {}

        def named_parameters(self):
            return self._params.items()

        def add_trainable_midblock(self, name="midblock.0.weight"):
            parameter = torch.nn.Parameter(torch.randn(3, 3))
            self._params[name] = parameter

    def test_validates_only_midblock_trainable(self):
        model = self.MockModel()
        model.add_trainable_midblock("midblock.0.weight")
        validate_model_for_training(model)

    def test_raises_on_non_midblock_trainable(self):
        model = self.MockModel()
        model.add_trainable_midblock("qwen.layers.0.weight")
        with pytest.raises(ValueError, match="Non-midblock"):
            validate_model_for_training(model)

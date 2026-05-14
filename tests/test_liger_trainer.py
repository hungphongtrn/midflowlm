from types import SimpleNamespace
from unittest.mock import patch

from transformers import Trainer

from src.training.liger_trainer import LigerTrainer


class DummyModel:
    uses_liger_kernel = True


def _trainer(use_liger_kernel=True):
    trainer = LigerTrainer.__new__(LigerTrainer)
    trainer.args = SimpleNamespace(use_liger_kernel=use_liger_kernel)
    trainer.model = DummyModel()
    return trainer


def test_prediction_step_adds_skip_logits_for_liger_loss_only_eval():
    trainer = _trainer(use_liger_kernel=True)
    inputs = {"input_ids": [1, 2], "labels": [1, 2]}
    captured = {}

    def fake_prediction_step(self, model, next_inputs, prediction_loss_only, ignore_keys=None):
        captured.update(next_inputs)
        return "loss", None, None

    with patch.object(Trainer, "prediction_step", fake_prediction_step):
        out = trainer.prediction_step(DummyModel(), inputs, True)

    assert out == ("loss", None, None)
    assert captured["skip_logits"] is True
    assert "skip_logits" not in inputs


def test_prediction_step_leaves_inputs_unchanged_when_liger_arg_disabled():
    trainer = _trainer(use_liger_kernel=False)
    inputs = {"input_ids": [1, 2], "labels": [1, 2]}
    captured = {}

    def fake_prediction_step(self, model, next_inputs, prediction_loss_only, ignore_keys=None):
        captured.update(next_inputs)
        return "loss", None, None

    with patch.object(Trainer, "prediction_step", fake_prediction_step):
        trainer.prediction_step(DummyModel(), inputs, True)

    assert "skip_logits" not in captured

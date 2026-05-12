"""HF Trainer callbacks and utilities for SFT training with FlowMidblock."""

import logging
import os

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MidblockSaveCallback(TrainerCallback):
    """Saves only midblock weights on HF Trainer checkpoint events.

    Redundant full-model saves (frozen Qwen backbone) are wasteful — only the
    midblock changes during training.  This callback overwrites the checkpoint
    directory with a compact ``midblock.pth`` on every ``on_save`` event.
    """

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "midblock"):
            return
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "midblock.pth")
        torch.save(model.midblock.state_dict(), save_path)
        logger.info("Midblock checkpoint saved to %s", save_path)


class MidblockMetricsCallback(TrainerCallback):
    """Custom HF Trainer callback that logs FlowMidblock-specific metrics.

    Stores a direct reference to the midblock module at on_train_begin
    to avoid wrapper/copy resolution issues in subsequent callbacks.
    """

    def __init__(self):
        super().__init__()
        self._last_grad_norm = 0.0
        self._midblock = None

    def _resolve_midblock(self, model):
        if model is None:
            return None
        if hasattr(model, "midblock"):
            return model.midblock
        if hasattr(model, "module") and hasattr(model.module, "midblock"):
            return model.module.midblock
        current = model
        for _ in range(5):
            if hasattr(current, "module"):
                current = current.module
            elif hasattr(current, "_orig_mod"):
                current = current._orig_mod
            else:
                break
        if hasattr(current, "midblock"):
            return current.midblock
        return None

    def _compute_midblock_norms(self):
        grad_norm = 0.0
        param_norm = 0.0
        total_params = 0
        for parameter in self._midblock.parameters():
            if parameter.requires_grad:
                total_params += parameter.numel()
                param_norm += parameter.data.float().norm(2).item() ** 2
                if parameter.grad is not None:
                    grad_norm += parameter.grad.float().norm(2).item() ** 2
        return grad_norm**0.5, param_norm**0.5, total_params

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._midblock = self._resolve_midblock(model)
            if hasattr(model, "trainable_params"):
                logger.info("Trainable parameters: %s", f"{model.trainable_params:,}")
            if hasattr(model, "frozen_params"):
                logger.info("Frozen parameters:    %s", f"{model.frozen_params:,}")

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self._midblock is None:
            return
        grad_norm, _, _ = self._compute_midblock_norms()
        self._last_grad_norm = grad_norm

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._midblock is None or logs is None:
            return
        _, param_norm, total_params = self._compute_midblock_norms()
        logs["midblock/grad_norm"] = self._last_grad_norm
        logs["midblock/param_norm"] = param_norm
        logs["midblock/total_params"] = total_params


def validate_model_for_training(model) -> None:
    """Pre-training validation checks for SFTFlowMidblockModel.

    Raises ValueError if the model is not correctly configured for SFT.
    """
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    non_midblock = [name for name in trainable_names if "midblock" not in name]
    if non_midblock:
        raise ValueError(f"Non-midblock parameters are trainable: {non_midblock}")

    frozen_midblock = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and "midblock" in name
    ]
    if frozen_midblock:
        logger.warning(
            "Midblock parameters are frozen (should be trainable): %s",
            frozen_midblock,
        )

    thinking_level = getattr(model, "thinking_level", None)
    if thinking_level is not None and thinking_level != 32:
        logger.warning(
            "thinking_level=%s, expected 32 for this experiment",
            thinking_level,
        )

    logger.info("Model validation: PASSED")


def estimate_training_budget(
    num_sequences: int,
    seq_length: int,
    thinking_level: int = 32,
    batch_size: int = 1,
    grad_accum: int = 1,
    num_epochs: int = 1,
    steps_per_second_estimate: float = 0.5,
) -> dict:
    """Estimate training time and GPU memory for an SFT run."""
    steps_per_epoch = num_sequences // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs
    total_tokens_per_step = batch_size * grad_accum * seq_length
    total_tokens = total_steps * total_tokens_per_step

    estimated_seconds = total_steps / steps_per_second_estimate
    estimated_hours = estimated_seconds / 3600

    print("\n" + "=" * 60)
    print("TRAINING BUDGET ESTIMATE")
    print("=" * 60)
    print(f"  Sequences:           {num_sequences:,}")
    print(f"  Seq length:          {seq_length}")
    print(f"  Thinking level (T):  {thinking_level}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Grad accumulation:   {grad_accum}")
    print(f"  Effective BS:        {batch_size * grad_accum}")
    print(f"  Epochs:              {num_epochs}")
    print(f"  Steps per epoch:     {steps_per_epoch:,}")
    print(f"  Total steps:         {total_steps:,}")
    print(f"  Tokens per step:     {total_tokens_per_step:,}")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"  Steps/sec (est):     {steps_per_second_estimate:.2f}")
    print(f"  Estimated hours:     {estimated_hours:.2f}")
    print(f"  Estimated days:      {estimated_hours / 24:.2f}")

    return {
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "estimated_hours": estimated_hours,
        "effective_batch_size": batch_size * grad_accum,
        "steps_per_epoch": steps_per_epoch,
    }

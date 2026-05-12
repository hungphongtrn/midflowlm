"""HF Trainer callbacks and utilities for SFT training with FlowMidblock."""

import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MidblockMetricsCallback(TrainerCallback):
    """Custom HF Trainer callback that logs FlowMidblock-specific metrics."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or logs is None:
            return

        if hasattr(model, "midblock"):
            midblock = model.midblock
            grad_norm = 0.0
            param_norm = 0.0
            for parameter in midblock.parameters():
                if parameter.requires_grad:
                    param_norm += parameter.data.norm(2).item() ** 2
                    if parameter.grad is not None:
                        grad_norm += parameter.grad.norm(2).item() ** 2

            logs["midblock/grad_norm"] = grad_norm**0.5
            logs["midblock/param_norm"] = param_norm**0.5
            logs["midblock/total_params"] = sum(
                parameter.numel() for parameter in midblock.parameters() if parameter.requires_grad
            )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "trainable_params"):
            logger.info("Trainable parameters: %s", f"{model.trainable_params:,}")
            if hasattr(model, "frozen_params"):
                logger.info("Frozen parameters:    %s", f"{model.frozen_params:,}")


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

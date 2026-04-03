"""Training module for midflowlm.

This module provides trainers and utilities for iterative midblock training.

Primary Classes:
    Trainer: The default trainer using online calculation (no caching required).
           This is the recommended trainer for all new workflows.

    OnlineNoCacheTrainer: Alias for Trainer (backward compatibility).

Deprecated Classes:
    CachedTrainer: Cache-based trainer (deprecated, kept for backward compatibility).
                   Use Trainer instead which defaults to online_no_cache mode.

Example:
    from src.training import Trainer

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.fit()
"""

from src.training.trainer import Trainer, OnlineNoCacheTrainer
from src.training.losses import DistillationLoss
from src.training.teacher_state import (
    TeacherStateMode,
    resolve_teacher_state_mode,
    get_teacher_state_mode,
    validate_teacher_state_config,
)

# Deprecated import - emits deprecation warning
from src.training.cached_trainer import CachedTrainer

__all__ = [
    "Trainer",
    "OnlineNoCacheTrainer",
    "CachedTrainer",
    "DistillationLoss",
    "TeacherStateMode",
    "resolve_teacher_state_mode",
    "get_teacher_state_mode",
    "validate_teacher_state_config",
]
